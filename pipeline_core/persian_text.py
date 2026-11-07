"""Deterministic Persian text normalization for speech synthesis.

The normalizer deliberately has no network or ML dependency.  It keeps the
original text in memory for callers that need a comparison, but callers should
not log it.  Number expansion is intentionally conservative: technical Latin
tokens are preserved and only unambiguous numeric constructs are verbalized.
"""

from __future__ import annotations

import html
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from urllib.parse import urlsplit

from pipeline_core.pronunciation import PronunciationLexicon


_ONES = (
    "صفر",
    "یک",
    "دو",
    "سه",
    "چهار",
    "پنج",
    "شش",
    "هفت",
    "هشت",
    "نه",
    "ده",
    "یازده",
    "دوازده",
    "سیزده",
    "چهارده",
    "پانزده",
    "شانزده",
    "هفده",
    "هجده",
    "نوزده",
)
_TENS = ("", "", "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود")
_HUNDREDS = ("", "صد", "دویست", "سیصد", "چهارصد", "پانصد", "ششصد", "هفتصد", "هشتصد", "نهصد")
_SCALES = ("", "هزار", "میلیون", "میلیارد", "تریلیون", "کوادریلیون")
_DIGIT_TRANSLATION = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_ARABIC_CHARACTER_TRANSLATION = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ۀ": "هٔ",
        "ة": "ه",
        "ؤ": "و",
        "إ": "ا",
        "أ": "ا",
    }
)
_CURRENCY_NAMES = {
    "$": "دلار",
    "€": "یورو",
    "£": "پوند",
    "ریال": "ریال",
    "تومان": "تومان",
    "دلار": "دلار",
    "یورو": "یورو",
}


@dataclass(frozen=True)
class NormalizedText:
    original: str
    normalized: str


def _under_thousand(value: int) -> str:
    parts: list[str] = []
    hundreds, remainder = divmod(value, 100)
    if hundreds:
        parts.append(_HUNDREDS[hundreds])
    if remainder:
        if remainder < 20:
            parts.append(_ONES[remainder])
        else:
            tens, ones = divmod(remainder, 10)
            parts.append(_TENS[tens])
            if ones:
                parts.append(_ONES[ones])
    return " و ".join(parts)


def integer_to_words(value: int) -> str:
    """Convert a signed integer to Persian words without locale side effects."""

    if value == 0:
        return _ONES[0]
    if value < 0:
        return f"منفی {integer_to_words(-value)}"
    if value >= 1000 ** len(_SCALES):
        # Speaking each digit is safer than returning a misleading scale.
        return " ".join(_ONES[int(digit)] for digit in str(value))

    chunks: list[str] = []
    scale_index = 0
    while value:
        value, chunk = divmod(value, 1000)
        if chunk:
            words = _under_thousand(chunk)
            scale = _SCALES[scale_index]
            chunks.append(f"{words} {scale}".strip())
        scale_index += 1
    return " و ".join(reversed(chunks))


def decimal_to_words(raw_value: str) -> str:
    normalized = raw_value.translate(_DIGIT_TRANSLATION).replace("٬", "").replace(",", "").replace("٫", ".")
    sign = ""
    if normalized.startswith(("-", "−", "+")):
        sign, normalized = ("مثبت " if normalized[0] == "+" else "منفی "), normalized[1:]
    if "." not in normalized:
        if len(normalized) > 1 and normalized.startswith("0"):
            return sign + _digits_to_words(normalized)
        return sign + integer_to_words(int(normalized or "0"))
    integer, fraction = normalized.split(".", 1)
    fraction_words = " ".join(_ONES[int(digit)] for digit in fraction if digit.isdigit())
    return f"{sign}{integer_to_words(int(integer or '0'))} ممیز {fraction_words}".strip()


def _digits_to_words(value: str) -> str:
    return " ".join(_ONES[int(digit)] for digit in value if digit in "0123456789")


# These rules run in specificity order. Each match is protected before the
# next rule so numbers inside a date, identifier, version or URL cannot be
# reinterpreted by the generic number rule.
_NUMBER = r"[+−-]?(?:\d{1,3}(?:[,٬]\d{3})+|\d+)(?:[.٫]\d+)?"
_LEFT = r"(?<![\w.٫/−+-])"
_RIGHT = r"(?![\w٫/]|\.\d)"
_UNITS = {
    "km/h": "کیلومتر بر ساعت", "m/s": "متر بر ثانیه",
    "kg": "کیلوگرم", "km": "کیلومتر", "cm": "سانتی‌متر", "mm": "میلی‌متر",
    "mg": "میلی‌گرم", "ml": "میلی‌لیتر", "mL": "میلی‌لیتر",
    "m²": "متر مربع", "m³": "متر مکعب", "m2": "متر مربع", "m3": "متر مکعب",
    "°C": "درجه سانتی‌گراد", "°F": "درجه فارنهایت",
    "GHz": "گیگاهرتز", "MHz": "مگاهرتز", "Hz": "هرتز",
    "GB": "گیگابایت", "MB": "مگابایت", "KB": "کیلوبایت",
    "g": "گرم", "m": "متر", "s": "ثانیه", "L": "لیتر",
}
_DENOMINATORS = {
    2: "دوم", 3: "سوم", 4: "چهارم", 5: "پنجم", 6: "ششم", 7: "هفتم",
    8: "هشتم", 9: "نهم", 10: "دهم", 100: "صدم", 1000: "هزارم",
}


class _ProtectedText:
    """Private-use placeholders contain no digits and cannot match other rules."""

    def __init__(self, text: str):
        self.values: dict[str, str] = {}
        # Do not let an input private-use character impersonate a placeholder.
        self.next_codepoint = 0xF0000
        self.original = text

    def keep(self, value: str) -> str:
        while chr(self.next_codepoint) in self.original:
            self.next_codepoint += 1
        key = chr(self.next_codepoint)
        self.next_codepoint += 1
        self.values[key] = value
        return key

    def restore(self, text: str) -> str:
        return "".join(self.values.get(character, character) for character in text)


def _speak_date(match: re.Match[str]) -> str:
    year, month, day = (int(match.group(name)) for name in ("year", "month", "day"))
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        return match.group()
    if 1200 <= year <= 1600:
        max_day = 31 if month <= 6 else 30
    else:
        from calendar import monthrange

        if not 1 <= year <= 9999:
            return match.group()
        max_day = monthrange(year, month)[1]
    if day > max_day:
        return match.group()
    return f"سال {integer_to_words(year)}، ماه {integer_to_words(month)}، روز {integer_to_words(day)}"


def _speak_time(match: re.Match[str]) -> str:
    hour, minute = int(match.group("hour")), int(match.group("minute"))
    second = match.group("second")
    if hour > 23 or minute > 59 or (second is not None and int(second) > 59):
        return match.group()
    words = f"ساعت {integer_to_words(hour)}"
    if minute:
        words += f" و {integer_to_words(minute)} دقیقه"
    if second is not None and int(second):
        words += f" و {integer_to_words(int(second))} ثانیه"
    return words


def _expand_numbers(text: str, protected: _ProtectedText) -> str:
    def replace(pattern: str, render) -> None:
        nonlocal text
        text = re.sub(pattern, lambda match: protected.keep(render(match)), text)

    # Explicit labels take priority even when the identifier resembles a date.
    label = (
        r"(?:کد(?:[ \t]+(?:ملی|پستی|پیگیری|تأیید|تایید|امنیتی|ورود))?"
        r"|شناسه(?:[ \t]+(?:پرداخت|تراکنش|سفارش))?"
        r"|شماره[ \t]+(?:سفارش|حساب|کارت|تلفن|تماس|موبایل|پیگیری)"
        r"|شماره|تلفن(?:[ \t]+همراه)?|موبایل)"
    )
    replace(
        rf"(?<!\w)(?P<label>{label}[ \t]*[:：]?[ \t]*)(?P<id>\+?\d(?:[ -]?\d){{1,31}})(?!\w)",
        lambda m: m.group("label").rstrip() + " " + ("مثبت " if m.group("id").startswith("+") else "") + _digits_to_words(m.group("id")),
    )
    replace(
        r"(?<![\w+])(?:\+98[ -]?9(?:[ -]?\d){9}|0098[ -]?9(?:[ -]?\d){9}|09(?:[ -]?\d){9})(?![\w-])",
        lambda m: ("مثبت " if m.group().startswith("+") else "") + _digits_to_words(m.group()),
    )
    replace(
        rf"{_LEFT}(?P<year>\d{{4}})(?P<separator>[/-])(?P<month>\d{{1,2}})(?P=separator)(?P<day>\d{{1,2}}){_RIGHT}",
        _speak_date,
    )
    replace(
        rf"(?<![\w:])(?:ساعت[ \t]+)?(?P<hour>\d{{1,2}}):(?P<minute>\d{{2}})(?::(?P<second>\d{{2}}))?(?![\w:])",
        _speak_time,
    )
    replace(
        rf"{_LEFT}(?P<top>[−-]?\d+)/(?P<bottom>\d+){_RIGHT}",
        lambda m: (
            integer_to_words(int(m.group("top").replace("−", "-"))) + " " + _DENOMINATORS[int(m.group("bottom"))]
            if int(m.group("bottom")) in _DENOMINATORS
            else decimal_to_words(m.group("top")) + " تقسیم بر " + decimal_to_words(m.group("bottom"))
        ),
    )
    # Multi-dot strings are versions/IP addresses, never decimal numbers.
    replace(
        r"(?<![\w.])\d+(?:\.\d+){2,}(?!\w|\.\d)",
        lambda m: " نقطه ".join(decimal_to_words(part) for part in m.group().split(".")),
    )
    replace(
        rf"(?<!\w)(?P<label>نسخه[ \t]+)(?P<version>\d+(?:\.\d+)+)(?!\w|\.\d)",
        lambda m: m.group("label") + " نقطه ".join(decimal_to_words(part) for part in m.group("version").split(".")),
    )
    # A Latin word containing digits is an opaque technical identifier.
    replace(r"(?<!\w)[A-Za-z][A-Za-z0-9_+./-]*\d[A-Za-z0-9_+./-]*(?!\w)", lambda m: m.group())
    currencies = "|".join(re.escape(name) for name in _CURRENCY_NAMES)
    replace(
        rf"(?P<currency>[$€£])[ \t]*(?P<number>{_NUMBER}){_RIGHT}",
        lambda m: decimal_to_words(m.group("number")) + " " + _CURRENCY_NAMES[m.group("currency")],
    )
    replace(
        rf"{_LEFT}(?P<number>{_NUMBER})[ \t]*(?P<currency>{currencies})(?!\w)",
        lambda m: decimal_to_words(m.group("number")) + " " + _CURRENCY_NAMES[m.group("currency")],
    )
    replace(
        rf"{_LEFT}(?P<number>{_NUMBER})[ \t]*[%٪]",
        lambda m: decimal_to_words(m.group("number")) + " درصد",
    )
    units = "|".join(re.escape(unit) for unit in sorted(_UNITS, key=len, reverse=True))
    replace(
        rf"{_LEFT}(?P<number>{_NUMBER})[ \t]*(?P<unit>{units})(?![\w/])",
        lambda m: decimal_to_words(m.group("number")) + " " + _UNITS[m.group("unit")],
    )
    # Every expression is consumed whole, preventing backtracking into a
    # decimal, malformed thousands grouping or a version suffix.
    replace(
        rf"(?<![\w.٫/٬,−+-]){_NUMBER}(?![\w٫/٬]|[.,]\d)",
        lambda m: decimal_to_words(m.group()),
    )
    return text


def _replace_markdown(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`([^`]*)`", r" \1 ", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r" \1 ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r" \1 ", text)
    text = re.sub(r"^\s{0,3}(?:#{1,6}|>|[-*+]\s)\s*", "", text, flags=re.MULTILINE)
    return re.sub(r"[*_~]", "", text)


def _speak_url(match: re.Match[str]) -> str:
    raw = match.group(0).rstrip(".,،؛:!؟)")
    trailing = match.group(0)[len(raw) :]
    parsed = urlsplit(raw if "://" in raw else f"https://{raw}")
    host = (parsed.hostname or raw).replace("www.", "", 1).replace(".", " نقطه ")
    path = parsed.path.strip("/").replace("/", " ")
    spoken = f"{host} {path}".strip()
    return f" {spoken} {trailing} "


def _speak_email(match: re.Match[str]) -> str:
    local, domain = match.group(0).split("@", 1)
    return f" {local} اَت {domain.replace('.', ' نقطه ')} "


def _strip_emoji(text: str) -> str:
    output: list[str] = []
    for character in text:
        category = unicodedata.category(character)
        if character != "°" and (character in {"\ufe0f", "\u200d"} or category in {"So", "Sk"}):
            output.append(" ")
        else:
            output.append(character)
    return "".join(output)


class PersianTextNormalizer:
    def __init__(
        self, *, expand_numbers: bool = True, strip_emoji: bool = True,
        diacritization: str = "lexicon", lexicon_path: str | Path | None = None,
        pronunciations: Mapping[str, str] | None = None,
    ):
        if diacritization not in {"off", "lexicon"}:
            raise ValueError("diacritization must be 'off' or 'lexicon'")
        self.expand_numbers = expand_numbers
        self.strip_emoji = strip_emoji
        self.lexicon = PronunciationLexicon(pronunciations, path=lexicon_path) if diacritization == "lexicon" else None

    @classmethod
    def from_env(cls) -> "PersianTextNormalizer":
        return cls(
            diacritization=os.getenv("TTS_DIACRITIZATION", "lexicon").strip().lower(),
            lexicon_path=os.getenv("TTS_PRONUNCIATION_LEXICON", "").strip() or None,
        )

    def normalize(self, value: str) -> NormalizedText:
        original = str(value or "")
        # NFC preserves squared/cubed unit symbols and existing vowel marks.
        text = unicodedata.normalize("NFC", html.unescape(original))
        text = text.translate(_ARABIC_CHARACTER_TRANSLATION)
        text = re.sub(r"[\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]", "", text)
        text = text.replace("ـ", "")
        text = _replace_markdown(text)
        text = text.translate(_DIGIT_TRANSLATION)
        protected = _ProtectedText(text)

        # Preserve meaningful Latin technical words while making URLs/e-mails
        # pronounceable instead of deleting them.
        text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", lambda m: protected.keep(_speak_email(m).strip()), text)
        text = re.sub(r"(?:https?://|www\.)[^\s<>]+", lambda m: protected.keep(_speak_url(m).strip()), text, flags=re.IGNORECASE)
        text = re.sub(r"\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?:/[^\s<>]*)?", lambda m: protected.keep(_speak_url(m).strip()), text)

        if self.strip_emoji:
            text = _strip_emoji(text)

        # Custom names (including GPT-4 or model numbers) take precedence over
        # numeric interpretation. URLs and emails remain protected.
        if self.lexicon is not None:
            text = self.lexicon.apply(text)

        if self.expand_numbers:
            text = _expand_numbers(text, protected)

        text = text.replace("?", "؟").replace(";", "؛")
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        text = re.sub(r"\s*\n+\s*", " ", text)
        text = re.sub(r"\s*\u200c\s*", "\u200c", text)
        text = re.sub(r"\u200c{2,}", "\u200c", text)
        text = re.sub(r"\s+([،؛:!؟.])", r"\1", text)
        text = re.sub(r"([،؛:!؟])(?=\S)", r"\1 ", text)
        text = re.sub(r"(?<!\d)\.(?=[^\s\d])", ". ", text)
        text = protected.restore(text)
        text = re.sub(r"\s+([،؛:!؟.])", r"\1", text)
        return NormalizedText(original=original, normalized=text.strip())


def main() -> None:
    """Preview exactly the speech text without loading Piper or any model."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Offline Persian text preparation for TTS")
    parser.add_argument("text", nargs="?", help="text to normalize; otherwise read stdin")
    parser.add_argument("--diacritization", choices=("off", "lexicon"), default=os.getenv("TTS_DIACRITIZATION", "lexicon"))
    parser.add_argument("--lexicon", default=os.getenv("TTS_PRONUNCIATION_LEXICON") or None)
    parser.add_argument("--json", action="store_true", help="include original and normalized text")
    arguments = parser.parse_args()
    try:
        normalizer = PersianTextNormalizer(diacritization=arguments.diacritization, lexicon_path=arguments.lexicon)
        result = normalizer.normalize(arguments.text if arguments.text is not None else sys.stdin.read())
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if arguments.json:
        print(json.dumps({"original": result.original, "normalized": result.normalized}, ensure_ascii=False))
    else:
        print(result.normalized)


if __name__ == "__main__":
    main()
