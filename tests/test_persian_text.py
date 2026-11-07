import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline_core.persian_text import PersianTextNormalizer, integer_to_words


class PersianTextNormalizerTests(unittest.TestCase):
    def setUp(self):
        self.normalizer = PersianTextNormalizer()

    def test_normalizes_arabic_characters_and_preserves_half_space(self):
        result = self.normalizer.normalize("مي روم و كتاب‌ها")
        self.assertEqual(result.normalized, "می روم و کتاب‌ها")

    def test_expands_dates_times_currency_and_percent(self):
        result = self.normalizer.normalize("۱۴۰۳/۰۷/۱۲ ساعت 09:05 مبلغ ۲۵.۵ تومان و 20%")
        self.assertIn("سال یک هزار و چهارصد و سه", result.normalized)
        self.assertIn("ساعت نه و پنج دقیقه", result.normalized)
        self.assertIn("بیست و پنج ممیز پنج تومان", result.normalized)
        self.assertIn("بیست درصد", result.normalized)

    def test_keeps_technical_words_and_handles_links(self):
        result = self.normalizer.normalize("API در https://example.com/docs و dev@example.com")
        self.assertIn("API", result.normalized)
        self.assertIn("example نقطه com", result.normalized)
        self.assertIn("dev اَت example نقطه com", result.normalized)

    def test_removes_markdown_and_emoji(self):
        result = self.normalizer.normalize("**سلام** [راهنما](https://example.com) 🙂")
        self.assertEqual(result.normalized, "سلام راهنما")

    def test_integer_words(self):
        self.assertEqual(integer_to_words(1200456), "یک میلیون و دویست هزار و چهارصد و پنجاه و شش")

    def test_numeric_formats(self):
        cases = {
            "۱۲۵٬۰۰۰ تومان": "صد و بیست و پنج هزار تومان",
            "1,234,567 تومان": "یک میلیون و دویست و سی و چهار هزار و پانصد و شصت و هفت تومان",
            "۲۰٪": "بیست درصد",
            "−۵°C": "منفی پنج درجه سانتی‌گراد",
            "+۱۲": "مثبت دوازده",
            "۳٫۱۴": "سه ممیز یک چهار",
            "۰.۰۵": "صفر ممیز صفر پنج",
            "۳/۴": "سه چهارم",
            "۱/۲": "یک دوم",
            "۱/۲.": "یک دوم.",
            "۵/۰": "پنج تقسیم بر صفر",
            "۲.۵kg": "دو ممیز پنج کیلوگرم",
            "۸۰km/h": "هشتاد کیلومتر بر ساعت",
            "۱۲m²": "دوازده متر مربع",
            "۱۲m³": "دوازده متر مکعب",
            "۲GB": "دو گیگابایت",
            "$12.50": "دوازده ممیز پنج صفر دلار",
            "42.": "چهل و دو.",
            "3.14.": "سه ممیز یک چهار.",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(self.normalizer.normalize(raw).normalized, expected)

    def test_identifiers_keep_zeroes_and_are_not_cardinal_numbers(self):
        cases = {
            "۰۰۴۲": "صفر صفر چهار دو",
            "کد ۰۰۴۲": "کد صفر صفر چهار دو",
            "کدملی ۰۰۴۲": "کدملی صفر صفر چهار دو",
            "شناسه پرداخت ۰۰۴۲": "شناسه پرداخت صفر صفر چهار دو",
            "۰۹۱۲۱۲۳۴۵۶۷": "صفر نه یک دو یک دو سه چهار پنج شش هفت",
            "شماره تلفن ۰۲۱-۱۲۳۴۵۶۷۸": "شماره تلفن صفر دو یک یک دو سه چهار پنج شش هفت هشت",
            "شماره کارت ۶۰۳۷-۹۹۱۲-۳۴۵۶-۷۸۹۰": "شماره کارت شش صفر سه هفت نه نه یک دو سه چهار پنج شش هفت هشت نه صفر",
            "کد ۱۴۰۳-۰۷-۱۲": "کد یک چهار صفر سه صفر هفت یک دو",
            "+۹۸۹۱۲۱۲۳۴۵۶۷": "مثبت نه هشت نه یک دو یک دو سه چهار پنج شش هفت",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(self.normalizer.normalize(raw).normalized, expected)

    def test_time_is_not_prefixed_twice_and_supports_seconds(self):
        cases = {
            "ساعت ۱۴:۳۰": "ساعت چهارده و سی دقیقه",
            "۱۴:۰۰": "ساعت چهارده",
            "00:00:05": "ساعت صفر و پنج ثانیه",
            "۰۹:۰۵:۰۲": "ساعت نه و پنج دقیقه و دو ثانیه",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(self.normalizer.normalize(raw).normalized, expected)

    def test_invalid_dates_times_and_malformed_numbers_are_not_invented(self):
        for text in ["2024-02-30", "1403/13/40", "1403/07/31", "25:90", "12,34", "1٫2٫3"]:
            with self.subTest(text=text):
                self.assertEqual(self.normalizer.normalize(text).normalized, text)

    def test_versions_and_technical_identifiers(self):
        self.assertEqual(self.normalizer.normalize("GPT-4 و v2.5").normalized, "GPT-4 و v2.5")
        self.assertEqual(self.normalizer.normalize("نسخه 1.2").normalized, "نسخه یک نقطه دو")
        self.assertEqual(self.normalizer.normalize("نسخه 1.2.3").normalized, "نسخه یک نقطه دو نقطه سه")
        self.assertEqual(self.normalizer.normalize("نسخه 1.2.3.").normalized, "نسخه یک نقطه دو نقطه سه.")
        self.assertEqual(self.normalizer.normalize("192.168.1.10").normalized, "صد و نود و دو نقطه صد و شصت و هشت نقطه یک نقطه ده")

    def test_numbers_remain_separate_at_punctuation(self):
        self.assertEqual(self.normalizer.normalize("۱۲، ۱۳ و ۱۴.").normalized, "دوازده، سیزده و چهارده.")

    def test_diacritics_and_ambiguous_words(self):
        self.assertEqual(self.normalizer.normalize("مَرد مُرد کرم شیر").normalized, "مَرد مُرد کرم شیر")
        self.assertEqual(self.normalizer.normalize("کرم خاکی و کرم ضد آفتاب").normalized, "کِرمِ خاکی و کِرِمِ ضدِ آفتاب")
        self.assertEqual(self.normalizer.normalize("کتابِ من").normalized, "کتابِ من")
        self.assertEqual(self.normalizer.normalize("گل رز").normalized, "گُلِ رُز")
        self.assertEqual(self.normalizer.normalize("ضدکرم خاکی").normalized, "ضدکرم خاکی")

    def test_diacritization_can_be_disabled_without_removing_explicit_marks(self):
        normalizer = PersianTextNormalizer(diacritization="off")
        self.assertEqual(normalizer.normalize("کرم خاکی و کتابِ من و ۲۰٪").normalized, "کرم خاکی و کتابِ من و بیست درصد")

    def test_number_expansion_can_be_disabled(self):
        self.assertEqual(PersianTextNormalizer(expand_numbers=False).normalize("۱۲٫۵٪").normalized, "12٫5٪")

    def test_custom_lexicon_is_loaded_and_overrides_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lexicon.json"
            path.write_text(json.dumps({"کرم خاکی": "کِرم خاکی", "GPT-4": "جی پی تی چهار", "CPU": ""}), encoding="utf-8")
            normalizer = PersianTextNormalizer(lexicon_path=path)
            self.assertEqual(normalizer.normalize("کرم خاکی GPT-4 CPU").normalized, "کِرم خاکی جی پی تی چهار CPU")

    def test_lexicon_prefers_phrases_and_does_not_cascade(self):
        normalizer = PersianTextNormalizer(pronunciations={"الف": "ب", "الف ب": "ج", "ب": "د"})
        self.assertEqual(normalizer.normalize("الف ب الف").normalized, "ج ب")

    def test_lexicon_rejects_invalid_config(self):
        for entries in [{"": "الف"}, {"الف": 2}, ["الف"]]:
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                PersianTextNormalizer(pronunciations=entries)
        with self.assertRaises(ValueError):
            PersianTextNormalizer(diacritization="full")

    def test_environment_configuration(self):
        with patch.dict(os.environ, {"TTS_DIACRITIZATION": "off", "TTS_PRONUNCIATION_LEXICON": ""}):
            self.assertEqual(PersianTextNormalizer.from_env().normalize("کرم خاکی").normalized, "کرم خاکی")

    def test_unicode_controls_and_original_text(self):
        raw = "\u202eكِتاب\u200f‌ها ۱۲۳"
        result = self.normalizer.normalize(raw)
        self.assertEqual(result.original, raw)
        self.assertEqual(result.normalized, "کِتاب‌ها صد و بیست و سه")

    def test_private_use_input_cannot_impersonate_internal_placeholders(self):
        result = self.normalizer.normalize("\U000f0000 ۱۲ و ۱۳")
        self.assertEqual(result.normalized, "\U000f0000 دوازده و سیزده")

    def test_normalization_is_idempotent(self):
        for text in ["کرم خاکی ۲۰٪", "۰۹۱۲۱۲۳۴۵۶۷", "۱۴۰۳/۰۷/۱۲ ساعت 14:30", "GPT-4", "۳٫۱۴", "۱۲m²"]:
            with self.subTest(text=text):
                first = self.normalizer.normalize(text).normalized
                self.assertEqual(self.normalizer.normalize(first).normalized, first)
