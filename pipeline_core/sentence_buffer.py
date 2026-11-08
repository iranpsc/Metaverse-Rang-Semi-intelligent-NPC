"""Streaming sentence boundaries that leave numbers and technical tokens intact."""

from __future__ import annotations

import re


_SPEECH_ATOM = re.compile(
    r"(?:(?:کد|شناسه|شماره|تلفن|موبایل|ساعت)"
    r"(?:[ \t]+[آ-ی\u200c]+){0,2}[ \t:]+)?"
    r"[+−-]?\d(?:[ \t]*[\d.,٫٬:/−+-])*"
    r"(?:[ \t]*(?:[A-Za-z°²³/]+|تومان|ریال|دلار|یورو|[%٪]))?"
)


class SentenceBuffer:
    def __init__(self, max_chars: int = 280):
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        self.text = ""
        self.max_chars = max_chars

    def _sentence_end(self) -> int | None:
        for match in re.finditer(r"[.!?؟\n]", self.text):
            index = match.start()
            character = match.group()
            if character == ".":
                # A token ending in '3.' may continue as '3.14'. Wait for
                # lookahead (or finish) before committing to that boundary.
                if index + 1 == len(self.text):
                    continue
                before = self.text[index - 1] if index else ""
                after = self.text[index + 1]
                if after == "." or (before.isdigit() and after.isdigit()):
                    continue
                if not after.isspace() and (
                    (before.isascii() and before.isalnum())
                    or (after.isascii() and after.isalnum())
                ):
                    continue
            # '?' and '!' can also occur inside a URL. A token starting with
            # an ASCII letter or scheme must finish before being emitted.
            if character in "!?":
                token = re.split(r"\s", self.text[:index + 1])[-1]
                if "://" in token or token.startswith("www."):
                    continue
            return match.end()
        return None

    def push(self, chunk: str) -> list[str]:
        self.text = (self.text + chunk).replace("[END]", "")
        ready: list[str] = []
        while self.text:
            end = self._sentence_end()
            if end is None and len(self.text) >= self.max_chars:
                boundaries = list(re.finditer(r"\s+", self.text))
                atoms = list(_SPEECH_ATOM.finditer(self.text))
                # max_chars is a soft limit: never slice a word, URL, decimal
                # or a space-separated phone/card number to meet the budget.
                safe = [
                    match.start() for match in boundaries
                    if match.start() > 0 and not any(
                        atom.start() < match.start() < atom.end() for atom in atoms
                    )
                ]
                before = [position for position in safe if position <= self.max_chars]
                end = before[-1] if before else (safe[0] if safe else None)
            if end is None:
                break
            sentence = self.text[:end].strip()
            self.text = self.text[end:].lstrip()
            if sentence:
                ready.append(sentence)
        return ready

    def finish(self) -> str:
        value = self.text.replace("[END]", "").strip()
        self.text = ""
        return value
