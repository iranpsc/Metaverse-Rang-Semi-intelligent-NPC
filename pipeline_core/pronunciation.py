"""Small, offline pronunciation lexicon; never guess an isolated homograph."""

from __future__ import annotations

import json
import re
from pathlib import Path
from collections.abc import Mapping


DEFAULT_LEXICON = Path(__file__).with_name("persian_pronunciations.json")


class PronunciationLexicon:
    """Replace whole words/phrases once, longest first, preserving explicit vowels.

    User entries override defaults. Empty values disable a default entry. The
    file is loaded at startup, not per sentence; restart after editing it.
    """

    def __init__(self, entries: Mapping[str, str] | None = None, *, path: str | Path | None = None):
        self.entries = self._load(DEFAULT_LEXICON)
        if path is not None:
            self.entries.update(self._load(Path(path)))
        if entries is not None:
            self._validate(entries)
            self.entries.update(entries)
        self.entries = {key: value for key, value in self.entries.items() if value}
        boundary = r"[\w\u064b-\u065f\u200c]"
        alternatives = "|".join(re.escape(key) for key in sorted(self.entries, key=len, reverse=True))
        self.pattern = re.compile(rf"(?<!{boundary})(?:{alternatives})(?!{boundary})") if alternatives else None

    @staticmethod
    def _validate(entries: Mapping[str, str]) -> None:
        if not isinstance(entries, Mapping) or any(
            not isinstance(key, str) or not key.strip() or not isinstance(value, str)
            for key, value in entries.items()
        ):
            raise ValueError("pronunciation lexicon must be a JSON object of nonempty phrases to strings")

    @classmethod
    def _load(cls, path: Path) -> dict[str, str]:
        with path.open(encoding="utf-8") as source:
            entries = json.load(source)
        cls._validate(entries)
        return entries

    def apply(self, text: str) -> str:
        if self.pattern is None:
            return text
        return self.pattern.sub(lambda match: self.entries[match.group()], text)
