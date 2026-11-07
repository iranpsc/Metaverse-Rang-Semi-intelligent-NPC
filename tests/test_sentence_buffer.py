import unittest

from pipeline_core.persian_text import PersianTextNormalizer
from pipeline_core.sentence_buffer import SentenceBuffer


class SentenceBufferTests(unittest.TestCase):
    def test_all_two_chunk_splits_preserve_numeric_meaning(self):
        normalizer = PersianTextNormalizer()
        for text in [
            "عدد ۳.۱۴ است.", "عدد 3.14 و ۱۲٫۵ درصد است.",
            "جلسه ۱۴۰۳/۰۷/۱۲ ساعت ۱۴:۳۰ است.",
            "شماره تلفن ۰۹۱۲ ۱۲۳ ۴۵۶۷ است.",
            "نسخه v1.2.3 و GPT-4 آماده است.",
            "راهنما https://example.com/a?x=3.14 است.",
        ]:
            expected = normalizer.normalize(text).normalized
            for split in range(len(text) + 1):
                with self.subTest(text=text, split=split):
                    buffer = SentenceBuffer()
                    output = buffer.push(text[:split]) + buffer.push(text[split:])
                    tail = buffer.finish()
                    if tail:
                        output.append(tail)
                    self.assertEqual(" ".join(normalizer.normalize(part).normalized for part in output), expected)

    def test_character_by_character_streaming(self):
        buffer = SentenceBuffer()
        output = []
        for character in "عدد 3.14 است. خوبه؟ بله!":
            output.extend(buffer.push(character))
        output.append(buffer.finish())
        self.assertEqual(output, ["عدد 3.14 است.", "خوبه؟", "بله!", ""])

    def test_period_looks_ahead_before_emitting(self):
        buffer = SentenceBuffer()
        self.assertEqual(buffer.push("مقدار ۳."), [])
        self.assertEqual(buffer.push("۱۴ است. "), ["مقدار ۳.۱۴ است."])

    def test_soft_limit_preserves_long_tokens_and_grouped_phone_numbers(self):
        buffer = SentenceBuffer(max_chars=10)
        self.assertEqual(buffer.push("۰۹۱۲ ۱۲۳ ۴۵۶۷"), [])
        self.assertEqual(buffer.push(" تمام"), ["۰۹۱۲ ۱۲۳ ۴۵۶۷"])
        buffer = SentenceBuffer(max_chars=5)
        self.assertEqual(buffer.push("https://example.com"), [])
        self.assertEqual(buffer.push(" پایان"), ["https://example.com"])

    def test_end_marker_can_arrive_in_multiple_chunks(self):
        buffer = SentenceBuffer()
        self.assertEqual(buffer.push("سلام [EN"), [])
        self.assertEqual(buffer.push("D]"), [])
        self.assertEqual(buffer.finish(), "سلام")

    def test_soft_limit_keeps_numeric_labels_and_units_with_their_values(self):
        for text, expected in [
            ("کد پیگیری ۱۲۳۴۵ پایان", "کد پیگیری ۱۲۳۴۵"),
            ("شماره کارت ۱۲۳۴ ۵۶۷۸ پایان", "شماره کارت ۱۲۳۴ ۵۶۷۸"),
            ("۲.۵ kg پایان", "۲.۵ kg"),
            ("۱۲۵۰۰ تومان پایان", "۱۲۵۰۰ تومان"),
        ]:
            with self.subTest(text=text):
                buffer = SentenceBuffer(max_chars=5)
                self.assertEqual(buffer.push(text), [expected])
                self.assertEqual(buffer.finish(), "پایان")

    def test_buffer_reuse(self):
        buffer = SentenceBuffer()
        self.assertEqual(buffer.push("اول؟ دوم؟"), ["اول؟", "دوم؟"])
        self.assertEqual(buffer.finish(), "")
        self.assertEqual(buffer.push("سوم"), [])
        self.assertEqual(buffer.finish(), "سوم")
