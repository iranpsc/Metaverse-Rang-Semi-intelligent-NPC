import unittest
from unittest.mock import call, patch

try:
    import pandas as pd

    from LLM.rag_system import VectorStoreManager
    from LLM.web_extraction import extract_web_content, format_tables_from_xml

    RAG_DEPENDENCIES_AVAILABLE = True
except ModuleNotFoundError:
    # The lightweight Django/web image intentionally excludes the model and
    # crawler packages. These tests run in the LLM image where those packages
    # are installed, while remaining safely discoverable by Django's suite.
    RAG_DEPENDENCIES_AVAILABLE = False


TABLE_XML = """\
<doc><main><table>
  <row><cell>برند</cell><cell>مدل گوشی</cell><cell>حافظه</cell><cell>قیمت امروز</cell></row>
  <row><cell>سامسونگ</cell><cell>Galaxy A57</cell><cell>128GB</cell><cell>162.000.000</cell></row>
  <row><cell>شیائومی</cell><cell>Redmi 15</cell><cell>256GB</cell><cell>75.000.000</cell></row>
</table></main></doc>
"""


@unittest.skipUnless(RAG_DEPENDENCIES_AVAILABLE, "LLM/RAG dependencies are not installed")
class WebTableExtractionTests(unittest.TestCase):
    def test_formats_every_table_row_and_column_with_headers(self):
        result = format_tables_from_xml(TABLE_XML)

        self.assertIn("ستون 1: برند", result)
        self.assertIn("مدل گوشی: Galaxy A57", result)
        self.assertIn("حافظه: 128GB", result)
        self.assertIn("قیمت امروز: 162.000.000", result)
        self.assertIn("مدل گوشی: Redmi 15", result)

    @patch("LLM.web_extraction.trafilatura.extract")
    def test_web_extraction_requests_tables_and_appends_structured_rows(self, mocked_extract):
        mocked_extract.side_effect = ["متن مقاله\nGalaxy A57 | 162.000.000", TABLE_XML]

        result = extract_web_content("<html>downloaded</html>")

        self.assertIn("متن مقاله", result)
        self.assertIn("مدل گوشی: Galaxy A57", result)
        self.assertEqual(mocked_extract.call_count, 2)
        self.assertEqual(
            mocked_extract.call_args_list,
            [
                call("<html>downloaded</html>", include_comments=False, include_tables=True),
                call(
                    "<html>downloaded</html>",
                    include_comments=False,
                    include_tables=True,
                    output_format="xml",
                ),
            ],
        )


@unittest.skipUnless(RAG_DEPENDENCIES_AVAILABLE, "LLM/RAG dependencies are not installed")
class VectorStoreChunkingTests(unittest.TestCase):
    def test_long_table_content_is_split_without_losing_late_rows(self):
        early_rows = "\n".join(f"مدل گوشی: Model-{index} | قیمت امروز: {index}" for index in range(80))
        content = f"{early_rows}\nمدل گوشی: Galaxy A57 | قیمت امروز: 162.000.000"
        frame = pd.DataFrame(
            [{"title": "phone-prices", "link": "https://example.test", "content": content}]
        )
        manager = VectorStoreManager(embeddings=None, user_id="phone-prices")

        with patch.dict("os.environ", {"RAG_CHUNK_SIZE": "400", "RAG_CHUNK_OVERLAP": "40"}):
            documents = manager._create_documents(frame)

        self.assertGreater(len(documents), 1)
        self.assertTrue(any("Galaxy A57" in document.page_content for document in documents))
        self.assertTrue(any("162.000.000" in document.page_content for document in documents))
        self.assertTrue(all(document.metadata["user_id"] == "phone-prices" for document in documents))
