"""Web-page extraction helpers that preserve tabular knowledge."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import trafilatura


_WHITESPACE = re.compile(r"\s+")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _cell_text(cell: ET.Element) -> str:
    return _WHITESPACE.sub(" ", " ".join(cell.itertext())).strip()


def format_tables_from_xml(extracted_xml: str | None) -> str:
    """Return every extracted table cell as header-labelled, searchable rows.

    Trafilatura's plain-text table output retains the visual pipe-separated
    rows, but a row may be indexed in a different chunk from its header.  This
    representation repeats the first row's column names for every later row so
    both retrieval and the LLM can understand each cell independently.
    """

    if not extracted_xml:
        return ""

    try:
        root = ET.fromstring(extracted_xml)
    except ET.ParseError:
        return ""

    rendered_tables: list[str] = []
    tables = [element for element in root.iter() if _local_name(element.tag) == "table"]
    for table_number, table in enumerate(tables, start=1):
        rows: list[list[str]] = []
        for row in table.iter():
            if _local_name(row.tag) != "row":
                continue
            cells = [
                _cell_text(cell)
                for cell in list(row)
                if _local_name(cell.tag) == "cell"
            ]
            if any(cells):
                rows.append(cells)

        if not rows:
            continue

        headers = rows[0]
        rendered_rows: list[str] = []
        for row_number, cells in enumerate(rows, start=1):
            if row_number > 1 and len(cells) == len(headers) and all(headers):
                values = [f"{header}: {value}" for header, value in zip(headers, cells)]
            else:
                values = [f"ستون {index}: {value}" for index, value in enumerate(cells, start=1)]
            rendered_rows.append(
                f"جدول {table_number}، ردیف {row_number}: " + " | ".join(values)
            )

        rendered_tables.append("\n".join(rendered_rows))

    return "\n\n".join(rendered_tables)


def extract_web_content(downloaded: str) -> str | None:
    """Extract article text plus every table row and column from downloaded HTML."""

    full_text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True,
    )
    if not full_text or not full_text.strip():
        return None

    extracted_xml = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True,
        output_format="xml",
    )
    structured_tables = format_tables_from_xml(extracted_xml)
    if structured_tables:
        return f"{full_text.strip()}\n\nجدول‌های ساختاریافته:\n{structured_tables}"
    return full_text.strip()
