from pypdf import PdfReader
from docx import Document as DocxDocument
import pandas as pd
import os


def extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    elif ext == ".docx":
        doc = DocxDocument(file_path)
        return "\n".join(p.text for p in doc.paragraphs)

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    elif ext == ".csv":
        df = pd.read_csv(file_path)
        return df.to_csv(index=False)

    else:
        raise ValueError(f"Unsupported file type: {ext}")
