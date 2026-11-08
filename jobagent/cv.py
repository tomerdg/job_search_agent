from __future__ import annotations

import os
import re
from pathlib import Path


def load_cv(path: str | Path) -> str:
    """CV text from the CV_TEXT env var, a .md/.txt file, or a PDF. Phone/email are stripped."""
    text = os.environ.get("CV_TEXT", "")
    if not text:
        p = Path(path)
        if p.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            text = "\n".join(page.extract_text() or "" for page in PdfReader(str(p)).pages)
        else:
            text = p.read_text(encoding="utf-8")
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[email]", text)
    text = re.sub(r"\+?\d[\d\- ]{7,}\d", "[phone]", text)
    return text.strip()
