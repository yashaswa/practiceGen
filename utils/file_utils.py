from __future__ import annotations
from pathlib import Path
from bs4 import BeautifulSoup
import pdfplumber
import docx
import re

WHITESPACE_RE = re.compile(r"\s+")

def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()

def load_text_from_pdf(path: Path) -> str:
    parts = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return clean_text("\n\n".join(parts))

def load_text_from_docx(path: Path) -> str:
    d = docx.Document(str(path))
    parts = []
    for para in d.paragraphs:
        parts.append(para.text)
    return clean_text("\n\n".join(parts))

def load_text_from_html(path: Path) -> str:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text("\n")
    return clean_text(text)

def load_text_from_txt(path: Path) -> str:
    return clean_text(path.read_text(encoding="utf-8", errors="ignore"))

def split_into_paragraphs(text: str) -> list[str]:
    raw = re.split(r"\n\s*\n", text)
    return [p.strip() for p in raw if p.strip()]

def make_chunks(paragraphs: list[str], max_chars: int = 2000, overlap_ratio: float = 0.10) -> list[str]:
    """
    Build length-based chunks from paragraphs with ~10% character overlap between
    consecutive chunks. Overlap is computed on the produced chunk text.
    """
    chunks: list[str] = []
    buf: list[str] = []
    current_len = 0

    for p in paragraphs:
        p_len = len(p)
        if current_len + p_len + 2 <= max_chars:  # +2 for joining newlines
            buf.append(p)
            current_len += p_len + 2
        else:
            if buf:
                chunk = "\n\n".join(buf).strip()
                chunks.append(chunk)
                overlap = int(len(chunk) * overlap_ratio)
                if overlap > 0:
                    tail = chunk[-overlap:]
                    buf = [tail, p]
                    current_len = len(tail) + 2 + p_len
                else:
                    buf = [p]
                    current_len = p_len
            else:
                # single paragraph longer than max; hard split
                hard = [p[i:i+max_chars] for i in range(0, len(p), max_chars)]
                for i, piece in enumerate(hard):
                    if i == 0:
                        chunks.append(piece)
                    else:
                        prev = chunks[-1]
                        overlap = int(len(prev) * overlap_ratio)
                        seed = prev[-overlap:] if overlap > 0 else ""
                        chunks.append((seed + piece).strip())
                buf = []
                current_len = 0

    if buf:
        chunks.append("\n\n".join(buf).strip())
    return chunks