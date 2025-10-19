from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

from .base_agent import BaseAgent, AgentContext
from utils.file_utils import (
    load_text_from_pdf,
    load_text_from_docx,
    load_text_from_html,
    load_text_from_txt,
    split_into_paragraphs,
    make_chunks,
)

class ParserAgent(BaseAgent):
    def parse(self, file_path: Path, *, max_chars: int = 2000, overlap_ratio: float = 0.10) -> Dict[str, Any]:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            text = load_text_from_pdf(file_path)
        elif suffix == ".docx":
            text = load_text_from_docx(file_path)
        elif suffix in {".html", ".htm"}:
            text = load_text_from_html(file_path)
        elif suffix == ".txt":
            text = load_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        paragraphs = split_into_paragraphs(text)
        chunks = make_chunks(paragraphs, max_chars=max_chars, overlap_ratio=overlap_ratio)

        return {
            "text": text,
            "paragraphs": paragraphs,
            "chunks": chunks,
            "meta": {
                "num_paragraphs": len(paragraphs),
                "num_chunks": len(chunks),
                "max_chars": max_chars,
                "overlap_ratio": overlap_ratio,
            },
        }