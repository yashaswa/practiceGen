from __future__ import annotations
from typing import List

from .base_agent import BaseAgent
from config import llm

SUM_PROMPT = (
    "You are a careful technical summarizer. Given a document chunk, write a concise, faithful summary "
    "(4-6 bullet points). Do not add facts not present. Keep names, dates, and key entities.\n\n"
    "Chunk:\n{chunk}\n\nSummary (bulleted):"
)

class SummarizerAgent(BaseAgent):
    def summarize_chunks(self, chunks: List[str]) -> List[str]:
        outputs: List[str] = []
        for i, chunk in enumerate(chunks, start=1):
            prompt = SUM_PROMPT.format(chunk=chunk)
            try:
                text = llm.invoke(prompt)  # LangChain LLM
            except Exception as e:
                text = f"- [error] Failed to summarize chunk {i}: {e}"
            outputs.append(str(text))
        return outputs
    

    def assemble_sectioned(self, summaries: List[str]) -> str:
        lines = []
        for idx, s in enumerate(summaries, start=1):
            lines.append(f"### Section {idx}\n{s}\n")
        return "\n".join(lines).strip()