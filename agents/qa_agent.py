# agents/qa_agent.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np

from .base_agent import BaseAgent ,AgentContext;
from config import embeddings, llm
from utils.vecstore import save_index, search, index_exists
from utils.storage import load_processed

BUILD_PROMPT = (
    "You are a precise assistant. Answer the user's question ONLY using the provided context. "
    "Cite the chunk numbers you used as [C{chunk_index}] inline where relevant. "
    "If the answer cannot be found, say \"I couldn't find that in the document.\" "
    "Be concise and faithful.\n\n"
    "Question:\n{question}\n\n"
    "Context (multiple chunks):\n{context}\n\n"
    "Answer:"
)

def _format_context(chunks_with_idx: List[Dict[str, Any]], max_chars: int = 3000) -> str:
    parts, count = [], 0
    for r in chunks_with_idx:
        prefix = f"[Chunk {r['chunk_index']}] "
        text = r["text"]
        piece = prefix + text
        if count + len(piece) > max_chars and parts:
            break
        parts.append(piece)
        count += len(piece)
    return "\n\n---\n\n".join(parts)

class QAAgent(BaseAgent):
    def build_index(self, doc_id: str) -> Dict[str, Any]:
        proc = load_processed(doc_id)
        if not proc or "chunks" not in proc:
            raise ValueError("Document not parsed/summarized yet; no chunks to index.")
        if index_exists(doc_id):
            return {"indexed": True, "already_existed": True, "num_chunks": len(proc["chunks"])}

        chunks: List[str] = proc["chunks"]
        # Embed all chunks
        vecs = embeddings.embed_documents(chunks)  # List[List[float]]
        mat = np.array(vecs, dtype=np.float32)     # (N, d)
        metas = [{"chunk_index": i+1, "text": c[:2000]} for i, c in enumerate(chunks)]  # trim to 2k chars per item
        save_index(doc_id, mat, metas)
        return {"indexed": True, "already_existed": False, "num_chunks": len(chunks)}

    def answer(self, doc_id: str, question: str, top_k: int = 4) -> Dict[str, Any]:
        # Ensure index exists (caller should have built; but be safe)
        if not index_exists(doc_id):
            _ = self.build_index(doc_id)

        q_emb = np.array(embeddings.embed_query(question), dtype=np.float32)
        retrieved = search(doc_id, q_emb, top_k=top_k)  # [{chunk_index, text, score}, ...]

        # Build prompt
        ctx_str = _format_context(retrieved)
        prompt = BUILD_PROMPT.format(question=question, context=ctx_str)

        try:
            raw = llm.invoke(prompt)
            answer = str(raw)
        except Exception as e:
            answer = f"[error] LLM failed to answer: {e}"

        return {
            "answer": answer,
            "sources": [{"chunk_index": r["chunk_index"], "score": r["score"]} for r in retrieved],
        }


