from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
import numpy as np

from config import DATA_DIR

VEC_DIR = DATA_DIR / "vectors"
VEC_DIR.mkdir(parents=True, exist_ok=True)

def _base(doc_id: str) -> Path:
    return VEC_DIR / doc_id

def _emb_path(doc_id: str) -> Path:
    return _base(doc_id).with_suffix(".npy")

def _meta_path(doc_id: str) -> Path:
    return _base(doc_id).with_suffix(".meta.json")

def index_exists(doc_id: str) -> bool:
    return _emb_path(doc_id).exists() and _meta_path(doc_id).exists()

def save_index(doc_id: str, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]) -> None:
    np.save(_emb_path(doc_id), embeddings.astype(np.float32))
    _meta_path(doc_id).write_text(json.dumps(metadatas, ensure_ascii=False, indent=2), encoding="utf-8")

def load_index(doc_id: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    embs = np.load(_emb_path(doc_id))
    metas = json.loads(_meta_path(doc_id).read_text(encoding="utf-8"))
    return embs, metas

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (d,), b: (N,d)
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return b_norm @ a_norm

def search(doc_id: str, query_emb: np.ndarray, top_k: int = 4) -> List[Dict[str, Any]]:
    embs, metas = load_index(doc_id)
    scores = cosine_sim(query_emb, embs)  # (N,)
    idxs = np.argsort(-scores)[:top_k]
    results = []
    for i in idxs:
        r = metas[i].copy()
        r["score"] = float(scores[i])
        results.append(r)
    return results