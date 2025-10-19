from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))

HF_REPO_ID = os.getenv("HF_REPO_ID", "HuggingFaceTB/SmolLM3-3B")
HF_TASK = os.getenv("HF_TASK", "text-generation")
HF_EMBEDDINGS_MODEL = os.getenv(
"HF_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDINGS_MODEL)
llm = HuggingFaceEndpoint(repo_id=HF_REPO_ID, task=HF_TASK)

__all__ = [
"BASE_DIR",
"DATA_DIR",
"UPLOAD_DIR",
"BACKEND_HOST",
"BACKEND_PORT",
"embeddings",
"llm",
]