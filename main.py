from __future__ import annotations
import uuid
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import UPLOAD_DIR
from utils.validation import validate_suffix
from utils.storage import save_processed, load_processed
from agents.base_agent import AgentContext
from agents.parser_agent import ParserAgent
from agents.summarizer_agent import SummarizerAgent
from agents.entity_agent import EntityAgent
from agents.qa_agent import QAAgent
from agents.critic_agent import CriticAgent

app = FastAPI(title="GenAI Doc System — Full")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FILES: Dict[str, Path] = {}

# ===== Models =====
class EditPayload(BaseModel):
    edited_summary: str

class QAPayload(BaseModel):
    question: str
    top_k: int | None = 4

class EntitiesPayload(BaseModel):
    names: list[str] | None = None
    dates: list[str] | None = None
    orgs: list[str] | None = None


# ===== Health =====
@app.get("/health")
def health():
    return {"status": "ok", "service": "genai-docs", "version": "full"}


# ===== Upload =====
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    suffix = Path(file.filename).suffix
    try:
        validate_suffix(Path(file.filename))
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e))

    doc_id = str(uuid.uuid4())
    out_path = UPLOAD_DIR / f"{doc_id}__{file.filename}"
    content = await file.read()
    out_path.write_bytes(content)
    FILES[doc_id] = out_path

    save_processed(doc_id, {"doc_id": doc_id, "filename": file.filename, "status": "uploaded"})
    return {"doc_id": doc_id, "filename": file.filename, "bytes": len(content)}


# ===== Parse + Summarize + Entities =====
@app.get("/summary/{doc_id}")
async def build_or_get_summary(doc_id: str, max_chars: int = 2000, overlap_ratio: float = 0.10):
    proc = load_processed(doc_id)
    if proc and proc.get("sectioned_summary"):
        return {
            "doc_id": doc_id,
            "filename": proc.get("filename"),
            "num_chunks": len(proc.get("chunks", [])),
            "sectioned_summary": proc["sectioned_summary"],
            "edited_summary": proc.get("edited_summary"),
            "entities": proc.get("entities"),
        }

    path = FILES.get(doc_id)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="doc_id not found; upload first")

    ctx = AgentContext(doc_id=doc_id, filename=path.name.split("__", 1)[-1])
    parser = ParserAgent(context=ctx)
    parsed = parser.parse(path, max_chars=max_chars, overlap_ratio=overlap_ratio)

    summarizer = SummarizerAgent(context=ctx)
    summaries = summarizer.summarize_chunks(parsed["chunks"])
    sectioned = summarizer.assemble_sectioned(summaries)

    ent_agent = EntityAgent(context=ctx)
    entities = ent_agent.extract(parsed["text"])

    payload = {
        "doc_id": doc_id,
        "filename": ctx.filename,
        "meta": parsed.get("meta", {}),
        "chunks": parsed["chunks"],
        "sectioned_summary": sectioned,
        "edited_summary": None,
        "entities": entities,
        "status": "summarized",
    }
    save_processed(doc_id, payload)
    return {
        "doc_id": doc_id,
        "filename": ctx.filename,
        "num_chunks": len(parsed["chunks"]),
        "sectioned_summary": sectioned,
        "edited_summary": None,
        "entities": entities,
    }


# ===== Save Edited Summary =====
@app.put("/summary/{doc_id}")
async def save_edited_summary(doc_id: str, body: EditPayload):
    proc = load_processed(doc_id)
    if not proc:
        raise HTTPException(status_code=404, detail="doc not processed yet")
    proc["edited_summary"] = body.edited_summary
    save_processed(doc_id, proc)
    return {"doc_id": doc_id, "saved": True}


# ===== Entities =====
@app.get("/entities/{doc_id}")
async def get_entities(doc_id: str):
    proc = load_processed(doc_id)
    if not proc:
        raise HTTPException(status_code=404, detail="doc not processed yet")
    return proc.get("entities", {"names": [], "dates": [], "orgs": []})

@app.put("/entities/{doc_id}")
async def save_entities(doc_id: str, body: EntitiesPayload):
    proc = load_processed(doc_id)
    if not proc:
        raise HTTPException(status_code=404, detail="doc not processed yet")
    proc["entities"] = {
        "names": body.names or [],
        "dates": body.dates or [],
        "orgs": body.orgs or [],
    }
    save_processed(doc_id, proc)
    return {"doc_id": doc_id, "saved": True}


# ===== RAG Q&A =====
@app.post("/index/{doc_id}")
async def build_index(doc_id: str):
    proc = load_processed(doc_id)
    if not proc or "chunks" not in proc:
        raise HTTPException(status_code=400, detail="Document not processed yet.")
    agent = QAAgent()
    return agent.build_index(doc_id)

@app.post("/qa/{doc_id}")
async def rag_qa(doc_id: str, body: QAPayload):
    agent = QAAgent()
    agent.build_index(doc_id)
    res = agent.answer(doc_id, body.question.strip(), top_k=body.top_k or 4)
    return {"doc_id": doc_id, "answer": res["answer"], "sources": res["sources"]}


# ===== Critic Review =====
@app.post("/critic/{doc_id}")
async def critic_review(doc_id: str):
    proc = load_processed(doc_id)
    if not proc:
        raise HTTPException(status_code=404, detail="doc not found")
    summary_md = proc.get("edited_summary") or proc.get("sectioned_summary")
    critic = CriticAgent()
    return critic.review_summary(summary_md)


# ===== Root =====
@app.get("/")
def root():
    return JSONResponse({
        "message": "GenAI Doc System backend — all milestones ready",
        "endpoints": ["/upload", "/summary/{doc_id}", "/entities/{doc_id}", "/index/{doc_id}", "/qa/{doc_id}", "/critic/{doc_id}"],
    })
