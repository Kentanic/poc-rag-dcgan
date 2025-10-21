from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List
from .settings import settings
from rag.ingest import ingest_dir
from rag.retrieve import HybridRetriever
from rag.rerank import rerank_passages
from rag.generate import stream_answer
from rag.guardrails import blocked, redact
from rag.eval import quick_eval

app = FastAPI(title="BMW PoC • Offline Multimodal RAG")

class IngestRequest(BaseModel):
    paths: List[str] = []   # optional; use default raw_dir if empty

class QueryRequest(BaseModel):
    query: str
    k: int = 40
    top_k: int = settings.cfg["rerank"]["top_k"]

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ingest")
def ingest(req: IngestRequest):
    raw_dir = settings.cfg["paths"]["raw_dir"] if not req.paths else None
    count = ingest_dir(req.paths or [raw_dir])
    return {"ok": True, "chunks_indexed": count}

@app.post("/query")
def query(q: QueryRequest):
    if blocked(q.query):
        raise HTTPException(400, "Blocked by policy")
    retriever = HybridRetriever()
    candidates = retriever(q.query, k=q.k)
    reranked = rerank_passages(q.query, candidates, topk=q.top_k)
    ctx = "\n\n".join([f"[{p['id']} p{p['meta'].get('page','?')}] {p['text']}" for p in reranked])

    prompt = f"""You are a grounded assistant. Use ONLY the context; add citations like [id:..., p:...].
If unsure, say you don't know.

Question: {redact(q.query)}

Context:
{ctx}
"""
    def gen():
        for tok in stream_answer(prompt):
            yield tok
    return StreamingResponse(gen(), media_type="text/plain")

@app.get("/eval")
def eval_endpoint():
    return JSONResponse(quick_eval())