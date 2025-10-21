import os, json, re, pickle
from typing import List, Dict
import fitz
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from app.settings import settings
from app.deps import text_model

def _extract_pdf(path: str) -> List[Dict]:
    doc = fitz.open(path)
    rows = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text and text.strip():
            rows.append({"doc_id": os.path.basename(path),
                         "page": i+1,
                         "text": text})
    return rows

def _sentences(text: str) -> List[str]:
    return re.split(r'(?<=[.!?])\s+', text.strip())

def _chunk_page(row: Dict, size: int, overlap: int) -> List[Dict]:
    sents = _sentences(row["text"])
    chunks, cur, tokens = [], [], 0
    for s in sents:
        t = max(1, int(len(s.split())*1.3))
        if tokens + t > size and cur:
            chunks.append(cur); cur=[]; tokens=0
        cur.append(s); tokens += t
    if cur: chunks.append(cur)

    out = []
    for ci, group in enumerate(chunks):
        text = " ".join(group)
        out.append({
            "id": f"{row['doc_id']}_p{row['page']}_{ci}",
            "text": text,
            "meta": {"doc_id": row["doc_id"], "page": row["page"]}
        })
    return out

def ingest_dir(paths: List[str]) -> int:
    cfg = settings.cfg
    size = cfg["chunking"]["size"]; overlap = cfg["chunking"]["overlap"]
    all_chunks = []
    for p in paths:
        if os.path.isdir(p):
            for fname in os.listdir(p):
                if fname.lower().endswith(".pdf"):
                    all_chunks += _ingest_file(os.path.join(p, fname), size, overlap)
        elif p.lower().endswith(".pdf"):
            all_chunks += _ingest_file(p, size, overlap)

    os.makedirs(os.path.dirname(cfg["paths"]["text_ids"]), exist_ok=True)
    with open(cfg["paths"]["text_ids"], "w", encoding="utf-8") as f:
        for ch in all_chunks:
            f.write(json.dumps(ch, ensure_ascii=False)+"\n")

    _build_dense(all_chunks)
    _build_bm25(all_chunks)
    return len(all_chunks)

def _ingest_file(path: str, size: int, overlap: int) -> List[Dict]:
    pages = _extract_pdf(path)
    chunks = []
    for row in pages:
        chunks += _chunk_page(row, size=size, overlap=overlap)
    return chunks

def _build_dense(chunks: List[Dict]):
    model = text_model()
    vecs = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
    vecs = np.asarray(vecs, dtype="float32")
    dim = vecs.shape[1]
    os.makedirs(os.path.dirname(settings.cfg["paths"]["text_index"]), exist_ok=True)
    index = faiss.IndexHNSWFlat(dim, 64)
    index.hnsw.efConstruction = 128
    index.add(vecs)
    faiss.write_index(index, settings.cfg["paths"]["text_index"])

def _build_bm25(chunks: List[Dict]):
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    payload = {"bm25": bm25, "docs": [c["text"] for c in chunks]}
    os.makedirs(os.path.dirname(settings.cfg["paths"]["bm25_index"]), exist_ok=True)
    with open(settings.cfg["paths"]["bm25_index"], "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)