from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss, os, json, pickle
from .settings import settings

_text_model = None
_reranker = None
_faiss = None
_text_ids = None
_bm25 = None

def text_model():
    global _text_model
    if _text_model is None:
        _text_model = SentenceTransformer(settings.cfg["paths"]["text_embedder"])
    return _text_model

def reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.cfg["paths"]["reranker"])
    return _reranker

def faiss_index():
    global _faiss
    if _faiss is None:
        idx_path = settings.cfg["paths"]["text_index"]
        if not os.path.exists(idx_path):
            raise RuntimeError("FAISS index not found. Run /ingest first.")
        _faiss = faiss.read_index(idx_path)
    return _faiss

def text_ids():
    global _text_ids
    if _text_ids is None:
        id_path = settings.cfg["paths"]["text_ids"]
        with open(id_path, "r", encoding="utf-8") as f:
            _text_ids = [json.loads(l.strip()) for l in f]
    return _text_ids

def bm25_index():
    global _bm25
    if _bm25 is None:
        with open(settings.cfg["paths"]["bm25_index"], "rb") as f:
            _bm25 = pickle.load(f)
    return _bm25