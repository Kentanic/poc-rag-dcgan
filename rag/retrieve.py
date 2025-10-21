import numpy as np, pickle
from sklearn.preprocessing import minmax_scale
from typing import List, Dict
from app.deps import faiss_index, text_ids, text_model
from app.settings import settings

def _dense_search(query: str, topk: int) -> List[tuple]:
    idx = faiss_index()
    model = text_model()
    qv = model.encode([query], normalize_embeddings=True).astype("float32")
    D, I = idx.search(qv, topk)
    ids = text_ids()
    out = []
    for dist, ix in zip(D[0], I[0]):
        if ix < 0: continue
        out.append((ids[ix], float(1.0 - dist)))
    return out

def _bm25_search(query: str, topk: int) -> List[tuple]:
    with open(settings.cfg["paths"]["bm25_index"], "rb") as f:
        payload = pickle.load(f)
    bm25 = payload["bm25"]; docs = payload["docs"]
    scores = bm25.get_scores(query.lower().split())
    order = np.argsort(scores)[::-1][:topk]
    ids = text_ids()
    return [(ids[i], float(scores[i])) for i in order]

class HybridRetriever:
    def __init__(self):
        cfg = settings.cfg
        self.kd = cfg["retrieval"]["k_dense"]
        self.ks = cfg["retrieval"]["k_bm25"]
        self.kf = cfg["retrieval"]["k_fused"]
        self.wd = cfg["retrieval"]["dense_weight"]
        self.ws = cfg["retrieval"]["bm25_weight"]

    def __call__(self, query: str, k: int=None) -> List[Dict]:
        kd = self.kd; ks = self.ks; kf = self.kf if k is None else k
        d = _dense_search(query, kd)
        s = _bm25_search(query, ks)

        d_scores = minmax_scale([sc for _,sc in d]).tolist() if d else []
        s_scores = minmax_scale([sc for _,sc in s]).tolist() if s else []

        fused = {}
        for (i,(meta,_)) in enumerate(d):
            fused[meta["id"]] = fused.get(meta["id"], 0) + self.wd * (d_scores[i] if d_scores else 0)
        for (i,(meta,_)) in enumerate(s):
            fused[meta["id"]] = fused.get(meta["id"], 0) + self.ws * (s_scores[i] if s_scores else 0)

        ranked = sorted(fused.items(), key=lambda x:x[1], reverse=True)[:kf]
        id_map = {m["id"]:m for m in text_ids()}
        return [id_map[r[0]] for r in ranked]
