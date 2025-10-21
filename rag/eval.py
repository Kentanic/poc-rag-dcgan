import time, numpy as np
from typing import Dict
from .retrieve import HybridRetriever
from .rerank import rerank_passages

def quick_eval() -> Dict:
    qs = [
        "battery safety procedure",
        "wheel bolt torque",
        "tire pressure reset",
    ]
    retr = HybridRetriever()
    t_retr, t_rer = [], []
    for q in qs:
        t0 = time.time(); hits = retr(q, k=40); t_retr.append((time.time()-t0)*1000)
        t1 = time.time(); _ = rerank_passages(q, hits, topk=6); t_rer.append((time.time()-t1)*1000)
    return {
        "retrieval_ms_p50": float(np.percentile(t_retr, 50)) if t_retr else None,
        "retrieval_ms_p95": float(np.percentile(t_retr, 95)) if t_retr else None,
        "rerank_ms_p50": float(np.percentile(t_rer, 50)) if t_rer else None,
        "rerank_ms_p95": float(np.percentile(t_rer, 95)) if t_rer else None,
        "notes": "Add labeled eval set for Recall@k/nDCG/faithfulness."
    }
