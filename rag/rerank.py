from typing import List, Dict
from app.deps import reranker

def rerank_passages(query: str, passages: List[Dict], topk: int=6) -> List[Dict]:
    if not passages:
        return []
    pairs = [(query, p["text"]) for p in passages]
    scores = reranker().predict(pairs).tolist()
    ranked = sorted(zip(passages, scores), key=lambda x:x[1], reverse=True)[:topk]
    return [p for p,_ in ranked]