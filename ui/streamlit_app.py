# ui/streamlit_app.py
import os, time, json, io
from typing import List, Dict
from PIL import Image
import streamlit as st

from app.settings import settings
from rag.ingest import ingest_dir
from rag.retrieve import HybridRetriever
from rag.rerank import rerank_passages
from rag.generate import stream_answer
from rag.guardrails import blocked, redact
from rag.eval import quick_eval
from app.deps import text_ids

# GAN helpers
from generative.gan_dcgan import train_dcgan, sample_grid, generate_images

# ---------- helpers ----------
def _timeit(fn, *args, **kwargs):
    t0 = time.time(); out = fn(*args, **kwargs)
    return out, int((time.time() - t0) * 1000)

def _ensure_dirs():
    for p in [
        settings.cfg["paths"]["raw_dir"],
        os.path.dirname(settings.cfg["paths"]["text_index"]),
        os.path.dirname(settings.cfg["paths"]["bm25_index"]),
    ]:
        os.makedirs(p, exist_ok=True)

def _save_uploaded_pdfs(uploaded_files):
    saved = []
    raw_dir = settings.cfg["paths"]["raw_dir"]
    os.makedirs(raw_dir, exist_ok=True)
    for f in uploaded_files:
        if not f.name.lower().endswith(".pdf"): continue
        path = os.path.join(raw_dir, f.name)
        with open(path, "wb") as out: out.write(f.read())
        saved.append(path)
    return saved

def _format_ctx(passages: List[Dict]) -> str:
    return "\n\n".join([f"[{p['id']} p{p['meta'].get('page','?')}] {p['text']}" for p in passages])

def _render_citations(passages: List[Dict]):
    with st.expander("📎 Retrieved passages (after rerank)"):
        for p in passages:
            st.markdown(f"{p['id']} — page {p['meta'].get('page','?')}\n\n{p['text']}")
        buf = io.StringIO()
        for p in passages: buf.write(json.dumps(p, ensure_ascii=False) + "\n")
        st.download_button("Download citations (JSONL)", buf.getvalue(), "citations.jsonl", "application/json")

# ---------- page ----------
st.set_page_config(page_title="BMW PoC • Offline RAG + GAN", page_icon="🚗", layout="wide")
_ensure_dirs()

st.sidebar.title("⚙️ Settings")
st.sidebar.caption("Loaded from configs/default.yaml — tweak live for the demo")
k_dense = st.sidebar.number_input("k (fused candidates)", 5, 100, settings.cfg["retrieval"]["k_fused"], 5)
top_k = st.sidebar.number_input("Rerank top_k", 1, 20, settings.cfg["rerank"]["top_k"], 1)
max_new = st.sidebar.number_input("Max new tokens", 32, 1024, settings.cfg["generation"]["max_new_tokens"], 32)
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, float(settings.cfg["generation"]["temperature"]), 0.05)
top_p = st.sidebar.slider("Top-p", 0.1, 1.0, float(settings.cfg["generation"]["top_p"]), 0.05)
st.sidebar.divider()
st.sidebar.markdown(f"GGUF: {os.path.basename(settings.cfg['paths']['gguf_path'])}")
st.sidebar.markdown(f"Embeddings: {settings.cfg['paths']['text_embedder']}")
st.sidebar.markdown(f"Reranker: {settings.cfg['paths']['reranker']}")

tabs = st.tabs(["📥 Ingest", "🔎 Search & Answer", "📊 Metrics", "🧱 Inspect Index", "🧪 Generative"])

# ---------- Ingest ----------
with tabs[0]:
    st.header("📥 Ingest PDFs")
    uploaded = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        saved = _save_uploaded_pdfs(uploaded)
        st.success(f"Saved {len(saved)} file(s) to {settings.cfg['paths']['raw_dir']}")
    if st.button("Build / Rebuild Indexes", type="primary"):
        with st.status("Indexing…", expanded=True) as s:
            paths = [settings.cfg["paths"]["raw_dir"]]
            count, t_ms = _timeit(ingest_dir, paths)
            s.update(label=f"Indexed {count} chunks in {t_ms} ms", state="complete")
            st.success(f"Done. {count} chunks.")
    st.info("Re-run indexing after you add PDFs. It overwrites FAISS, BM25 and IDs JSONL.")

# ---------- Search & Answer ----------
with tabs[1]:
    st.header("🔎 Ask a question")
    q = st.text_input("Your question", placeholder="e.g., 12V battery safety procedure — include page numbers")
    if st.button("Answer"):
        if not q:
            st.warning("Please enter a question."); st.stop()
        if blocked(q):
            st.error("Blocked by policy / jailbreak text detected."); st.stop()

        retriever = HybridRetriever()
        candidates, t_retr_ms = _timeit(retriever, q, k=int(k_dense))
        reranked, t_rer_ms = _timeit(rerank_passages, q, candidates, int(top_k))
        st.caption(f"Retrieval: {t_retr_ms} ms • Rerank: {t_rer_ms} ms • ctx chunks: {len(reranked)}")
        _render_citations(reranked)

        ctx = _format_ctx(reranked)
        prompt = f"""You are a grounded assistant. Use ONLY the context; add citations like [id:..., p:...].
If unsure, say you don't know.

Question: {redact(q)}

Context:
{ctx}
"""
        st.subheader("🧠 Answer")
        t0 = time.time()
        out_placeholder = st.empty(); buf = []; gen_tokens = 0
        for tok in stream_answer(prompt):
            buf.append(tok); gen_tokens += 1
            out_placeholder.markdown("".join(buf))
        gen_ms = int((time.time() - t0) * 1000)
        st.caption(f"Generation: {gen_ms} ms (streamed) • tokens ~{gen_tokens}")

# ---------- Metrics ----------
with tabs[2]:
    st.header("📊 Quick eval")
    if st.button("Run quick eval suite"):
        res = quick_eval(); st.json(res)
    st.info("Expand with Recall@k/MRR/nDCG/faithfulness for fuller evals.")

# ---------- Inspect Index ----------
with tabs[3]:
    st.header("🧱 Inspect indexed chunks")
    try:
        ids = text_ids(); st.write(f"Total chunks: {len(ids)}")
        n = min(50, len(ids))
        st.dataframe([{"id": r["id"], "page": r["meta"].get("page","?"), "text": r["text"][:200]+"…"} for r in ids[:n]])
    except Exception as e:
        st.warning(f"Index not ready: {e}")

# ---------- Generative (DCGAN) ----------
with tabs[4]:
    st.header("🧪 Generative • DCGAN (local)")

    sub = st.radio("Mode", ["Quick demo (Fashion-MNIST)", "Fine-tune on my images"], horizontal=True)

    col1, col2 = st.columns(2)
    with col1:
        epochs = st.number_input("Train epochs", 1, 50, 2, 1)
        if sub == "Fine-tune on my images":
            upload = st.file_uploader("Upload PNG/JPG (10–200 small images)", type=["png","jpg","jpeg"], accept_multiple_files=True)
            custom_dir = None
            if upload:
                custom_dir = "data/custom_gan"; os.makedirs(custom_dir, exist_ok=True)
                for f in upload:
                    with open(os.path.join(custom_dir, f.name), "wb") as w: w.write(f.read())
                st.success(f"Saved {len(upload)} images → {custom_dir}")
        else:
            custom_dir = None

        if st.button("Train"):
            with st.spinner("Training DCGAN..."):
                if custom_dir:
                    ck = train_dcgan(dataset="custom", custom_folder=custom_dir, epochs=int(epochs))
                else:
                    ck = train_dcgan(dataset="fashion-mnist", epochs=int(epochs))
            st.success(f"Training done. Checkpoint: {ck}")

    with col2:
        st.subheader("Generate / Download")
        n = st.slider("How many images?", 4, 64, 36, 4)
        seed = st.number_input("Seed", 0, 1_000_000, 0, 1)
        if st.button("Generate Now"):
            with st.spinner("Sampling from latest checkpoint..."):
                grid = sample_grid(n=int(n), seed=int(seed))
                ims, zip_bytes = generate_images(n=int(n), seed=int(seed))
            st.image(grid, caption="Preview grid", use_column_width=True)
            st.download_button("Download ZIP", data=zip_bytes, file_name="gan_samples.zip", mime="application/zip")
        st.caption("Pitch: synthetic data for rare classes, logos, or parts—fully offline.")

