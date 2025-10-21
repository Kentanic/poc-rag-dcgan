import os
from llama_cpp import Llama
from app.settings import settings

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        gguf = settings.cfg["paths"]["gguf_path"]
        if not os.path.exists(gguf):
            raise RuntimeError(f"GGUF not found at {gguf}. Put a TinyLlama/Qwen gguf there.")
        _llm = Llama(
            model_path=gguf,
            n_ctx=settings.cfg["generation"]["context_window"],
            n_gpu_layers=0,   # CPU; set >0 if GPU available
            logits_all=False,
            use_mmap=True,
            verbose=False
        )
    return _llm

def stream_answer(prompt: str):
    llm = _get_llm()
    params = dict(
        max_tokens=settings.cfg["generation"]["max_new_tokens"],
        temperature=settings.cfg["generation"]["temperature"],
        top_p=settings.cfg["generation"]["top_p"],
        stream=True
    )
    full_prompt = f"<|system|>You are concise and cite sources like [id:..., p:...].<|end|>\n<|user|>{prompt}<|end|>\n<|assistant|>"
    for out in llm(full_prompt, **params):
        if "choices" in out:
            delta = out["choices"][0]["text"]
            if delta: yield delta
