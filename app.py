# app.py — CPU-only Streamlit app for visual chat (Streamlit Cloud friendly)
import io
from typing import Optional, List, Tuple

import streamlit as st
from PIL import Image
import torch
from packaging import version

# ---- Transformers version check ----
import transformers as tf
from transformers import AutoProcessor
from transformers import Qwen2VLForConditionalGeneration  # <-- correct class for Qwen2-VL

MIN_TF = "4.41.0"  # Qwen2-VL works best on >=4.43, but 4.41+ is acceptable with AutoProcessor.
if version.parse(tf.__version__) < version.parse(MIN_TF):
    raise RuntimeError(
        f"transformers>={MIN_TF} required, found {tf.__version__}. "
        "Update requirements.txt and redeploy."
    )

# -------------------------
# Config (CPU-friendly)
# -------------------------
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"   # small enough for CPU
REVISION_PIN = "main"                    # replace with a commit hash for strict reproducibility

st.set_page_config(page_title="Image Chat (CPU) — Streamlit", page_icon="🖼️", layout="wide")
st.title("🖼️ Visual Chat — CPU-Only (Streamlit Cloud)")

# -------------------------
# Loading helpers (CPU only)
# -------------------------
def load_cpu_model(model_id: str):
    """
    Load a small multimodal chat model on CPU with safe, low-memory settings.
    """
    kwargs = dict(
        trust_remote_code=True,
        revision=REVISION_PIN,
        device_map="cpu",
        low_cpu_mem_usage=True,
        offload_folder="/tmp/hf_offload",   # reduce RAM spikes when assembling weights
        torch_dtype=torch.float32,          # explicit float32 on CPU
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, revision=REVISION_PIN)
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, processor

@torch.inference_mode()
def chat_once(
    model: Qwen2VLForConditionalGeneration,
    processor: AutoProcessor,
    image: Optional[Image.Image],
    question: str,
    history_pairs: Optional[List[Tuple[str, str]]] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> str:
    """
    Qwen2-VL flow (CPU):
    - Build messages as a list of dicts with text+image entries
    - Let AutoProcessor pack everything into tensors
    - Generate & decode
    """
    messages = []
    if history_pairs:
        for u, a in history_pairs:
            messages.append({"role": "user", "content": [{"type": "text", "text": u}]})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": a}]})

    user_content = []
    if image is not None:
        user_content.append({"type": "image", "image": image})
    user_content.append({"type": "text", "text": question})
    messages.append({"role": "user", "content": user_content})

    # Prepare inputs (includes pixel values if image present)
    inputs = processor(
        messages=messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    # Move to CPU device explicitly (Streamlit Cloud is CPU)
    inputs = {k: v.to("cpu") if hasattr(v, "to") else v for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=model.config.eos_token_id,
    )

    output_ids = model.generate(**inputs, **gen_kwargs)

    # New tokens start after input length
    input_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][input_len:]

    # Decode with the processor's tokenizer
    tokenizer = processor.tokenizer
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()

# -------------------------
# UI — image on the left, chat on the right
# -------------------------
with st.sidebar:
    st.subheader("Model (CPU)")
    st.caption("Designed for Streamlit Cloud (no GPU).")
    st.text(MODEL_ID)

    st.subheader("Generation")
    max_new_tokens = st.slider("Max new tokens", 64, 1024, 384, step=32)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.2, step=0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.95, step=0.05)

    load_btn = st.button("Load / Reload model", type="primary")

# Session state
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.processor = None
    st.session_state.history = []  # list of (user, assistant)

# Load model (CPU only)
def needs_load():
    return st.session_state.model is None

if load_btn or needs_load():
    with st.spinner(f"Loading {MODEL_ID} on CPU…"):
        try:
            model, proc = load_cpu_model(MODEL_ID)
            st.session_state.model = model
            st.session_state.processor = proc
            st.success(f"Loaded {MODEL_ID} (CPU)")
        except Exception as e:
            st.session_state.model = None
            st.session_state.processor = None
            st.error(f"Error loading model: {e}")

# Main layout
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Image")
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    image = None
    if uploaded:
        image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
        st.image(image, caption="Input image", use_container_width=True)

with col_right:
    st.subheader("Chat")
    # show history
    if st.session_state.history:
        for u, a in st.session_state.history:
            st.markdown(f"**You:** {u}")
            st.markdown(f"**Model:** {a}")

    user_q = st.text_area(
        "Your question / instruction",
        value="",
        height=100,
        placeholder='e.g., "Describe this screenshot and locate the login button."'
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        clear_btn = st.button("Clear history", use_container_width=True)
    with c2:
        ask_btn = st.button("Ask", type="primary", use_container_width=True)

    if clear_btn:
        st.session_state.history = []
        st.rerun()

    if ask_btn:
        if st.session_state.model is None:
            st.warning("Click “Load / Reload model” in the sidebar first.")
        elif not user_q.strip():
            st.warning("Type a question.")
        else:
            with st.spinner("Thinking…"):
                reply = chat_once(
                    model=st.session_state.model,
                    processor=st.session_state.processor,
                    image=image,
                    question=user_q.strip(),
                    history_pairs=st.session_state.history,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                st.session_state.history.append((user_q.strip(), reply))
                st.rerun()
