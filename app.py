# app.py
import io
from typing import Optional, List, Tuple

import streamlit as st
from PIL import Image
import torch
from packaging import version

# Transformers sanity check (prevents "AutoModelForCausalLM not found" issues)
import transformers as tf
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

MIN_TF = "4.41.0"
if version.parse(tf.__version__) < version.parse(MIN_TF):
    raise RuntimeError(
        f"transformers>={MIN_TF} required, found {tf.__version__}. "
        "Update requirements.txt and reinstall."
    )

# -------------------------
# Config
# -------------------------
DEFAULT_MODEL = "THUDM/cogagent-chat-hf"    # great for GUI/grounding + multi-turn
ALT_MODEL     = "THUDM/cogvlm-chat-hf"      # general visual chat/VQA
# For reproducibility, you can replace "main" with a specific commit hash from the model card
REVISION_PIN  = "main"

st.set_page_config(page_title="CogVLM / CogAgent • Streamlit", page_icon="🖼️", layout="wide")
st.title("🖼️ CogVLM / CogAgent — Streamlit")

# -------------------------
# Utilities
# -------------------------
def pick_default_device() -> str:
    """Prefer CUDA, else MPS, else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon fallback (no 4-bit)
    return "cpu"

def dtype_from_choice(choice: str):
    if choice == "bfloat16":
        return torch.bfloat16
    if choice == "float16":
        return torch.float16
    return None  # float32

def is_cog_family(model_id: str) -> bool:
    mid = model_id.lower()
    return ("cogvlm" in mid) or ("cogagent" in mid)

def load_model(model_id: str, device: str, dtype_choice: str, use_4bit: bool):
    """Load HF ports of CogAgent/CogVLM with safe defaults for CPU and GPU."""
    kwargs = dict(
        trust_remote_code=True,
        revision=REVISION_PIN,
    )

    # dtype handling
    torch_dtype = dtype_from_choice(dtype_choice)
    # Force float32 on pure CPU to avoid slowdowns / unsupported ops
    if device == "cpu":
        torch_dtype = None

    # device map
    if device == "cuda":
        kwargs["device_map"] = "auto"
        if use_4bit:
            # requires bitsandbytes + CUDA
            kwargs.update(
                dict(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if dtype_choice == "bfloat16" else torch.float16,
                )
            )
        else:
            kwargs["torch_dtype"] = torch_dtype or torch.bfloat16
    elif device == "mps":
        kwargs["device_map"] = {"": "mps"}
        kwargs["torch_dtype"] = torch_dtype or torch.float16
    else:
        kwargs["device_map"] = "cpu"
        # leave torch_dtype None on CPU

    # ---------- Tokenizer ----------
    if is_cog_family(model_id):
        # Use Vicuna's Llama tokenizer for CogVLM/CogAgent v1 checkpoints
        tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, revision=REVISION_PIN)

    # pad token fix (Vicuna/Llama often lack pad); align pad with eos to avoid warnings
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------- Model ----------
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tokenizer

@torch.inference_mode()
def run_chat(model, tokenizer, image: Optional[Image.Image], question: str,
             history: Optional[List[Tuple[str, str]]],
             max_new_tokens: int = 512, temperature: float = 0.2, top_p: float = 0.95):
    """
    Preferred path: Cog* HF ports expose model.chat(tokenizer, image, question, history=...).
    Fallback path: build a chat template manually.
    Returns: reply text
    """
    if hasattr(model, "chat"):
        # Most CogAgent/CogVLM HF ports provide this
        return model.chat(tokenizer, image, question, history=history)

    # Fallback: generic chat template
    msgs = []
    if history:
        for u, a in history:
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})

    content = question
    if image is not None:
        # Many Cog* templates use a special <image> tag when not using .chat()
        content = f"<image>\n{question}"

    msgs.append({"role": "user", "content": content})
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    # Move to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    reply = tokenizer.decode(gen_out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return reply

def cuda_available() -> bool:
    return torch.cuda.is_available()

def mps_available() -> bool:
    return getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()

# -------------------------
# Sidebar (runtime controls)
# -------------------------
with st.sidebar:
    st.subheader("Model & Runtime")
    model_id = st.selectbox(
        "Model",
        [DEFAULT_MODEL, ALT_MODEL],
        help="CogAgent (GUI/grounding, higher-res). CogVLM (general visual chat/VQA)."
    )

    device_default = pick_default_device()
    device_options = ["cpu"]
    if cuda_available():
        device_options.append("cuda")
    if mps_available():
        device_options.append("mps")

    device = st.selectbox(
        "Device", device_options,
        index=device_options.index(device_default) if device_default in device_options else 0,
        help="CPU works everywhere. CUDA uses NVIDIA GPUs. MPS uses Apple Silicon."
    )

    # dtype choices (float32 recommended on CPU)
    dtype_options = ["float32", "bfloat16", "float16"]
    dtype_index = 0 if device == "cpu" else 1  # default bfloat16 on GPU
    dtype_choice = st.selectbox("Torch dtype", dtype_options, index=dtype_index)

    # 4-bit only when CUDA is available
    use_4bit = False
    if device == "cuda":
        use_4bit = st.checkbox("Load in 4-bit (bitsandbytes)", value=False,
                               help="Saves VRAM. Requires CUDA + bitsandbytes.")

    st.divider()
    st.subheader("Generation")
    max_new_tokens = st.slider("Max new tokens", 64, 2048, 512, step=64)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.2, step=0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.95, step=0.05)

    st.divider()
    load_btn = st.button("Load / Reload model", type="primary")

# -------------------------
# Session state
# -------------------------
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.tokenizer = None
    st.session_state.history = []  # list[(user, assistant)]
    st.session_state.model_id = None

# -------------------------
# Model loading
# -------------------------
def need_load() -> bool:
    return (
        st.session_state.model is None
        or st.session_state.model_id != model_id
    )

if load_btn or need_load():
    with st.spinner(f"Loading {model_id} on {device}…"):
        try:
            model, tok = load_model(model_id, device=device, dtype_choice=dtype_choice, use_4bit=use_4bit)
            st.session_state.model = model
            st.session_state.tokenizer = tok
            st.session_state.model_id = model_id
            st.success(f"Loaded {model_id} ({device})")
        except Exception as e:
            st.session_state.model = None
            st.session_state.tokenizer = None
            st.error(f"Error loading model: {e}")

# -------------------------
# UI: left image, right chat
# -------------------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Image")
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    image = None
    if uploaded:
        bytes_data = uploaded.read()
        image = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        st.image(image, caption="Input image", use_container_width=True)

with col_right:
    st.subheader("Chat")
    # History preview
    if st.session_state.history:
        for u, a in st.session_state.history:
            st.markdown(f"**You:** {u}")
            st.markdown(f"**Model:** {a}")

    user_q = st.text_area(
        "Your question / instruction",
        value="",
        height=100,
        placeholder='e.g., "Describe this screenshot and find the login button."'
    )

    c1, c2, _ = st.columns([1, 1, 1])
    with c1:
        clear_btn = st.button("Clear", use_container_width=True)
    with c2:
        ask_btn = st.button("Ask", type="primary", use_container_width=True)

    if clear_btn:
        st.session_state.history = []
        st.rerun()

    if ask_btn:
        if st.session_state.model is None:
            st.warning("Load a model first from the sidebar.")
        elif not user_q.strip():
            st.warning("Type a question.")
        else:
            with st.spinner("Thinking…"):
                try:
                    reply = run_chat(
                        st.session_state.model,
                        st.session_state.tokenizer,
                        image,
                        user_q.strip(),
                        st.session_state.history,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    # Some Cog* return (reply, history). Normalize both cases.
                    if isinstance(reply, tuple) and len(reply) == 2:
                        reply_text, _ = reply
                    else:
                        reply_text = reply
                    st.session_state.history.append((user_q.strip(), reply_text))
                    st.rerun()
                except Exception as e:
                    st.error(f"Inference error: {e}")
