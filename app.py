import io
import os
from typing import Optional, List, Tuple

import streamlit as st
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------
# Helpers
# -------------------------
def load_model(model_id: str, quant4: bool, dtype_choice: str, device_map: str):
    kwargs = dict(trust_remote_code=True)
    # dtype
    if dtype_choice == "bfloat16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif dtype_choice == "float16":
        kwargs["torch_dtype"] = torch.float16

    # device map
    kwargs["device_map"] = device_map  # "auto" (GPU) or "cpu"

    # 4-bit quant (optional; needs bitsandbytes + GPU)
    if quant4:
        kwargs.update(dict(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if dtype_choice == "bfloat16" else torch.float16,
        ))

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tokenizer

def run_chat(model, tokenizer, image: Optional[Image.Image], question: str, history: Optional[List[Tuple[str, str]]]):
    """
    Uses the model's custom .chat() (provided via trust_remote_code)
    history format: list of (user, assistant) turns
    """
    # Some HF ports of CogVLM/CogAgent ship a .chat util
    # Fallback: build a chat template if .chat doesn't exist
    if hasattr(model, "chat"):
        return model.chat(tokenizer, image, question, history=history)
    else:
        # Generic fallback using chat template
        msgs = []
        if history:
            for u, a in history:
                msgs.append({"role": "user", "content": u})
                msgs.append({"role": "assistant", "content": a})
        if image is not None:
            # The Cog* models expect an image tensor internally when using .chat;
            # without .chat we rely on their chat template with image token.
            # Most Cog* ports expose .chat, so this path is rarely used.
            question = f"<image>\n{question}"
        text = tokenizer.apply_chat_template(msgs + [{"role": "user", "content": question}],
                                             tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.2)
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return resp, history

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="CogVLM / CogAgent • Streamlit", page_icon="🖼️", layout="wide")
st.title("🖼️ CogVLM / CogAgent — Streamlit")

with st.sidebar:
    st.subheader("Model & Runtime")
    model_id = st.selectbox(
        "Model",
        [
            "THUDM/cogagent-chat-hf",     # GUI/grounding/multi-turn
            "THUDM/cogvlm-chat-hf",       # general chat + VQA
        ],
        help="Pick CogAgent for GUI/grounding-heavy tasks; CogVLM for general visual chat."
    )
    device_pref = st.selectbox("Device", ["auto (GPU if available)", "cpu"])
    dtype_choice = st.selectbox("Torch dtype", ["bfloat16", "float16", "float32"], index=0)
    quant4 = st.checkbox("Load in 4-bit (bitsandbytes)", value=False,
                         help="Saves VRAM; needs a CUDA GPU and bitsandbytes.")
    load_btn = st.button("Load / Reload model", type="primary")

    st.caption("Tip: CogAgent supports higher-res inputs and GUI tasks; CogVLM is great for general image QA. "
               "Both expose a convenient `.chat()` in the HF ports.")

# Session state
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.tokenizer = None
    st.session_state.history = []  # list[(user, assistant)]
    st.session_state.model_id = None

# Load model
if load_btn or (st.session_state.model is None):
    with st.spinner(f"Loading {model_id}…"):
        try:
            device_map = "auto" if device_pref.startswith("auto") else "cpu"
            model, tok = load_model(model_id, quant4=quant4, dtype_choice=dtype_choice, device_map=device_map)
            st.session_state.model = model
            st.session_state.tokenizer = tok
            st.session_state.model_id = model_id
            st.success(f"Loaded {model_id}")
        except Exception as e:
            st.error(f"Error loading model: {e}")

# Chat area
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
    # Show prior turns
    if st.session_state.history:
        for u, a in st.session_state.history:
            st.markdown(f"**You:** {u}")
            st.markdown(f"**Model:** {a}")

    user_q = st.text_area("Your question / instruction", value="", height=100,
                          placeholder='e.g., "Describe this screenshot and find the login button."')

    col_a, col_b = st.columns([1, 1])
    with col_a:
        clear_btn = st.button("Clear conversation", use_container_width=True)
    with col_b:
        ask_btn = st.button("Ask", type="primary", use_container_width=True)

    if clear_btn:
        st.session_state.history = []
        st.experimental_rerun()

    if ask_btn:
        if st.session_state.model is None:
            st.warning("Load a model first from the sidebar.")
        elif not user_q.strip():
            st.warning("Type a question.")
        else:
            with st.spinner("Thinking…"):
                try:
                    reply, _ = run_chat(st.session_state.model, st.session_state.tokenizer, image, user_q, st.session_state.history)
                    st.session_state.history.append((user_q, reply))
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error during inference: {e}")
