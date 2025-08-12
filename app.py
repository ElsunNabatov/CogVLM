import streamlit as st
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

st.set_page_config(page_title="VLM Chat (LLaVA-OneVision / Qwen2-VL)", page_icon="🤖", layout="wide")

# ---- Device & dtype detection ----
@st.cache_data(show_spinner=False)
def get_device_dtype():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
    return device, dtype

device, dtype = get_device_dtype()

# ---- Model/processor loader (cached) ----
@st.cache_resource(show_spinner=True)
def load_model_and_processor(model_id: str, torch_dtype, device_choice: str):
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map={"": device_choice},
    ).eval()
    return proc, model

# ---- Sidebar ----
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    default_model = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    model_id = st.text_input("Model ID", value=default_model, help="Any compatible Vision-Language model ID from Hugging Face Hub.")
    st.caption(f"Detected device: **{device}** · dtype: **{str(dtype).split('.')[-1]}**")
    max_new_tokens = st.slider("Max new tokens", 32, 1024, 192, step=32)
    if st.button("Reload model", type="primary"):
        # Trigger cache invalidation by clearing the resource
        load_model_and_processor.clear()
        st.rerun()

# Load model and processor
try:
    proc, model = load_model_and_processor(model_id, dtype, device)
except Exception as e:
    st.error(f"Failed to load model `{model_id}`: {e}")
    st.stop()

# ---- Session state ----
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"|"assistant", "content": "text"}]
if "image" not in st.session_state:
    st.session_state.image = None
if "img_name" not in st.session_state:
    st.session_state.img_name = None

# ---- Header ----
st.markdown(
    """
    <style>
    .bubble { padding: 0.8rem 1rem; border-radius: 1rem; }
    .user   { background: #eef2ff; }
    .bot    { background: #f1f5f9; }
    </style>
    """,
    unsafe_allow_html=True,
)

col_left, col_right = st.columns([2, 1])

with col_right:
    st.markdown("### 🖼️ Image (optional)")
    uploaded = st.file_uploader("Upload an image to include with your prompts (persists until cleared)", type=["png", "jpg", "jpeg", "webp"])
    if uploaded is not None:
        try:
            img = Image.open(uploaded).convert("RGB")
            st.session_state.image = img
            st.session_state.img_name = uploaded.name
            st.image(img, caption=uploaded.name, use_column_width=True)
        except Exception as e:
            st.warning(f"Could not open image: {e}")
    elif st.session_state.image is not None:
        st.image(st.session_state.image, caption=st.session_state.img_name or "Pinned image", use_column_width=True)

    if st.button("Clear pinned image"):
        st.session_state.image = None
        st.session_state.img_name = None
        st.experimental_rerun()

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.experimental_rerun()

with col_left:
    st.title("🤖 VLM Chat")
    st.caption("LLaVA-OneVision & Qwen2-VL compatible chat using the model's chat template.")

    # Display history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    # Chat input
    user_msg = st.chat_input("Type your message…")
    if user_msg:
        # Add user message to UI immediately
        st.session_state.messages.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.write(user_msg)

        # Build the message payload (image block only if we have a pinned image)
        user_content = []
        if st.session_state.image is not None:
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": user_msg})
        messages = [{"role": "user", "content": user_content}]
        prompt = proc.apply_chat_template(messages, add_generation_prompt=True)

        # Tokenization
        try:
            if st.session_state.image is None:
                inputs = proc(text=prompt, return_tensors="pt").to(device)
            else:
                inputs = proc(images=[st.session_state.image], text=prompt, return_tensors="pt").to(device)
        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"Preprocessing failed: {e}")
            st.stop()

        # Generate
        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
                ans = proc.batch_decode(out, skip_special_tokens=True)[0].strip()
            except Exception as e:
                ans = f"Generation failed: {e}"

            placeholder.write(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})
