# vlm_cli.py — LLaVA-OneVision & Qwen2-VL compatible CLI (uses chat template)
import sys, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device=="mps" else torch.float32

print(f"Loading {MODEL_ID} on {device} ({dtype}) ...")
proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID, trust_remote_code=True, torch_dtype=dtype, device_map={"": device}
).eval()

print("Tip: press ENTER at image path for text-only chat.")
img_path = input("image path >>>>> ").strip()
image = None if img_path == "" else Image.open(img_path).convert("RGB")

history = []
while True:
    q = input("Human: ").strip()
    if q.lower() in {"exit", "quit"}:
        break
    if q.lower() == "clear":
        history = []
        print("(history cleared)")
        continue

    # Build chat messages; include an image content block when we have one
    user_content = []
    if image is not None:
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": q})

    messages = [{"role": "user", "content": user_content}]

    # You could tack history here if the model supports multi-turn well; for OneVision keep it simple

    prompt = proc.apply_chat_template(messages, add_generation_prompt=True)

    if image is None:
        inputs = proc(text=prompt, return_tensors="pt").to(device)
    else:
        inputs = proc(images=[image], text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=192)
    ans = proc.batch_decode(out, skip_special_tokens=True)[0].strip()
    print("\nVLM:", ans, "\n")

    history.append((q, ans))
