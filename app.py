#!/usr/bin/env python3
"""
Verbose VLM CLI for macOS-friendly local inference.

- Default model: llava-hf/llava-onevision-qwen2-0.5b-ov-hf
- Works with: Qwen/Qwen2-VL-2B (pass --model-id)
- Devices: auto (MPS if available, else CPU), or force --device mps/cpu
- Key fix: uses processor.apply_chat_template with an image content block
           so the <image> placeholder is inserted (avoids token/feature mismatch).
"""

import argparse
import time
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, __version__ as HF_VERSION

# Support both current and future transformers model class names.
try:
    from transformers import AutoModelForImageTextToText as AutoVLMModel
except Exception:
    from transformers import AutoModelForVision2Seq as AutoVLMModel  # deprecated name but widely used


def parse_args():
    p = argparse.ArgumentParser(description="Verbose VLM CLI (macOS-friendly)")
    p.add_argument(
        "--model-id",
        type=str,
        default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        help="HF model id (e.g., Qwen/Qwen2-VL-2B)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cpu"],
        help="Device to use (default: auto picks MPS if available else CPU)",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate (smaller = faster)",
    )
    p.add_argument(
        "--resize-max",
        type=int,
        default=0,
        help="If >0, downscale image so longest side == this value (keeps aspect)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy)",
    )
    p.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p nucleus sampling (1.0 disables)",
    )
    return p.parse_args()


def pick_device(choice: str) -> str:
    if choice == "mps":
        if not torch.backends.mps.is_available():
            print("[warn] --device mps requested but MPS is not available; falling back to CPU")
            return "cpu"
        return "mps"
    if choice == "cpu":
        return "cpu"
    # auto
    return "mps" if torch.backends.mps.is_available() else "cpu"


def maybe_resize(img: Image.Image, resize_max: int) -> Image.Image:
    if resize_max and resize_max > 0:
        w, h = img.size
        longer = max(w, h)
        if longer > resize_max:
            scale = resize_max / float(longer)
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            return img.resize((new_w, new_h), Image.BICUBIC)
    return img


def log_header(model_id: str, device: str, dtype, max_new_tokens: int, resize_max: int, temperature: float, top_p: float):
    print("=" * 76)
    print("Verbose VLM CLI")
    print(f"- Transformers : {HF_VERSION}")
    print(f"- Model ID     : {model_id}")
    print(f"- Device       : {device}")
    print(f"- Torch dtype  : {dtype}")
    print(f"- Max tokens   : {max_new_tokens}")
    print(f"- Resize max   : {resize_max if resize_max else 'off'}")
    print(f"- Sampling     : temperature={temperature}  top_p={top_p}")
    print("=" * 76)


def build_inputs(processor, image: Optional[Image.Image], prompt: str, device: str):
    """
    Builds model inputs. For image turns, we include an explicit {'type': 'image'}
    content block so the chat template inserts the <image> placeholder.
    """
    user_content = []
    if image is not None:
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": user_content}]
    chat_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    if image is not None:
        return processor(images=[image], text=chat_prompt, return_tensors="pt").to(device)
    else:
        return processor(text=chat_prompt, return_tensors="pt").to(device)


def main():
    args = parse_args()

    # Device + dtype
    device = pick_device(args.device)
    dtype = torch.float16 if device == "mps" else torch.float32

    log_header(args.model_id, device, dtype, args.max_new_tokens, args.resize_max, args.temperature, args.top_p)

    # Load artifacts
    t0 = time.time()
    # Most VLMs require custom modules; trust_remote_code=True is common/expected here.
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoVLMModel.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map={"": device},
    ).eval()
    t1 = time.time()
    print(f"[load] Loaded processor+model in {t1 - t0:.1f}s")
    print("Tip: press ENTER at image path for text-only chat.\n")

    while True:
        img_path = input("image path >>>>> ").strip()
        image = None
        if img_path:
            try:
                raw = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[error] failed to open image: {e}")
                continue
            image = maybe_resize(raw, args.resize_max)
            if args.resize_max:
                print(f"[image] original={raw.size}  resized={image.size}")
        else:
            print("(text-only mode)")

        # Simple per-image session; type 'clear' to reset, 'quit'/'exit' to leave.
        while True:
            q = input("Human: ").strip()
            if q.lower() in {"quit", "exit"}:
                print("[info] exiting.")
                return
            if q.lower() == "clear":
                print("(history cleared — note: this CLI doesn’t re-feed history by design)")
                break
            if q == "":
                print("(empty prompt; please type a question)")
                continue

            inputs = build_inputs(processor, image, q, device)

            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
            }
            # enable sampling only if temperature > 0.0 or top_p < 1.0
            if args.temperature > 0.0 or args.top_p < 1.0:
                gen_kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})

            tgen0 = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)
            tgen1 = time.time()

            ans = processor.batch_decode(out, skip_special_tokens=True)[0].strip()

            print("\n--- RESULT -----------------------------------------------------")
            print(ans)
            print("----------------------------------------------------------------")
            print(f"[timing] generation={tgen1 - tgen0:.2f}s  device={device}  dtype={dtype}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[info] interrupted by user")
