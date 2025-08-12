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
