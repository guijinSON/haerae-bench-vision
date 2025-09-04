#!/usr/bin/env python
"""
Usage examples

# vLLM, Qwen2-VL style hf model
python gen_mm.py \
  --engine vllm \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --data_path /path/to/klvr.csv \
  --image_col image_path --question_col question \
  --output results/qwen2vl.csv \
  --temperature 0.2 --top_p 0.95 --max_tokens 512

# LiteLLM, OpenAI-compatible vision model (e.g., gpt-4o / gpt-4.1-mini)
export OPENAI_API_KEY=sk-...
python gen_mm.py \
  --engine litellm \
  --model gpt-4o-mini \
  --data_path /path/to/klvr.csv \
  --image_col image_path --question_col question \
  --output results/gpt4omini.csv
"""

import argparse
import base64
import os
import sys
from io import BytesIO
from typing import List, Dict, Any, Tuple

import pandas as pd
from PIL import Image

# Optional imports (engine-specific)
try:
    import torch
    from vllm import LLM, SamplingParams
except Exception:
    LLM = None
    SamplingParams = None

try:
    from litellm import batch_completion
except Exception:
    batch_completion = None


# ------------------------- CLI -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Multimodal QA with vLLM or LiteLLM")
    p.add_argument("--engine", choices=["vllm", "litellm"], required=True)
    p.add_argument("--model", required=True, help="Model name or HF repo id")
    p.add_argument("--data_path", required=True, help="CSV/Parquet file with image+question")
    p.add_argument("--image_root", default="", help="Prefix to join with per-row image paths")
    p.add_argument("--image_col", default="image_path")
    p.add_argument("--question_col", default="question")
    p.add_argument("--id_col", default=None, help="Optional ID column to carry through")
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--system", default="You are a helpful math assistant. Answer precisely.")
    p.add_argument("--reasoning", action="store_true",
                   help="If the model returns <>thinking blocks, strip them.")
    return p.parse_args()


# ------------------------- IO helpers -------------------------
def load_df(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[-1].lower()
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def open_image(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def to_data_url(img: Image.Image) -> str:
    """Encode PIL image to base64 data URL for OpenAI-style payloads."""
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ------------------------- Prompt builders -------------------------
def build_messages_for_vl(question: str) -> List[Dict[str, Any]]:
    """
    Generic chat messages with one image + one text prompt.
    Works with many HF chat templates for VLMs (Qwen-VL/LLaVA/etc.).
    """
    return [
        {"role": "system", "content": [{"type": "text", "text": "You are a precise math assistant."}]},
        {"role": "user", "content": [
            {"type": "image"},  # image will be attached separately
            {"type": "text", "text": question.strip()},
        ]},
    ]


def strip_reasoning(text: str) -> str:
    if text is None:
        return ""
    # common patterns
    for tag in ["</think>", "</thinking>", "<|assistant|>"]:
        if tag in text:
            text = text.split(tag)[-1].strip()
    return text.strip()


# ------------------------- vLLM path -------------------------
def run_vllm(model: str, df: pd.DataFrame, args) -> List[str]:
    if LLM is None:
        raise RuntimeError("vLLM not installed. `pip install vllm`")

    llm_kwargs = {
        "model": model,
        "tensor_parallel_size": max(1, torch.cuda.device_count()),
        "trust_remote_code": True,
    }
    llm = LLM(**llm_kwargs)
    tok = llm.get_tokenizer()

    prompts_and_images: List[Tuple[str, Image.Image]] = []
    for _, row in df.iterrows():
        img_path = os.path.join(args.image_root, str(row[args.image_col]))
        img = open_image(img_path)
        messages = build_messages_for_vl(str(row[args.question_col]))
        # Try to use chat template with images argument (Qwen-VL style).
        try:
            prompt = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                images=[img],
            )
        except TypeError:
            # Fallback: some templates don’t accept images in apply_chat_template.
            # We will attach images only via multi_modal_data to vLLM.
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        prompts_and_images.append((prompt, img))

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # vLLM can take a list of dicts with prompt + multi_modal_data
    reqs = [{"prompt": p, "multi_modal_data": {"image": [img]}}
            for (p, img) in prompts_and_images]
    outputs = llm.generate(reqs, sampling)

    texts = []
    for out in outputs:
        text = out.outputs[0].text
        texts.append(strip_reasoning(text) if args.reasoning else text.strip())
    return texts


# ------------------------- LiteLLM path -------------------------
def run_litellm(model: str, df: pd.DataFrame, args) -> List[str]:
    if batch_completion is None:
        raise RuntimeError("LiteLLM not installed. `pip install litellm`")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY for LiteLLM.")

    messages_batch = []
    for _, row in df.iterrows():
        img_path = os.path.join(args.image_root, str(row[args.image_col]))
        img = open_image(img_path)
        data_url = to_data_url(img)

        # OpenAI-style vision content
        messages = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": [
                {"type": "text", "text": str(row[args.question_col]).strip()},
                {"type": "input_image", "image_url": {"url": data_url}},
            ]},
        ]
        messages_batch.append(messages)

    responses = batch_completion(
        model=model,
        messages=messages_batch,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    def _safe(o):
        try:
            return o["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""

    texts = [_safe(r) for r in responses]
    if args.reasoning:
        texts = [strip_reasoning(t) for t in texts]
    return texts


# ------------------------- main -------------------------
def main():
    args = parse_args()

    df = load_df(args.data_path)
    for col in [args.image_col, args.question_col]:
        if col not in df.columns:
            print(f"Missing column '{col}' in {args.data_path}", file=sys.stderr)
            sys.exit(2)

    if args.engine == "vllm":
        preds = run_vllm(args.model, df, args)
    else:
        preds = run_litellm(args.model, df, args)

    out_df = df.copy()
    out_df["response"] = preds
    # Optionally keep only id/question/image/response
    keep_cols = [c for c in [args.id_col, args.image_col, args.question_col, "response"] if c and c in out_df.columns]
    if keep_cols:
        out_df = out_df[keep_cols + (["response"] if "response" not in keep_cols else [])]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Saved {len(out_df)} rows -> {args.output}")


if __name__ == "__main__":
    main()
