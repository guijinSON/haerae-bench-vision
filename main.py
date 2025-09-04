import argparse, os, sys
from utils.data_io import load_df, ensure_dir
from utils.image_utils import get_data_url, process_hf_image, pil_to_base64
from datasets import load_dataset
from utils.prompts import build_messages_litellm, build_messages_vllm, build_messages_openai
import pandas as pd
from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

def parse_args():
    p = argparse.ArgumentParser(description="Multimodal QA runner (images + text) via vLLM or LiteLLM")
    p.add_argument("--engine", choices=["vllm", "litellm", "openai"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--system", default="You are a helpful math assistant. Answer precisely in Korean if the prompt is Korean.")
    return p.parse_args()

def collect_rows(ds, engine, system):

    rows = []
    for item in ds:
        question_idx = item["question_idx"]
        question = item["question"]
        category = item["category"]
        source = item["source"]
        images = item["images"]  # HF Image objects
        checklist = item["checklist"]
        ground_truth = item["answer"]
        
        processed_images = []
        
        for img in images:
            processed_img = process_hf_image(img)
            if engine == "vllm":
                processed_images.append(
                    processed_img
                )
                
            else:
                processed_images.append(
                    pil_to_base64(processed_img, img_format="JPEG", add_data_uri=True)
                )
                
        if engine == "vllm":
            rows.append(
                build_messages_vllm(question,processed_images,system)
            )
            continue
        elif engine =='openai':
            rows.append(
                build_messages_openai(question,processed_images,system)
            )
        else:
            rows.append(
                build_messages_litellm(question,processed_images,system)
            )
    return rows

def main():
    args = parse_args()
    
    ds = load_dataset(
        'HR-H/HAE_RAE_BENCH_VISON2', 
        split='train'
    )
    df=ds.to_pandas()
    
    rows = collect_rows(ds, args.engine, args.system)

    if args.engine == "vllm":
        from engines.vllm_engine import run as run_vllm
        preds = run_vllm(args.model, rows, args.max_tokens, args.temperature, args.top_p)
        
    else:
        from engines.litellm_engine import run as run_litellm
        preds = run_litellm(args.model, rows, args.max_tokens, args.temperature, args.top_p)

    out_df = df.copy()
    out_df["response"] = preds
    ensure_dir(args.output)
    out_df.to_csv(args.output, index=False)
    print(f"Saved {len(out_df)} rows -> {args.output}")

if __name__ == "__main__":
    main()
