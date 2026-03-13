"""
Generate LLaMA-2-7B activations for the WikiText dataset.

Usage (on a GPU machine):
    python generate_wikitext_acts.py                          # defaults: layer 12, batch 25
    python generate_wikitext_acts.py --device cpu             # CPU
    python generate_wikitext_acts.py --batch_size 50          # larger batches
    python generate_wikitext_acts.py --layers 10 11 12 13     # multiple layers
"""

import argparse
import os
import sys

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=[12])
    parser.add_argument("--batch_size", type=int, default=8, help="Reduce if OOM on GPU (e.g. 4)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", default="wikitext")
    args = parser.parse_args()

    model_name = "meta-llama/Llama-2-7b-hf"
    model_short = "llama-2-7b"

    # --- Load dataset ---
    csv_path = os.path.join(REPO_ROOT, "datasets", f"{args.dataset}.csv")
    df = pd.read_csv(csv_path)
    statements = df["statement"].tolist()
    print(f"Loaded {len(statements)} statements from {csv_path}")

    # --- Load model & tokenizer ---
    print(f"Loading {model_name} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()
    print("Model loaded.")

    # Output directories
    save_dir = os.path.join(REPO_ROOT, "acts", model_short, args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    # Extract activations
    with torch.no_grad():
        for idx in tqdm(range(0, len(statements), args.batch_size)):
            batch = statements[idx : idx + args.batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(args.device)

            outputs = model(**inputs, output_hidden_states=True)

            for layer in args.layers:
                # hidden_states[layer]: (batch, seq_len, hidden_dim)
                # We want the last non-pad token for each sequence
                hidden = outputs.hidden_states[layer]
                # attention_mask: 1 for real tokens, 0 for padding
                # Since padding_side="left", the last token is always the rightmost
                last_tok_acts = hidden[:, -1, :]  # (batch, hidden_dim)

                filename = os.path.join(save_dir, f"layer_{layer}_{idx}.pt")
                torch.save(last_tok_acts.cpu(), filename)

            del inputs, outputs
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    n_files = len(args.layers) * len(range(0, len(statements), args.batch_size))
    print(f"Done. Saved {n_files} files to {save_dir}/")


if __name__ == "__main__":
    main()
