"""
Generate LLaMA-2-7B activations for all paraphrase datasets
(layer_{L}_{batch_start_idx}.pt) so collect_acts-style loading works

Usage:
    python paraphrase_activations.py                                    # all datasets, layer 12, cuda:0
    python paraphrase_activations.py --device cpu                       # CPU (slow)
    python paraphrase_activations.py --layers 10 11 12 13               # multiple layers
    python paraphrase_activations.py --batch_size 4                     # smaller batches if OOM
    python paraphrase_activations.py --datasets para_train_cities       # one dataset only

Output structure:
    para_acts/llama-2-7b/{dataset_name}/layer_{layer}_{batch_idx}.pt
"""

import argparse
import os
from glob import glob

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ALL_DATASETS = [
    "train_cities", "train_facts", "train_fever", "test_fever",
    "para_train_cities", "para_train_facts", "para_train_fever", "para_test_fever",
]


def main():
    parser = argparse.ArgumentParser(description="Generate activations for paraphrase datasets")
    parser.add_argument("--layers", nargs="+", type=int, default=[12])
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Reduce to 4 if you hit OOM on a 16 GB GPU")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                        help=f"Which datasets to process. Choices: {ALL_DATASETS}")
    args = parser.parse_args()

    model_name = "meta-llama/Llama-2-7b-hf"
    model_short = "llama-2-7b"

    # Load model & tokenizer
    print(f"Loading {model_name} on {args.device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=args.device  # requires: pip install accelerate
    )
    model.eval()
    print("Model loaded.\n")

    # Process datasets
    for dataset_name in args.datasets:
        csv_path = os.path.join(SCRIPT_DIR, "para_datasets", f"{dataset_name}.csv")
        if not os.path.exists(csv_path):
            print(f"[SKIP] {dataset_name}: {csv_path} not found")
            continue

        df = pd.read_csv(csv_path)
        statements = df["statement"].tolist()

        save_dir = os.path.join(SCRIPT_DIR, "para_acts", model_short, dataset_name)
        os.makedirs(save_dir, exist_ok=True)

        expected_batches = len(range(0, len(statements), args.batch_size))
        existing = glob(os.path.join(save_dir, f"layer_{args.layers[0]}_*.pt"))
        if len(existing) >= expected_batches:
            print(f"[SKIP] {dataset_name}: already complete "
                  f"({len(existing)} files for layer {args.layers[0]})")
            continue

        print(f"[RUN]  {dataset_name}: {len(statements)} statements, "
              f"layers {args.layers}, batch_size {args.batch_size}")

        with torch.no_grad():
            for idx in tqdm(range(0, len(statements), args.batch_size),
                            desc=dataset_name):
                batch = statements[idx : idx + args.batch_size]
                inputs = tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=512,
                ).to(args.device)

                outputs = model(**inputs, output_hidden_states=True)

                for layer in args.layers:
                    hidden = outputs.hidden_states[layer]  # (B, seq, dim)
                    last_tok = hidden[:, -1, :]             # (B, dim)
                    path = os.path.join(save_dir, f"layer_{layer}_{idx}.pt")
                    torch.save(last_tok.cpu(), path)

                del inputs, outputs
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

        print(f"       Done — saved to {save_dir}/\n")

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for dataset_name in args.datasets:
        save_dir = os.path.join(SCRIPT_DIR, "para_acts", model_short, dataset_name)
        for layer in args.layers:
            files = glob(os.path.join(save_dir, f"layer_{layer}_*.pt"))
            if files:
                sample = torch.load(files[0], map_location="cpu")
                total = sum(torch.load(f, map_location="cpu").shape[0] for f in files)
                print(f"  {dataset_name:25s}  layer {layer:2d}  "
                      f"{len(files):4d} files  {total:6d} vectors  dim={sample.shape[1]}")
            else:
                print(f"  {dataset_name:25s}  layer {layer:2d}  — no files —")


if __name__ == "__main__":
    main()
