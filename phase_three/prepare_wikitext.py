"""
Download WikiText-2-raw-v1 from HuggingFace and prepare a CSV for activation extraction.

Filters out empty lines, section headers (lines starting with ' = '), and very short
entries, then samples ~10,000 rows of text. The output CSV has a `statement`
column so it is directly compatible with generate_acts.py.

Usage:
    python prepare_wikitext.py                     # default: 10000 rows
    python prepare_wikitext.py --n_rows 5000       # custom count
"""

import argparse
import pandas as pd
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rows", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_chars", type=int, default=512,
                        help="Truncate each text to this many characters (keeps model context short)")
    args = parser.parse_args()

    print("Downloading WikiText-2-raw-v1 (train split)...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")

    texts = [row["text"] for row in ds]
    print(f"  Raw rows: {len(texts)}")

    # Filter: non-empty, not a section header, at least 40 chars of real content
    filtered = []
    for t in texts:
        t = t.strip()
        if len(t) < 40:
            continue
        if t.startswith("="):
            continue
        # Truncate to max_chars so model doesn't get extremely long inputs
        filtered.append(t[:args.max_chars])

    print(f"  After filtering: {len(filtered)}")

    if len(filtered) < args.n_rows:
        print(f"  Warning: only {len(filtered)} rows available (requested {args.n_rows})")
        sampled = filtered
    else:
        import random
        random.seed(args.seed)
        sampled = random.sample(filtered, args.n_rows)

    # Dummy label column (unsupervised, but for consistency))
    df = pd.DataFrame({"statement": sampled, "label": 0})
    out_path = "datasets/wikitext.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows to {out_path}")

if __name__ == "__main__":
    main()
