"""Check whether a GLIP prompt set fits in MAX_QUERY_LEN tokens.

GLIP concatenates ALL class names into a single BERT prompt by default
(``TEST.CHUNKED_EVALUATION = -1``) and silently truncates anything past
``MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN`` (default 256). For TCT_NGC-style
medical prompts with diagnostic-system annotations (PSC / Bethesda / Paris)
the concatenated query can easily reach 500-700 tokens — meaning more than
half the classes are silently dropped from the language input.

This script reads a GLIP ``OVERRIDE_CATEGORY`` JSON (or a raw COCO ann file)
and reports:

  - how many BERT tokens the full concatenation produces
  - whether it fits the configured MAX_QUERY_LEN
  - which class would be the first to get truncated, if any
  - the smallest ``TEST.CHUNKED_EVALUATION`` value that keeps every chunk
    under the limit

Run from the repo root::

    python tools/check_prompt_token_length.py \\
        data/medical/prompts/tct_ngc_base30.diag_raw.override.json

    python tools/check_prompt_token_length.py \\
        data/medical/prompts/tct_ngc_base30.diag_raw.override.json \\
        --max-query-len 256 --separation " " --tokenizer bert-base-uncased
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List


def load_class_names(path: Path) -> List[str]:
    """Accept either an OVERRIDE_CATEGORY list ``[{id, name, ...}]`` or a COCO
    ann file ``{categories: [{id, name, ...}], ...}``.
    """
    obj = json.loads(path.read_text())
    if isinstance(obj, dict) and "categories" in obj:
        cats = obj["categories"]
    elif isinstance(obj, list):
        cats = obj
    else:
        raise ValueError(f"unrecognized JSON shape in {path}")
    return [c["name"] for c in sorted(cats, key=lambda c: c["id"])]


def find_safe_chunk_size(
    tokenizer, names: List[str], sep: str, max_query_len: int
) -> int:
    """Return the largest K such that every chunk of K consecutive names
    tokenizes to ≤ max_query_len. Returns 0 if even a single class exceeds
    the budget.
    """
    sep_len = len(tokenizer(sep, add_special_tokens=False)["input_ids"])
    # Worst case is the heaviest contiguous K-window. We can pick K
    # incrementally — but the simple O(N²) loop is fine for N ≤ 100.
    for K in range(len(names), 0, -1):
        ok = True
        for i in range(0, len(names), K):
            chunk = names[i:i + K]
            n = len(tokenizer(sep.join(chunk), add_special_tokens=True)["input_ids"])
            if n > max_query_len:
                ok = False
                break
        if ok:
            return K
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("prompt_file", type=Path,
                   help="OVERRIDE_CATEGORY JSON or COCO ann file (uses category names)")
    p.add_argument("--max-query-len", type=int, default=256,
                   help="GLIP MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN (default 256)")
    p.add_argument("--separation", default=" ",
                   help="GLIP DATASETS.SEPARATION_TOKENS (default single space)")
    p.add_argument("--tokenizer", default="bert-base-uncased",
                   help="HuggingFace tokenizer name (default bert-base-uncased)")
    args = p.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: 'transformers' not installed in this environment", file=sys.stderr)
        return 2
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    names = load_class_names(args.prompt_file)
    if not names:
        print("ERROR: no categories found", file=sys.stderr)
        return 2

    concat = args.separation.join(names)
    n_full = len(tokenizer(concat, add_special_tokens=True)["input_ids"])

    print(f"file:          {args.prompt_file}")
    print(f"classes:       {len(names)}")
    print(f"tokenizer:     {args.tokenizer}")
    print(f"separator:     {args.separation!r}  (DATASETS.SEPARATION_TOKENS)")
    print(f"max_query_len: {args.max_query_len}  (MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN)")
    print(f"total tokens:  {n_full}")

    if n_full <= args.max_query_len:
        print(f"\nOK — fits in one query (no chunking needed). "
              f"TEST.CHUNKED_EVALUATION -1 is safe.")
        return 0

    # Show where truncation would bite at the current MAX_QUERY_LEN
    sep_len = len(tokenizer(args.separation, add_special_tokens=False)["input_ids"])
    running = 1  # [CLS]
    truncated_at = None
    for i, nm in enumerate(names):
        running += len(tokenizer(nm, add_special_tokens=False)["input_ids"])
        if i < len(names) - 1:
            running += sep_len
        if running > args.max_query_len:
            truncated_at = i
            break

    print(f"\nWARNING — concatenated prompt EXCEEDS MAX_QUERY_LEN by "
          f"{n_full - args.max_query_len} tokens.")
    if truncated_at is not None:
        survivors = truncated_at
        print(f"  At default CHUNKED_EVALUATION=-1, only the first ~{survivors}/"
              f"{len(names)} classes survive the silent BERT truncation.")
        print(f"  First dropped class: #{truncated_at} {names[truncated_at]!r}")

    K = find_safe_chunk_size(tokenizer, names, args.separation, args.max_query_len)
    if K == 0:
        print("\nERROR: at least one single class already exceeds MAX_QUERY_LEN.")
        print("       Shorten that class's description or raise MAX_QUERY_LEN.")
        return 1

    n_queries = (len(names) + K - 1) // K
    print(f"\nRecommended: TEST.CHUNKED_EVALUATION {K}")
    print(f"             → {n_queries} forward passes per image, "
          f"each chunk ≤ {args.max_query_len} tokens.")
    print(f"             Pass it as a yacs override on the CLI, e.g.")
    print(f"               TEST.CHUNKED_EVALUATION {K}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
