"""Build GLIP caption prompts from the WeDetect-style medical taxonomy.

Given a COCO annotation file (just for the canonical class-name → cat-id
mapping) and a taxonomy JSON, emit GLIP-compatible prompt artefacts so the
language encoder sees the medically-precise descriptions (PSC / Bethesda /
Paris) instead of the bare ``"organ-shortname"`` labels.

Two output formats:

1. ``OVERRIDE_CATEGORY`` JSON — the inline list format used in
   ``configs/odinw_13/*.yaml`` (see ``DATASETS.OVERRIDE_CATEGORY``). Set
   ``DATASETS.USE_OVERRIDE_CATEGORY True`` in the config to actually pick it
   up at eval time.

2. ``caption_prompt`` YAML — list of ``{prefix, name, suffix}`` entries
   suitable for ``DATASETS.CAPTION_PROMPT`` + ``DATASETS.USE_CAPTION_PROMPT
   True``. Lets you separate the descriptive *name* from the surrounding
   prompt grammar.

Description sources (``--source``):

- ``diag_raw``    — e.g. "PSC Category II: Negative — acute neutrophilic inflammation"
- ``diag_full``   — e.g. "PSC II Negative"
- ``diag_entity`` — e.g. "acute neutrophilic inflammation"
- ``fullnames``   — pull names from a separate
                    ``tct_ngc_fullnames_30.json``-style file
                    (positional indexing, requires --fullnames). These are
                    hand-curated descriptive prompts like "Thyroid
                    cytopathology - Papillary thyroid carcinoma (Bethesda VI:
                    Malignant)".

The default ``--organ-prefix`` mode prepends the organ name so prompts read
naturally; pass ``--no-organ-prefix`` for the raw description.

Example
-------

Build GLIP prompts for the TCT_NGC base30 split with the Bethesda /
PSC-rich diag_raw descriptions::

    python tools/build_caption_prompt_from_taxonomy.py \\
        --ann /home1/liwenjie/TCT_NGC/annotations/instances_test_base_clean.json \\
        --taxonomy data/medical/tct_ngc_taxonomy.json \\
        --source diag_raw --organ-prefix \\
        --out-override-json data/medical/tct_ngc_base30_override.json \\
        --out-caption-prompt data/medical/tct_ngc_base30_caption_prompt.yaml

Or use the hand-curated descriptive prompts directly (no transformation)::

    python tools/build_caption_prompt_from_taxonomy.py \\
        --ann /path/to/instances_test_base.json \\
        --taxonomy data/medical/tct_ngc_taxonomy.json \\
        --source fullnames --fullnames data/medical/tct_ngc_fullnames_30.json \\
        --out-override-json /tmp/override.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_taxonomy(path: Path) -> Tuple[Dict[str, dict], List[str]]:
    """Return ``(classes_meta, organ_names)`` from the taxonomy JSON."""
    tax = json.loads(path.read_text())
    return tax["classes"], tax["organs"]


def load_fullnames(path: Path) -> List[str]:
    """Read a ``tct_ngc_fullnames_*.json``: list of [single-name] lists → flat list."""
    raw = json.loads(path.read_text())
    out: List[str] = []
    for entry in raw:
        if isinstance(entry, list):
            out.append(entry[0])
        else:
            out.append(str(entry))
    return out


def pick_description(
    class_name: str,
    classes_meta: Dict[str, dict],
    source: str,
    fullnames: Optional[List[str]],
) -> Optional[str]:
    """Resolve one class's prompt text from the requested source.

    For ``source="fullnames"`` the lookup is by ``classes_meta[name]["class_id"]``
    (the canonical taxonomy index), not by position in the ann file. The
    fullnames file is canonical-ordered, so this is the only way that
    survives any reordering of the dataset's categories.
    """
    if source == "fullnames":
        if fullnames is None:
            raise ValueError("--source fullnames requires --fullnames path")
        meta = classes_meta.get(class_name)
        if meta is None:
            return None
        cid = meta.get("class_id")
        if cid is None or cid >= len(fullnames):
            return None
        return fullnames[cid]

    meta = classes_meta.get(class_name)
    if meta is None:
        return None
    val = meta.get(source)
    return None if val is None else str(val)


def build_prompts(
    ann_path: Path,
    taxonomy_path: Path,
    source: str,
    *,
    fullnames_path: Optional[Path] = None,
    organ_prefix: bool = True,
) -> List[dict]:
    """Return one record per category::

        {"id": int, "name": str, "supercategory": str, "prompt": str}

    Records are sorted by COCO ``id`` (same order WeDetect uses when emitting
    GLIP overrides). Categories without a taxonomy match get ``prompt = name``.
    """
    ann = json.loads(ann_path.read_text())
    classes_meta, organ_names = load_taxonomy(taxonomy_path)
    fullnames = load_fullnames(fullnames_path) if fullnames_path else None

    sorted_cats = sorted(ann["categories"], key=lambda c: c["id"])
    records: List[dict] = []
    missing: List[str] = []
    for cat in sorted_cats:
        name = cat["name"]
        desc = pick_description(name, classes_meta, source, fullnames)
        if desc is None:
            missing.append(name)
            desc = name
        if organ_prefix and source != "fullnames":
            organ = classes_meta.get(name, {}).get("organ")
            if organ:
                desc = f"{organ} cytology - {desc}"
        records.append(
            {
                "id": int(cat["id"]),
                "name": name,
                "supercategory": cat.get("supercategory", classes_meta.get(name, {}).get("organ", "unknown")),
                "prompt": desc,
            }
        )

    if missing:
        print(
            f"[warn] {len(missing)} categories had no taxonomy entry — kept raw name: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}",
            file=sys.stderr,
        )
    return records


def write_override_json(records: List[dict], out_path: Path) -> None:
    """Emit ``DATASETS.OVERRIDE_CATEGORY`` list — id/name/supercategory only.

    Per GLIP convention, ``name`` is what the language encoder sees, so we
    swap in the descriptive prompt as the name.
    """
    entries = [
        {"id": r["id"], "name": r["prompt"], "supercategory": r["supercategory"]}
        for r in records
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    print(f"wrote {out_path} ({len(entries)} categories)")


def write_caption_prompt_yaml(records: List[dict], out_path: Path) -> None:
    """Emit ``DATASETS.CAPTION_PROMPT`` YAML — list of {prefix, name, suffix}.

    Keeps prefix/suffix empty so the descriptive prompt is the bare ``name``
    string and the GLIP separator tokens handle the joining. Edit by hand if
    you want richer prompt grammar.
    """
    lines: List[str] = []
    for r in records:
        # Manual YAML emit so we don't depend on PyYAML.
        prompt_escaped = (
            r["prompt"]
            .replace("\\", "\\\\")
            .replace('"', '\\"')
        )
        lines.append("- prefix: \"\"")
        lines.append(f'  name: "{prompt_escaped}"')
        lines.append("  suffix: \"\"")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path} ({len(records)} entries)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--ann", required=True, type=Path,
                   help="COCO annotation JSON whose categories drive the output order")
    p.add_argument("--taxonomy", required=True, type=Path,
                   help="WeDetect-style taxonomy JSON (see data/medical/tct_ngc_taxonomy.json)")
    p.add_argument("--source", required=True,
                   choices=["diag_raw", "diag_full", "diag_entity", "fullnames"],
                   help="Which description field to extract")
    p.add_argument("--fullnames", type=Path, default=None,
                   help="Path to a fullnames JSON (required when --source=fullnames)")
    p.add_argument("--organ-prefix", action="store_true", default=True,
                   help="Prepend organ name (default on, ignored for --source=fullnames)")
    p.add_argument("--no-organ-prefix", dest="organ_prefix", action="store_false")
    p.add_argument("--out-override-json", type=Path, default=None,
                   help="Write GLIP DATASETS.OVERRIDE_CATEGORY JSON here")
    p.add_argument("--out-caption-prompt", type=Path, default=None,
                   help="Write GLIP DATASETS.CAPTION_PROMPT YAML here")
    args = p.parse_args()

    if args.source == "fullnames" and args.fullnames is None:
        p.error("--source fullnames requires --fullnames PATH")
    if args.out_override_json is None and args.out_caption_prompt is None:
        p.error("specify at least one of --out-override-json / --out-caption-prompt")

    records = build_prompts(
        ann_path=args.ann,
        taxonomy_path=args.taxonomy,
        source=args.source,
        fullnames_path=args.fullnames,
        organ_prefix=args.organ_prefix,
    )

    if args.out_override_json is not None:
        write_override_json(records, args.out_override_json)
    if args.out_caption_prompt is not None:
        write_caption_prompt_yaml(records, args.out_caption_prompt)

    # Small preview for human sanity
    print("\nFirst 5 categories:")
    for r in records[:5]:
        print(f"  id={r['id']:3d}  name={r['name']!r}  →  prompt={r['prompt']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
