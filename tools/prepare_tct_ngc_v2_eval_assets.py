#!/usr/bin/env python
"""Build evaluation YAML configs for TCT_NGC v2 dev30 and surface a negative-class
id list that downstream scripts use to derive the post-hoc ``base_no_negative``
view.

Three eval YAMLs are emitted under ``<eval_root>/artifacts/``:

* ``base_with_negative_eval.yaml`` — full dev30 base test (30 cats from the
  shifted ``instances_test_base_dev30.json``, ids 1..30).
* ``novel_eval.yaml`` — pseudo + main novel set (``instances_test_novel.json``,
  5 cats; native sparse ids 21,22,24,25,26 — NOT shifted).
* ``hard_eval.yaml`` — hard novel set (``instances_hard_test.json``, 4 cats;
  native ids 27..30 — NOT shifted).

There is no longer a ``base_no_negative`` annotation file or YAML — that view
is derived post-hoc by re-running ``summarize_tct_ngc_v2_eval.py`` on the
SAME predictions+GT with ``--cat-ids-subset`` set to the complement of
``negative_class_ids_for_view`` (recorded in ``eval_metadata.json``).

Stale ``base_no_negative_eval.yaml`` / ``base_eval_no_negative.json`` files
from prior runs are unlinked so the artifacts dir is always consistent with
the new design.
"""

import argparse
import json
from pathlib import Path

import yaml


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def dump_yaml(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def supercat_for(domain: str) -> str:
    return {
        "respiratory_tract": "respiratory_tract",
        "Serous_effusion": "serous_effusion",
        "Thyroid_gland": "thyroid_gland",
        "Urine": "urine",
        "TCT_CCD": "tct_ccd",
    }.get(domain, domain.lower())


def build_override_for_split(ann_categories, metadata_categories_by_current_name,
                             prompt_overrides_by_id=None):
    """Return OVERRIDE_CATEGORY entries mapped to each ann category.

    Prompt-source priority (per category id):
      1. ``prompt_overrides_by_id`` — usually the OVERRIDE_CATEGORY block from
         the training YAML, so eval prompts match exactly what the model was
         trained on. This is REQUIRED for the base eval (mismatched prompts
         tokenize to different lengths, which can blow past MAX_QUERY_LEN /
         BERT 512 cap and trigger CUDA index-out-of-bounds in
         ``convert_grounding_to_od_logits``).
      2. ``metadata_categories_by_current_name`` — long-form ``prompt_name``
         from the dataset metadata. Used for novel/hard splits where there is
         no training-side prompt to inherit.
      3. Source category ``name`` — fallback.
    """
    overrides = []
    for cat in ann_categories:
        cid = cat["id"]
        if prompt_overrides_by_id and cid in prompt_overrides_by_id:
            entry = prompt_overrides_by_id[cid]
            overrides.append({
                "id": cid,
                "name": entry["name"],
                "supercategory": entry.get("supercategory",
                                            cat.get("supercategory", "unknown")),
            })
            continue
        meta = metadata_categories_by_current_name.get(cat["name"])
        if meta is not None:
            prompt = meta["prompt_name"]
            super_cat = supercat_for(meta["domain"])
        else:
            prompt = cat["name"]
            super_cat = cat.get("supercategory", "unknown")
        overrides.append({"id": cid, "name": prompt, "supercategory": super_cat})
    return overrides


def load_train_prompt_overrides(train_yaml_path: Path):
    """Read the training YAML's OVERRIDE_CATEGORY and index by id."""
    if not train_yaml_path.exists():
        return None
    with train_yaml_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    oc = cfg.get("DATASETS", {}).get("OVERRIDE_CATEGORY") or []
    return {int(entry["id"]): entry for entry in oc}


# Diagnostic-level "negative" classes excluded from the base_no_negative view.
# These are the categories explicitly labeled as cytologically negative for
# malignancy / disease in the original project taxonomy — i.e. they correspond
# to "no finding" diagnoses, not just routine cell types.
# Listing by source category name (current_name field in metadata); the dev30
# Urine-Negative was merged into Urine-NHGUC during the dev30 projection, so
# we match by either the merged or source name to handle both schemas.
NEGATIVE_CLASS_NAMES = {
    "normal",                                  # TCT_CCD-normal
    "TCT_CCD-normal",
    "Serous effusion-Negative samples",
    "Thyroid gland-Negative samples",
    "Thyroid gland-NS",                        # Bethesda II Benign, semantically negative
    "Urine-Negative",                          # may be merged into Urine-NHGUC
    "Urine-NHGUC",                             # the merged dev30 class
    "respiratory tract-Negative samples",      # absent in dev30 but kept for portability
}


def collect_negative_ids(ann, metadata_categories_by_current_name):
    """Return the list of GT category ids that are diagnostic-negative classes
    (per ``NEGATIVE_CLASS_NAMES``). These get dropped from the
    ``base_no_negative`` post-hoc COCOeval view so the reported AP is over
    classes that represent actual cytologic findings (~25 classes in dev30).
    """
    negative_ids = []
    negative_names = []
    for cat in ann["categories"]:
        if cat["name"] in NEGATIVE_CLASS_NAMES:
            negative_ids.append(cat["id"])
            negative_names.append(cat["name"])
    if not negative_ids:
        raise RuntimeError(
            "No diagnostic-negative class names matched. Source category names: "
            f"{[c['name'] for c in ann['categories']]}"
        )
    return sorted(negative_ids), negative_names


def make_eval_cfg(ann_path: Path, img_dir: Path, override_categories, dataset_key: str):
    return {
        "DATALOADER": {
            "ASPECT_RATIO_GROUPING": False,
            "SIZE_DIVISIBILITY": 32,
        },
        "DATASETS": {
            "TEST": (dataset_key,),
            "USE_OVERRIDE_CATEGORY": True,
            "USE_CAPTION_PROMPT": False,
            "OVERRIDE_CATEGORY": override_categories,
            "REGISTER": {
                dataset_key: {
                    "ann_file": str(ann_path),
                    "img_dir": str(img_dir),
                }
            },
        },
        "TEST": {
            "IMS_PER_BATCH": 8,
            # Chunk categories so each prompt fits under MAX_QUERY_LEN=256.
            # The BERT branch in inference.py:265 doesn't truncate the
            # tokenized prompt, so the full 30-cat prompt (~450 tokens)
            # would push positive_map indices past logits.shape[-1] and
            # trigger a CUDA index-out-of-bounds in
            # convert_grounding_to_od_logits.
            "CHUNKED_EVALUATION": 10,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="/root/commonfile/TCT_NGC")
    parser.add_argument(
        "--metadata",
        default=None,
        help="Path to category_prompt_name_map_v2.json (defaults to "
        "<dataset_root>/metadata/category_prompt_name_map_v2.json).",
    )
    parser.add_argument(
        "--checkpoint-config",
        default="OUTPUT/tct_ngc_v2_base_dev30_glip_tiny_goldg_ccsbu/ft_task_1/config.yml",
    )
    parser.add_argument(
        "--weight",
        default="OUTPUT/tct_ngc_v2_base_dev30_glip_tiny_goldg_ccsbu/ft_task_1/model_final.pth",
    )
    parser.add_argument(
        "--eval-root",
        default="OUTPUT/tct_ngc_v2_base_dev30_glip_tiny_goldg_ccsbu/eval",
    )
    parser.add_argument(
        "--base-ann-name",
        default="instances_test_base_dev30.json",
        help="Project-controlled, 1-indexed dev30 test file (produced by "
        "tools/build_tct_ngc_v2_dev30_splits.py).",
    )
    parser.add_argument("--novel-ann-name", default="instances_test_novel.json")
    parser.add_argument("--hard-ann-name", default="instances_hard_test.json")
    parser.add_argument(
        "--train-yaml",
        default="configs/tct_ngc/tct_ngc_v2_base.yaml",
        help="Training task YAML; its OVERRIDE_CATEGORY is reused verbatim for "
        "the base eval split so prompt tokenization matches what the model saw "
        "during training. Pass empty to disable and fall back to metadata "
        "long-form prompts.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    annotations_root = dataset_root / "annotations"
    img_dir = dataset_root / "images"
    metadata_path = Path(args.metadata).resolve() if args.metadata else (
        dataset_root / "metadata" / "category_prompt_name_map_v2.json"
    )
    eval_root = Path(args.eval_root).resolve()
    artifacts_dir = eval_root / "artifacts"
    report_dir = eval_root / "report"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Drop stale outputs from the prior pre-filter design so re-runs are clean.
    for stale_name in ("base_no_negative_eval.yaml", "base_eval_no_negative.json"):
        (artifacts_dir / stale_name).unlink(missing_ok=True)

    metadata = load_json(metadata_path)
    meta_by_current_name = {c["current_name"]: c for c in metadata["categories"]}

    base_ann_path = annotations_root / args.base_ann_name
    novel_ann_path = annotations_root / args.novel_ann_name
    hard_ann_path = annotations_root / args.hard_ann_name

    base_ann = load_json(base_ann_path)
    novel_ann = load_json(novel_ann_path)
    hard_ann = load_json(hard_ann_path)

    # Patch the merged dev30 Urine-NHGUC class so it picks up a sensible prompt
    # via the metadata fallback rules (Urine-NILM is the dominant source).
    if "Urine-NHGUC" not in meta_by_current_name and "Urine-NILM" in meta_by_current_name:
        nilm = meta_by_current_name["Urine-NILM"]
        meta_by_current_name["Urine-NHGUC"] = {
            **nilm,
            "current_name": "Urine-NHGUC",
            "ontology": "negative",
            "prompt_name": "Urinary cytology - Negative for high-grade urothelial carcinoma (NHGUC)",
            "short_name": "Negative for high-grade urothelial carcinoma (NHGUC)",
        }

    # ---- base_with_negative -----------------------------------------------
    train_prompts_by_id = (
        load_train_prompt_overrides(Path(args.train_yaml).resolve())
        if args.train_yaml else None
    )
    base_overrides = build_override_for_split(
        base_ann["categories"], meta_by_current_name,
        prompt_overrides_by_id=train_prompts_by_id,
    )
    base_yaml = artifacts_dir / "base_with_negative_eval.yaml"
    dump_yaml(
        base_yaml,
        make_eval_cfg(base_ann_path, img_dir, base_overrides, "test_base_with_negative"),
    )

    negative_ids, negative_names = collect_negative_ids(base_ann, meta_by_current_name)
    base_cat_ids = [c["id"] for c in base_ann["categories"]]
    no_negative_subset = sorted(set(base_cat_ids) - set(negative_ids))

    # ---- novel ------------------------------------------------------------
    novel_overrides = build_override_for_split(novel_ann["categories"], meta_by_current_name)
    novel_yaml = artifacts_dir / "novel_eval.yaml"
    dump_yaml(novel_yaml, make_eval_cfg(novel_ann_path, img_dir, novel_overrides, "test_novel"))

    # ---- hard -------------------------------------------------------------
    hard_overrides = build_override_for_split(hard_ann["categories"], meta_by_current_name)
    hard_yaml = artifacts_dir / "hard_eval.yaml"
    dump_yaml(hard_yaml, make_eval_cfg(hard_ann_path, img_dir, hard_overrides, "test_hard"))

    summary = {
        "dataset_root": str(dataset_root),
        "metadata": str(metadata_path),
        "checkpoint_config": str(Path(args.checkpoint_config).resolve()),
        "weight": str(Path(args.weight).resolve()),
        "negative_class_ids_for_view": negative_ids,
        "negative_class_names_for_view": negative_names,
        "no_negative_subset_cat_ids": no_negative_subset,
        "splits": {
            "base_with_negative": {
                "ann": str(base_ann_path),
                "yaml": str(base_yaml),
                "categories": len(base_overrides),
                "images": len(base_ann["images"]),
                "annotations": len(base_ann["annotations"]),
            },
            "novel": {
                "ann": str(novel_ann_path),
                "yaml": str(novel_yaml),
                "categories": len(novel_overrides),
                "images": len(novel_ann["images"]),
                "annotations": len(novel_ann["annotations"]),
            },
            "hard": {
                "ann": str(hard_ann_path),
                "yaml": str(hard_yaml),
                "categories": len(hard_overrides),
                "images": len(hard_ann["images"]),
                "annotations": len(hard_ann["annotations"]),
            },
        },
    }
    dump_json(artifacts_dir / "eval_metadata.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
