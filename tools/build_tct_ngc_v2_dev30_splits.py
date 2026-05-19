#!/usr/bin/env python
"""Project source train_dev / val_dev annotations onto the dev30 schema.

Source schema (``instances_train_dev.json`` / ``instances_val_dev.json``) has
32 categories with sparse ids ``(0-7, 8-15, 16-20, 23, 31-40)`` (id 8 =
"Serous effusion-Diseased cells" is present; the gap is only between the
TCT_CCD block and the rest).

Dev30 schema (canonical: ``instances_test_base_clean_dev30.json``) has 30
categories. It differs from the 32-category source by:

* Merging the three Urine-negative classes (``Urine-NILM``, ``Urine-Negative``,
  ``Urine-Negative Degeneration``) into a single ``Urine-NHGUC`` class;
* Remapping the sparse source ids to a contiguous dev30 space.

This script writes:

* ``instances_train_dev_dev30.json`` (1-indexed, ids 1..30 by default)
* ``instances_val_dev_dev30.json`` (1-indexed)
* ``instances_test_base_dev30.json`` (1-indexed projection of the canonical
  ``instances_test_base_clean_dev30.json``).

The default ``--id-offset 1`` exists because GLIP's
``CocoGrounding.categories(no_background=True)`` filters out any category whose
JSON id equals 0, treating it as background. Shifting our project-controlled
dev30 files to start at id 1 avoids losing the "respiratory tract Neutrophil"
class from the OVD prompt at eval time.

Scope: this script touches only the dev30 BASE files. ``instances_test_novel.json``
(ids 21,22,24,25,26) and ``instances_hard_test.json`` (ids 27-30) start above
0 and must NOT be passed through this script — their ids are consumed as-is.

Annotation counts are preserved exactly from each source file.
"""

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path


URINE_NHGUC_ALIASES = {
    "Urine-NILM",
    "Urine-Negative",
    "Urine-Negative Degeneration",
}
URINE_NHGUC_TARGET_NAME = "Urine-NHGUC"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_source_to_dev30(source_cats, dev30_cats, id_offset: int):
    """Map source category id -> shifted dev30 category id (by name, with Urine merge)."""
    dev30_by_name = {c["name"]: c["id"] for c in dev30_cats}
    mapping = {}
    unmapped = []
    for src in source_cats:
        name = src["name"]
        if name in URINE_NHGUC_ALIASES:
            target_id = dev30_by_name.get(URINE_NHGUC_TARGET_NAME)
        else:
            target_id = dev30_by_name.get(name)
        if target_id is None:
            unmapped.append(name)
        else:
            mapping[src["id"]] = target_id + id_offset
    if unmapped:
        raise RuntimeError(
            f"Source categories without dev30 mapping: {unmapped}. "
            f"dev30 names: {sorted(dev30_by_name)}"
        )
    return mapping


def shifted_dev30_categories(dev30_cats, id_offset: int):
    """Return dev30 categories with ids shifted by id_offset."""
    return [{**c, "id": c["id"] + id_offset} for c in dev30_cats]


def project_split(source_path: Path, dev30_cats, id_map, output_path: Path, id_offset: int):
    src = load_json(source_path)
    src_ann_count = len(src["annotations"])
    src_img_count = len(src["images"])

    out = {
        "info": dict(src.get("info", {})),
        "licenses": src.get("licenses", []),
        "images": src["images"],
        "annotations": [],
        "categories": shifted_dev30_categories(dev30_cats, id_offset),
    }
    out["info"]["dev30_projected_from"] = source_path.name
    out["info"]["dev30_id_offset"] = id_offset
    out["info"]["dev30_id_map"] = {str(k): v for k, v in id_map.items()}

    for ann in src["annotations"]:
        new_ann = dict(ann)
        new_ann["category_id"] = id_map[ann["category_id"]]
        out["annotations"].append(new_ann)

    if len(out["annotations"]) != src_ann_count:
        raise RuntimeError(
            f"{source_path.name}: annotation count drift "
            f"({src_ann_count} -> {len(out['annotations'])})"
        )

    dump_json(output_path, out)

    cat_counts = Counter(a["category_id"] for a in out["annotations"])
    return {
        "source": str(source_path),
        "output": str(output_path),
        "images": src_img_count,
        "annotations": len(out["annotations"]),
        "categories": len(out["categories"]),
        "annotations_per_dev30_id": {
            str(c["id"]): cat_counts.get(c["id"], 0) for c in out["categories"]
        },
    }


def reindex_canonical_test(canonical_path: Path, output_path: Path, id_offset: int):
    """Take the canonical 0-indexed dev30 test file and produce a shifted copy."""
    src = deepcopy(load_json(canonical_path))
    for cat in src["categories"]:
        cat["id"] = cat["id"] + id_offset
    for ann in src["annotations"]:
        ann["category_id"] = ann["category_id"] + id_offset
    src.setdefault("info", {})
    src["info"]["dev30_id_offset"] = id_offset
    src["info"]["dev30_projected_from"] = canonical_path.name
    dump_json(output_path, src)
    cat_counts = Counter(a["category_id"] for a in src["annotations"])
    return {
        "source": str(canonical_path),
        "output": str(output_path),
        "images": len(src["images"]),
        "annotations": len(src["annotations"]),
        "categories": len(src["categories"]),
        "annotations_per_dev30_id": {
            str(c["id"]): cat_counts.get(c["id"], 0) for c in src["categories"]
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="/root/commonfile/TCT_NGC",
        help="Root directory of the TCT_NGC release.",
    )
    parser.add_argument(
        "--dev30-template",
        default=None,
        help="Path to the dev30 annotation file used as the canonical schema "
        "(defaults to <dataset_root>/annotations/instances_test_base_clean_dev30.json).",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["instances_train_dev.json", "instances_val_dev.json"],
        help="Source annotation file names (under <dataset_root>/annotations/) to project. "
        "Do NOT pass instances_test_novel.json or instances_hard_test.json — those use "
        "different category-id spaces and must be consumed as-is.",
    )
    parser.add_argument(
        "--output-suffix",
        default="_dev30",
        help="Suffix appended before .json for projected outputs.",
    )
    parser.add_argument(
        "--test-output-name",
        default="instances_test_base_dev30.json",
        help="Filename for the project-controlled, shifted dev30 test file.",
    )
    parser.add_argument(
        "--id-offset",
        type=int,
        default=1,
        help="Shift applied to all dev30 category ids (default: 1, so outputs use ids "
        "1..30 instead of 0..29 — required because GLIP treats COCO category id 0 as "
        "background and would silently drop that class from the OVD prompt at eval).",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    annotations_root = dataset_root / "annotations"
    dev30_path = Path(args.dev30_template).resolve() if args.dev30_template else (
        annotations_root / "instances_test_base_clean_dev30.json"
    )
    dev30 = load_json(dev30_path)
    dev30_cats = dev30["categories"]

    summaries = []
    id_map = None
    for source_name in args.sources:
        source_path = annotations_root / source_name
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        source_data = load_json(source_path)
        new_map = build_source_to_dev30(source_data["categories"], dev30_cats, args.id_offset)
        if id_map is None:
            id_map = new_map
        elif new_map != id_map:
            raise RuntimeError(
                f"{source_name}: source->dev30 map differs from prior split. "
                f"prior={id_map} new={new_map}"
            )

        stem = source_path.stem
        output_path = annotations_root / f"{stem}{args.output_suffix}.json"
        summaries.append(project_split(source_path, dev30_cats, id_map, output_path, args.id_offset))

    test_output_path = annotations_root / args.test_output_name
    test_summary = reindex_canonical_test(dev30_path, test_output_path, args.id_offset)

    report = {
        "dataset_root": str(dataset_root),
        "dev30_template": str(dev30_path),
        "id_offset": args.id_offset,
        "source_to_dev30_id_map": {str(k): v for k, v in id_map.items()},
        "splits": summaries,
        "test_split": test_summary,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
