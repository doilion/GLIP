#!/usr/bin/env python

import argparse
import contextlib
import csv
import io
import json
import re
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def build_parser():
    parser = argparse.ArgumentParser(description="Summarize TCT_NGC V2 COCO evaluation results.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--pred-json", required=True)
    parser.add_argument("--runtime-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument(
        "--cat-ids-subset",
        default=None,
        help="Optional comma-separated list of GT category ids to scope COCOeval to. "
        "When provided, COCOeval.params.catIds is restricted to this subset before "
        "evaluate() — yielding a true post-hoc view (same images, same predictions, "
        "fewer scored classes). Leave unset for the standard full-set eval.",
    )
    return parser


def parse_cat_ids_subset(raw):
    if raw is None or raw == "":
        return None
    return [int(x) for x in raw.split(",") if x.strip()]


def parse_inference_timing(log_path: Path):
    text = log_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"Total inference time: (?P<pretty>.+?) \((?P<per_img>[0-9.]+) s / img per device, on (?P<num_devices>\d+) devices\)"
    )
    matches = pattern.findall(text)
    if not matches:
        return {}
    pretty, per_img, num_devices = matches[-1]
    return {
        "total_inference_time_pretty": pretty,
        "seconds_per_image_per_device": float(per_img),
        "num_devices": int(num_devices),
    }


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def main():
    args = build_parser().parse_args()
    cat_ids_subset = parse_cat_ids_subset(args.cat_ids_subset)

    gt = COCO(args.gt_json)
    pred = json.loads(Path(args.pred_json).read_text(encoding="utf-8"))
    coco_dt = gt.loadRes(pred) if pred else COCO()

    coco_eval = COCOeval(gt, coco_dt, "bbox")
    if cat_ids_subset is not None:
        gt_cat_ids = set(gt.getCatIds())
        unknown = [c for c in cat_ids_subset if c not in gt_cat_ids]
        if unknown:
            raise RuntimeError(
                f"--cat-ids-subset contains ids not present in GT: {unknown}. "
                f"GT cat ids: {sorted(gt_cat_ids)}"
            )
        coco_eval.params.catIds = list(cat_ids_subset)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    summary_text = buffer.getvalue().strip()

    metrics = {
        "AP": float(coco_eval.stats[0]),
        "AP50": float(coco_eval.stats[1]),
        "AP75": float(coco_eval.stats[2]),
        "APs": float(coco_eval.stats[3]),
        "APm": float(coco_eval.stats[4]),
        "APl": float(coco_eval.stats[5]),
    }

    precisions = coco_eval.eval["precision"]
    per_class_rows = []
    for idx, cat_id in enumerate(coco_eval.params.catIds):
        category = gt.cats[int(cat_id)]
        precision_all = precisions[:, :, idx, 0, -1]
        precision_all = precision_all[precision_all > -1]
        ap = float(np.mean(precision_all)) if precision_all.size else float("nan")

        precision_50 = precisions[0, :, idx, 0, -1]
        precision_50 = precision_50[precision_50 > -1]
        ap50 = float(np.mean(precision_50)) if precision_50.size else float("nan")

        precision_75 = precisions[5, :, idx, 0, -1]
        precision_75 = precision_75[precision_75 > -1]
        ap75 = float(np.mean(precision_75)) if precision_75.size else float("nan")

        per_class_rows.append(
            {
                "category_id": cat_id,
                "category_name": category["name"],
                "AP": ap,
                "AP50": ap50,
                "AP75": ap75,
            }
        )

    runtime = json.loads(Path(args.runtime_json).read_text(encoding="utf-8"))
    timing_from_log = parse_inference_timing(Path(runtime["log_file"]))
    if timing_from_log:
        runtime.update(timing_from_log)
    if runtime.get("elapsed_seconds", 0) > 0:
        runtime["images_per_second_wallclock"] = len(gt.imgs) / runtime["elapsed_seconds"]

    summary = {
        "name": args.name,
        "num_images": len(gt.imgs),
        "num_annotations": len(gt.anns),
        "num_categories": len(coco_eval.params.catIds),
        "eval_subset_cat_ids": list(cat_ids_subset) if cat_ids_subset is not None else None,
        "metrics": metrics,
        "per_class": per_class_rows,
        "runtime": runtime,
        "coco_summary": summary_text,
        "gt_json": str(Path(args.gt_json).resolve()),
        "pred_json": str(Path(args.pred_json).resolve()),
    }
    summary = to_jsonable(summary)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category_id", "category_name", "AP", "AP50", "AP75"])
        writer.writeheader()
        writer.writerows(per_class_rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
