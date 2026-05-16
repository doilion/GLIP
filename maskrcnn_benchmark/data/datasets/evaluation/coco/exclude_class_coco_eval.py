"""COCO evaluation that drops a configurable set of "negative" classes.

Port of WeDetect's ``ExcludeClassCocoMetric`` (mmdet-based) into the
maskrcnn_benchmark / pycocotools eval stack used by GLIP.

The COCOeval object exposes ``params.catIds``; restricting that set is the
"official" way to exclude categories from the averaged mAP. On top of that we
also strip any *predictions* whose ``category_id`` falls in the exclude list
so the per-class precision table is not polluted with detections that the
metric ignores anyway.

Two entry points:

- :func:`evaluate_predictions_on_coco_with_exclusion` — drop-in replacement for
  ``coco_eval.evaluate_predictions_on_coco``; pass it ``exclude_coco_ids``.
- :func:`evaluate_coco_json_with_exclusion` — post-hoc helper that re-evaluates
  a previously-dumped ``bbox.json`` against a COCO annotation file without
  re-running inference. Useful after a standard ``tools/test_grounding_net.py``
  run, since that already writes the bbox JSON.
"""

import itertools
import json
import logging
import os
from typing import Iterable, List, Optional, Sequence, Tuple, Union


# Default list mirrors WeDetect's NEGATIVE_CLASS_NAMES (cytology TCT line).
# Override per-dataset by passing ``exclude_class_names`` explicitly.
DEFAULT_NEGATIVE_CLASS_NAMES: Tuple[str, ...] = (
    "respiratory tract-Impurity",
    "Serous effusion-Negative samples",
    "Thyroid gland-Negative samples",
    "Urine-NILM",
    "Urine-Negative",
    "Urine-Negative Degeneration",
    "TCT_CCD-normal",
)


def resolve_exclude_coco_ids(
    coco_api,
    exclude_inputs: Sequence[Union[int, str]],
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[int], List[str]]:
    """Resolve a mixed list of names / COCO IDs against ``coco_api.cats``.

    Returns ``(coco_ids, resolved_names)`` — only entries that actually match a
    category in the GT file are kept; unknown entries log a warning.
    """
    if logger is None:
        logger = logging.getLogger("maskrcnn_benchmark.inference")

    name_to_id = {cat["name"]: cat["id"] for cat in coco_api.cats.values()}
    id_set = set(coco_api.cats.keys())

    coco_ids: List[int] = []
    names: List[str] = []
    for item in exclude_inputs:
        if isinstance(item, str):
            if item in name_to_id:
                cid = name_to_id[item]
                coco_ids.append(cid)
                names.append(item)
            else:
                logger.warning(
                    "exclude-class: name %r not found in COCO categories — skipping",
                    item,
                )
        else:
            cid = int(item)
            if cid in id_set:
                coco_ids.append(cid)
                names.append(coco_api.cats[cid]["name"])
            else:
                logger.warning(
                    "exclude-class: COCO id %d not found in GT — skipping", cid
                )

    # Deduplicate while preserving order.
    seen = set()
    dedup_ids, dedup_names = [], []
    for cid, nm in zip(coco_ids, names):
        if cid in seen:
            continue
        seen.add(cid)
        dedup_ids.append(cid)
        dedup_names.append(nm)
    return dedup_ids, dedup_names


def _filter_predictions_by_category(
    coco_results: Iterable[dict], exclude_coco_ids: Sequence[int]
) -> Tuple[List[dict], int, int]:
    """Drop any prediction whose ``category_id`` is in ``exclude_coco_ids``."""
    exclude_set = set(exclude_coco_ids)
    kept: List[dict] = []
    total = 0
    dropped = 0
    for r in coco_results:
        total += 1
        if r.get("category_id") in exclude_set:
            dropped += 1
            continue
        kept.append(r)
    return kept, total, dropped


def _log_classwise_table(coco_eval, kept_cat_ids: Sequence[int], logger: logging.Logger) -> None:
    """Print a per-category AP table for the categories that actually participated.

    Layout mirrors WeDetect's table: ``category | mAP | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l``.
    """
    try:
        precisions = coco_eval.eval["precision"]
    except (AttributeError, KeyError):
        logger.warning("classwise: COCOeval has no 'precision' field — skipping table")
        return

    import numpy as np

    rows = []
    for idx, cat_id in enumerate(kept_cat_ids):
        name = coco_eval.cocoGt.cats[cat_id]["name"]
        # precision: [T, R, K, A, M] = [IoU, recall, cat, area, maxDet]
        p_all = precisions[:, :, idx, 0, -1]
        ap = float(np.mean(p_all[p_all > -1])) if (p_all > -1).any() else float("nan")

        cells: List[str] = [name, f"{ap:.3f}"]
        for iou_idx in (0, 5):  # IoU=0.5, IoU=0.75
            p = precisions[iou_idx, :, idx, 0, -1]
            v = float(np.mean(p[p > -1])) if (p > -1).any() else float("nan")
            cells.append(f"{v:.3f}")
        for area_idx in (1, 2, 3):  # small, medium, large
            p = precisions[:, :, idx, area_idx, -1]
            v = float(np.mean(p[p > -1])) if (p > -1).any() else float("nan")
            cells.append(f"{v:.3f}")
        rows.append(cells)

    headers = ["category", "mAP", "mAP_50", "mAP_75", "mAP_s", "mAP_m", "mAP_l"]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def fmt(row):
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"

    lines = [sep, fmt(headers), sep]
    lines += [fmt(r) for r in rows]
    lines.append(sep)
    logger.info("Per-category AP (excluded categories omitted):\n" + "\n".join(lines))


def evaluate_predictions_on_coco_with_exclusion(
    coco_gt,
    coco_results: Sequence[dict],
    json_result_file: str,
    iou_type: str = "bbox",
    exclude_coco_ids: Sequence[int] = (),
    classwise: bool = True,
    logger: Optional[logging.Logger] = None,
):
    """Run COCOeval restricted to the non-excluded categories.

    Drop-in style: mirrors the signature of
    :func:`maskrcnn_benchmark.data.datasets.evaluation.coco.coco_eval.evaluate_predictions_on_coco`,
    with an extra ``exclude_coco_ids`` argument.
    """
    from pycocotools.coco import COCO  # noqa: F401  (kept for parity)
    from pycocotools.cocoeval import COCOeval

    if logger is None:
        logger = logging.getLogger("maskrcnn_benchmark.inference")

    kept_results, total, dropped = _filter_predictions_by_category(
        coco_results, exclude_coco_ids
    )
    logger.info(
        "exclude-class: filtered predictions %d -> %d (dropped %d in excluded coco_ids=%s)",
        total,
        len(kept_results),
        dropped,
        list(exclude_coco_ids),
    )

    os.makedirs(os.path.dirname(json_result_file) or ".", exist_ok=True)
    with open(json_result_file, "w") as f:
        json.dump(kept_results, f)

    coco_dt = coco_gt.loadRes(str(json_result_file)) if kept_results else None
    if coco_dt is None:
        logger.error("exclude-class: no predictions left after filtering — skipping eval")
        return None

    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)

    all_cat_ids = coco_gt.getCatIds()
    excl = set(exclude_coco_ids)
    kept_cat_ids = [cid for cid in all_cat_ids if cid not in excl]
    coco_eval.params.catIds = kept_cat_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    if classwise:
        _log_classwise_table(coco_eval, kept_cat_ids, logger)

    return coco_eval


def evaluate_coco_json_with_exclusion(
    coco_gt_path: str,
    bbox_json_path: str,
    exclude_class_names: Optional[Sequence[str]] = None,
    exclude_coco_ids: Optional[Sequence[int]] = None,
    output_file: Optional[str] = None,
    iou_type: str = "bbox",
    classwise: bool = True,
):
    """Post-hoc re-evaluation of a dumped ``bbox.json`` with class exclusion.

    Useful path: a standard GLIP eval run already writes
    ``{output_folder}/inference/{dataset}/bbox.json`` plus the GT path is in
    the dataset config. Point this helper at both to get the "excluding
    negatives" numbers without re-running inference.
    """
    from pycocotools.coco import COCO

    logger = logging.getLogger("maskrcnn_benchmark.inference")

    coco_gt = COCO(coco_gt_path)

    inputs: List[Union[int, str]] = []
    if exclude_class_names:
        inputs.extend(exclude_class_names)
    if exclude_coco_ids:
        inputs.extend(int(c) for c in exclude_coco_ids)
    if not inputs:
        inputs = list(DEFAULT_NEGATIVE_CLASS_NAMES)
        logger.info(
            "exclude-class: no --exclude-* args given, falling back to DEFAULT_NEGATIVE_CLASS_NAMES"
        )

    resolved_ids, resolved_names = resolve_exclude_coco_ids(coco_gt, inputs, logger)
    logger.info(
        "exclude-class: excluding %d categories %s (coco_ids=%s)",
        len(resolved_ids),
        resolved_names,
        resolved_ids,
    )

    with open(bbox_json_path, "r") as f:
        coco_results = json.load(f)

    if output_file is None:
        base, ext = os.path.splitext(bbox_json_path)
        output_file = f"{base}.exclude_negative{ext}"

    return evaluate_predictions_on_coco_with_exclusion(
        coco_gt=coco_gt,
        coco_results=coco_results,
        json_result_file=output_file,
        iou_type=iou_type,
        exclude_coco_ids=resolved_ids,
        classwise=classwise,
        logger=logger,
    )
