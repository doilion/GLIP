"""Organ-restricted COCO evaluation for GLIP.

Medical patches in TCT_NGC and similar datasets come with a *known* organ /
tissue label (encoded in the image filename, e.g. ``Thyroid_gland__xxxx.jpg``).
A detector that knows the organ at test time should never predict a class
belonging to a different organ. This module implements that prior post-hoc:

  - read each prediction's ``image_id`` → resolve the image's filename via
    ``coco.imgs[id]["file_name"]``
  - parse the organ id from the filename (default rule: ``<prefix>__<rest>``)
  - drop the prediction unless its ``category_id`` belongs to the same organ

Then run COCOeval three ways for reporting:

  1. **all-class flat** — vanilla mAP over the entire (filtered) prediction set
  2. **per-organ AP** — for each organ, COCOeval restricted to its class subset
  3. **aggregates** — macro mean over organs, instance-weighted mean, and the
     all-class number for reference

The semantics mirror WeDetect's ``OrganRestrictedCocoMetric`` (eval-time
aggregation) and YOLOE's ``eval_domain_prior_infer_tct_ngc.py`` (inference-time
score masking). The two converge on the same numbers when the score-masking
threshold is anything > 0, because masked-to-zero predictions never survive
COCO's score-sort.

Inputs
------

Taxonomy / mask can be supplied either as:

A) a WeDetect-style **mask .pt** produced by ``tools/build_class_organ_mask.py``
   containing ``{class_names, class_ids, organ_names, organ_to_id, mask}``;

B) a **taxonomy JSON** (the ``tct_ngc_taxonomy.json`` schema) — looked up by
   class *name* against ``coco.cats``.

The CLI accepts either.

Path → organ rule
-----------------

The default rule is YOLOE's: split ``os.path.basename(file)`` on ``"__"`` and
look the prefix up in ``PREFIX_TO_ORGAN``. For other naming conventions you
can pass ``--path-prefix-map`` (JSON dict ``prefix -> organ_name``) or a
custom callable when calling from Python.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
from collections import OrderedDict, defaultdict
from contextlib import redirect_stdout
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union


# Default filename → organ-name map for TCT_NGC (mirrors YOLOE's
# tools/domain_prior_utils.py PREFIX_TO_DOMAIN, but maps to the *human* organ
# names used in WeDetect's taxonomy so this module stays prefix-agnostic).
DEFAULT_PREFIX_TO_ORGAN: Dict[str, str] = {
    "respiratory_tract": "respiratory tract",
    "Serous_effusion":   "Serous effusion",
    "Thyroid_gland":     "Thyroid gland",
    "Urine":             "Urine",
    "TCT_CCD":           "TCT_CCD",
}


# --------------------------------------------------------------------------
# Taxonomy / mask loading
# --------------------------------------------------------------------------


def _load_taxonomy_json(path: str) -> Tuple[List[str], Dict[str, str]]:
    """Read a WeDetect-style taxonomy JSON. Returns (organ_names, class_name → organ_name)."""
    with open(path, "r") as f:
        tax = json.load(f)
    organ_names: List[str] = list(tax["organs"])
    class_to_organ: Dict[str, str] = {
        cname: cmeta["organ"] for cname, cmeta in tax["classes"].items()
    }
    return organ_names, class_to_organ


def _load_mask_pt(path: str) -> Tuple[List[str], Dict[str, str]]:
    """Read a WeDetect mask.pt. Returns (organ_names, class_name → organ_name)."""
    import torch

    # weights_only kwarg only exists from torch 1.13+; GLIP's recommended
    # docker image ships torch 1.9. Fall back to the legacy call signature.
    try:
        pkg = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        pkg = torch.load(path, map_location="cpu")
    organ_names: List[str] = list(pkg["organ_names"])
    class_names: List[str] = list(pkg["class_names"])
    mask = pkg["mask"]  # [C, O] float {0,1}

    row_sums = mask.sum(dim=1)
    if not bool((row_sums == 1.0).all()):
        bad = (row_sums != 1.0).nonzero(as_tuple=True)[0].tolist()
        raise ValueError(
            f"mask file {path}: classes {[class_names[i] for i in bad[:5]]} "
            f"don't map to exactly one organ (row sums {row_sums.unique().tolist()})"
        )
    organ_idx = mask.argmax(dim=1).tolist()
    class_to_organ = {n: organ_names[organ_idx[i]] for i, n in enumerate(class_names)}
    return organ_names, class_to_organ


def load_taxonomy(taxonomy_path: str) -> Tuple[List[str], Dict[str, str]]:
    """Dispatch on extension. Returns ``(organ_names, class_name → organ_name)``.

    Supports ``.json`` (taxonomy schema) and ``.pt`` (mask package).
    """
    ext = os.path.splitext(taxonomy_path)[1].lower()
    if ext == ".json":
        return _load_taxonomy_json(taxonomy_path)
    if ext in (".pt", ".pth"):
        return _load_mask_pt(taxonomy_path)
    raise ValueError(
        f"unsupported taxonomy extension {ext!r}; use .json (taxonomy) or .pt (mask)"
    )


# --------------------------------------------------------------------------
# Path → organ
# --------------------------------------------------------------------------


def make_default_path_to_organ_fn(
    prefix_to_organ: Dict[str, str],
    organ_names: Sequence[str],
    *,
    split_token: str = "__",
    fallback_segment_match: bool = True,
) -> Callable[[str], Optional[str]]:
    """Build a ``filename → organ_name`` callable.

    Primary rule: take ``basename(path).split(split_token, 1)[0]`` and look it
    up in ``prefix_to_organ``. If that misses and ``fallback_segment_match``
    is on, look for any organ name as a *full path segment* of ``path``
    (handles layouts like ``/.../Thyroid gland/image_42.jpg`` without the
    false-positive risk of a raw substring scan — e.g. organ ``GI`` would
    not match ``/data/GI_normal/img.jpg``).
    """
    organ_set = set(organ_names)

    # Pre-compute the candidate segment spellings for each organ — both the
    # raw form ("Thyroid gland") and the underscore form ("Thyroid_gland")
    # since YOLOE-style file conventions swap spaces.
    organ_segment_candidates: List[Tuple[str, List[str]]] = [
        (organ, [organ, organ.replace(" ", "_")]) for organ in organ_names
    ]

    def _fn(path: str) -> Optional[str]:
        base = os.path.basename(path)
        head = base.split(split_token, 1)[0]
        if head in prefix_to_organ:
            organ = prefix_to_organ[head]
            return organ if organ in organ_set else None
        if fallback_segment_match:
            # Split on any path separator AND the split_token so that
            # both directory segments and filename-prefix segments count.
            segments = re.split(r"[\\/]+|" + re.escape(split_token), path)
            seg_set = {s for s in segments if s}
            for organ, candidates in organ_segment_candidates:
                if any(c in seg_set for c in candidates):
                    return organ
        return None

    return _fn


# --------------------------------------------------------------------------
# Resolve dataset's COCO cat_ids → organ_id
# --------------------------------------------------------------------------


def resolve_cat_id_to_organ(
    coco_api,
    class_to_organ: Dict[str, str],
    organ_names: Sequence[str],
    logger: Optional[logging.Logger] = None,
) -> Dict[int, int]:
    """Return ``{coco_cat_id → organ_idx}`` for every category in ``coco_api``.

    Categories whose name isn't in ``class_to_organ`` are silently skipped
    (logged as a warning). They will never be excluded from per-organ subsets
    nor counted as belonging to any organ.
    """
    if logger is None:
        logger = logging.getLogger("maskrcnn_benchmark.inference")

    organ_to_idx = {n: i for i, n in enumerate(organ_names)}
    cat_id_to_organ: Dict[int, int] = {}
    missing: List[str] = []
    for cat in coco_api.cats.values():
        cname = cat["name"]
        organ = class_to_organ.get(cname)
        if organ is None:
            missing.append(cname)
            continue
        if organ not in organ_to_idx:
            logger.warning(
                "organ-prior: class %r maps to organ %r which is not in taxonomy organ list",
                cname,
                organ,
            )
            continue
        cat_id_to_organ[cat["id"]] = organ_to_idx[organ]

    if missing:
        logger.warning(
            "organ-prior: %d/%d dataset categories have no organ mapping (e.g. %s) "
            "— they will never survive the organ-prior filter",
            len(missing),
            len(coco_api.cats),
            missing[:5],
        )
    return cat_id_to_organ


# --------------------------------------------------------------------------
# Per-image organ resolution + prediction filter
# --------------------------------------------------------------------------


def resolve_image_id_to_organ(
    coco_api,
    path_to_organ_fn: Callable[[str], Optional[str]],
    organ_names: Sequence[str],
    logger: Optional[logging.Logger] = None,
) -> Dict[int, int]:
    """Return ``{image_id → organ_idx}``. Images whose path doesn't resolve are dropped."""
    if logger is None:
        logger = logging.getLogger("maskrcnn_benchmark.inference")

    organ_to_idx = {n: i for i, n in enumerate(organ_names)}
    img_id_to_organ: Dict[int, int] = {}
    unresolved: List[str] = []
    for img_id, info in coco_api.imgs.items():
        fn = info.get("file_name", "")
        organ = path_to_organ_fn(fn)
        if organ is None or organ not in organ_to_idx:
            unresolved.append(fn)
            continue
        img_id_to_organ[img_id] = organ_to_idx[organ]

    if unresolved:
        logger.warning(
            "organ-prior: %d/%d images could not be resolved to an organ "
            "(e.g. %s) — their predictions will be DROPPED",
            len(unresolved),
            len(coco_api.imgs),
            unresolved[:5],
        )
    return img_id_to_organ


def filter_predictions_by_organ(
    preds: Iterable[dict],
    img_id_to_organ: Dict[int, int],
    cat_id_to_organ: Dict[int, int],
) -> Tuple[List[dict], Dict[str, int]]:
    """Drop any prediction whose class organ != its image's organ.

    Returns ``(kept_preds, stats)`` where ``stats`` has counts:
    ``total``, ``kept``, ``dropped_wrong_organ``, ``dropped_unresolved_img``,
    ``dropped_unresolved_cat``.
    """
    kept: List[dict] = []
    stats = defaultdict(int)
    for r in preds:
        stats["total"] += 1
        img_organ = img_id_to_organ.get(r["image_id"])
        cat_organ = cat_id_to_organ.get(r["category_id"])
        if img_organ is None:
            stats["dropped_unresolved_img"] += 1
            continue
        if cat_organ is None:
            stats["dropped_unresolved_cat"] += 1
            continue
        if img_organ != cat_organ:
            stats["dropped_wrong_organ"] += 1
            continue
        kept.append(r)
        stats["kept"] += 1
    return kept, dict(stats)


# --------------------------------------------------------------------------
# COCOeval runners
# --------------------------------------------------------------------------


def _run_coco_eval(
    coco_gt,
    coco_dt,
    cat_ids: Sequence[int],
    iou_type: str = "bbox",
):
    from pycocotools.cocoeval import COCOeval

    if not cat_ids:
        return None
    ev = COCOeval(coco_gt, coco_dt, iou_type)
    ev.params.catIds = list(cat_ids)
    ev.evaluate()
    ev.accumulate()
    return ev


def _capture_summarize(coco_eval, logger: logging.Logger, label: str) -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        coco_eval.summarize()
    txt = buf.getvalue().strip()
    if txt:
        logger.info("[%s] COCOeval summary:\n%s", label, txt)


def _ascii_table(rows: List[List[str]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [sep]
    for i, row in enumerate(rows):
        out.append("| " + " | ".join(c.ljust(widths[j]) for j, c in enumerate(row)) + " |")
        if i == 0:
            out.append(sep)
    out.append(sep)
    return "\n".join(out)


def evaluate_with_organ_prior(
    coco_gt,
    raw_predictions: Sequence[dict],
    *,
    taxonomy_path: str,
    output_folder: str,
    path_to_organ_fn: Optional[Callable[[str], Optional[str]]] = None,
    extra_prefix_to_organ: Optional[Dict[str, str]] = None,
    apply_prior: bool = True,
    iou_type: str = "bbox",
    logger: Optional[logging.Logger] = None,
) -> Dict[str, object]:
    """Run all-class + per-organ COCOeval with the organ prior applied.

    ``apply_prior=False`` skips the per-image prediction filter — useful as a
    baseline (matches YOLOE's ``prior_OFF`` mode). Per-organ COCOeval is still
    run, so you can compare per-organ AP with vs without the prior.

    Returns a JSON-serializable summary dict and writes:
      - ``bbox.organ_prior.json``       (filtered predictions, when apply_prior=True)
        OR ``bbox.organ_prior_off.json`` (raw predictions, when apply_prior=False)
      - ``bbox.organ_prior.summary.json`` / ``bbox.organ_prior_off.summary.json``
    """
    if logger is None:
        logger = logging.getLogger("maskrcnn_benchmark.inference")

    organ_names, class_to_organ = load_taxonomy(taxonomy_path)

    prefix_to_organ = dict(DEFAULT_PREFIX_TO_ORGAN)
    if extra_prefix_to_organ:
        prefix_to_organ.update(extra_prefix_to_organ)

    if path_to_organ_fn is None:
        path_to_organ_fn = make_default_path_to_organ_fn(prefix_to_organ, organ_names)

    cat_id_to_organ = resolve_cat_id_to_organ(coco_gt, class_to_organ, organ_names, logger)
    img_id_to_organ = resolve_image_id_to_organ(coco_gt, path_to_organ_fn, organ_names, logger)

    # --- apply (or skip) the prior ---
    if apply_prior:
        kept_preds, filter_stats = filter_predictions_by_organ(
            raw_predictions, img_id_to_organ, cat_id_to_organ
        )
        logger.info("organ-prior filter stats: %s", filter_stats)
    else:
        kept_preds = list(raw_predictions)
        filter_stats = {"total": len(kept_preds), "kept": len(kept_preds)}
        logger.info("organ-prior: apply_prior=False, using all %d predictions", len(kept_preds))

    # Distinct filename per mode so a baseline pass written into the SAME
    # output_folder doesn't clobber the filtered predictions.
    suffix = "organ_prior" if apply_prior else "organ_prior_off"

    os.makedirs(output_folder, exist_ok=True)
    bbox_json = os.path.join(output_folder, f"bbox.{suffix}.json")
    with open(bbox_json, "w") as f:
        json.dump(kept_preds, f)

    if not kept_preds:
        logger.error("organ-prior: no surviving predictions — abort eval")
        return {"filter_stats": filter_stats, "all_class": None, "per_organ": {}}

    coco_dt = coco_gt.loadRes(bbox_json)

    # Group cat_ids by organ (only kept ones — i.e. those with a known mapping)
    organ_to_cat_ids: Dict[int, List[int]] = defaultdict(list)
    for cat_id, organ_idx in cat_id_to_organ.items():
        organ_to_cat_ids[organ_idx].append(cat_id)

    # --- (1) all-class flat on the (possibly filtered) prediction set ---
    flat_ev = _run_coco_eval(coco_gt, coco_dt, sorted(cat_id_to_organ.keys()), iou_type)
    _capture_summarize(flat_ev, logger, "all-class flat")
    flat_stats = list(flat_ev.stats) if flat_ev is not None else []

    # --- (2) per-organ ---
    gt_inst_per_organ = defaultdict(int)
    for ann in coco_gt.anns.values():
        organ_idx = cat_id_to_organ.get(ann["category_id"])
        if organ_idx is None:
            continue
        gt_inst_per_organ[organ_idx] += 1

    per_organ: Dict[str, Dict[str, object]] = OrderedDict()
    organ_aps: List[Tuple[str, float, int]] = []  # (organ_name, AP, n_inst)
    table_rows = [["organ", "n_classes", "n_instances", "mAP", "AP50", "AP75"]]
    for organ_idx, organ_name in enumerate(organ_names):
        cats = organ_to_cat_ids.get(organ_idx, [])
        n_inst = int(gt_inst_per_organ.get(organ_idx, 0))
        if not cats or n_inst == 0:
            table_rows.append(
                [organ_name, str(len(cats)), str(n_inst), "n/a", "n/a", "n/a"]
            )
            per_organ[organ_name] = {
                "n_classes": len(cats),
                "n_instances": n_inst,
                "skipped": True,
            }
            continue
        ev = _run_coco_eval(coco_gt, coco_dt, cats, iou_type)
        _capture_summarize(ev, logger, f"organ={organ_name}")
        ap = float(ev.stats[0])
        ap50 = float(ev.stats[1])
        ap75 = float(ev.stats[2])
        table_rows.append(
            [organ_name, str(len(cats)), str(n_inst), f"{ap:.4f}", f"{ap50:.4f}", f"{ap75:.4f}"]
        )
        per_organ[organ_name] = {
            "n_classes": len(cats),
            "n_instances": n_inst,
            "mAP": round(ap, 4),
            "mAP_50": round(ap50, 4),
            "mAP_75": round(ap75, 4),
            "all_stats": [float(s) for s in ev.stats],
        }
        organ_aps.append((organ_name, ap, n_inst))

    # --- (3) aggregates ---
    if organ_aps:
        macro = sum(a for _, a, _ in organ_aps) / len(organ_aps)
        total_inst = sum(n for _, _, n in organ_aps) or 1
        inst_weighted = sum(a * n for _, a, n in organ_aps) / total_inst
    else:
        macro = float("nan")
        inst_weighted = float("nan")

    flat_map = flat_stats[0] if flat_stats else float("nan")
    table_rows.append(["-"] * 6)
    table_rows.append(["overall_macro", "-", "-", f"{macro:.4f}", "-", "-"])
    table_rows.append(["instance_weighted", "-", "-", f"{inst_weighted:.4f}", "-", "-"])
    table_rows.append(
        ["all_class_flat", str(len(cat_id_to_organ)), "-", f"{flat_map:.4f}", "-", "-"]
    )
    logger.info(
        "organ-prior summary (apply_prior=%s):\n%s",
        apply_prior,
        _ascii_table(table_rows),
    )

    summary: Dict[str, object] = OrderedDict()
    summary["apply_prior"] = apply_prior
    summary["taxonomy_path"] = taxonomy_path
    summary["filter_stats"] = filter_stats
    summary["organ_names"] = list(organ_names)
    summary["all_class"] = {
        "mAP": float(flat_map),
        "stats": flat_stats,
        "stat_keys": [
            "AP", "AP50", "AP75", "APs", "APm", "APl",
            "ARmax1", "ARmax10", "ARmax100", "ARs", "ARm", "ARl",
        ],
    }
    summary["per_organ"] = per_organ
    summary["overall_macro_mAP"] = round(macro, 4) if organ_aps else None
    summary["instance_weighted_mAP"] = round(inst_weighted, 4) if organ_aps else None

    out_summary = os.path.join(output_folder, f"bbox.{suffix}.summary.json")
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def evaluate_coco_json_with_organ_prior(
    coco_gt_path: str,
    bbox_json_path: str,
    taxonomy_path: str,
    *,
    output_folder: Optional[str] = None,
    apply_prior: bool = True,
    extra_prefix_to_organ: Optional[Dict[str, str]] = None,
    path_to_organ_fn: Optional[Callable[[str], Optional[str]]] = None,
    iou_type: str = "bbox",
):
    """Post-hoc helper: load GT + prediction JSON and run organ-prior eval."""
    from pycocotools.coco import COCO

    coco_gt = COCO(coco_gt_path)
    with open(bbox_json_path, "r") as f:
        preds = json.load(f)
    if output_folder is None:
        output_folder = os.path.dirname(os.path.abspath(bbox_json_path))
    return evaluate_with_organ_prior(
        coco_gt,
        preds,
        taxonomy_path=taxonomy_path,
        output_folder=output_folder,
        apply_prior=apply_prior,
        extra_prefix_to_organ=extra_prefix_to_organ,
        path_to_organ_fn=path_to_organ_fn,
        iou_type=iou_type,
    )
