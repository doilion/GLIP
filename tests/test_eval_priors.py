"""Regression tests for the post-hoc COCO eval extensions.

Covers:

- ``maskrcnn_benchmark/data/datasets/evaluation/coco/exclude_class_coco_eval.py``
  — exclude a configurable set of "negative" categories from predictions and
    ``COCOeval.params.catIds``.

- ``maskrcnn_benchmark/data/datasets/evaluation/coco/organ_restricted_coco_eval.py``
  — per-image organ prior: drop predictions whose class belongs to a different
    organ than the one parsed from the image filename, then compute per-organ
    AP + macro / instance-weighted aggregates.

These modules don't depend on the maskrcnn_benchmark C extension or the
broken ``torch._six`` import path that blocks the full GLIP runtime in
non-docker envs — they're pure pycocotools + (optionally) torch.load for the
mask .pt path. So this test file is meant to run as a *standalone* sanity
suite::

    python tests/test_eval_priors.py

Not wired into a pytest configuration (GLIP doesn't ship one). If you do add
pytest later, every top-level ``test_*`` function here is a valid testcase.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name, relpath):
    """Load a module from disk without triggering maskrcnn_benchmark.__init__."""
    spec = importlib.util.spec_from_file_location(name, str(REPO_ROOT / relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


excl = _load(
    "excl",
    "maskrcnn_benchmark/data/datasets/evaluation/coco/exclude_class_coco_eval.py",
)
orcm = _load(
    "orcm",
    "maskrcnn_benchmark/data/datasets/evaluation/coco/organ_restricted_coco_eval.py",
)


# --------------------------------------------------------------------------
# Shared mock taxonomy / dataset
# --------------------------------------------------------------------------

MOCK_TAXONOMY = {
    "organs": ["Thyroid gland", "Urine"],
    "organ_to_id": {"Thyroid gland": 0, "Urine": 1},
    "classes": {
        "Thyroid gland-PTC":   {"organ": "Thyroid gland", "organ_id": 0},
        "Thyroid gland-FC":    {"organ": "Thyroid gland", "organ_id": 0},
        "Urine-HGUC":          {"organ": "Urine", "organ_id": 1},
        "Urine-NHGUC":         {"organ": "Urine", "organ_id": 1},
    },
}


def _make_mock_gt():
    return {
        "images": [
            {"id": 1, "width": 100, "height": 100, "file_name": "Thyroid_gland__a.jpg"},
            {"id": 2, "width": 100, "height": 100, "file_name": "Thyroid_gland__b.jpg"},
            {"id": 3, "width": 100, "height": 100, "file_name": "Urine__c.jpg"},
            {"id": 4, "width": 100, "height": 100, "file_name": "Urine__d.jpg"},
        ],
        "categories": [
            {"id": 10, "name": "Thyroid gland-PTC"},
            {"id": 11, "name": "Thyroid gland-FC"},
            {"id": 20, "name": "Urine-HGUC"},
            {"id": 21, "name": "Urine-NHGUC"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 10, "bbox": [10,10,20,20], "area": 400, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 11, "bbox": [10,10,20,20], "area": 400, "iscrowd": 0},
            {"id": 3, "image_id": 3, "category_id": 20, "bbox": [10,10,20,20], "area": 400, "iscrowd": 0},
            {"id": 4, "image_id": 4, "category_id": 21, "bbox": [10,10,20,20], "area": 400, "iscrowd": 0},
        ],
    }


# --------------------------------------------------------------------------
# exclude_class_coco_eval
# --------------------------------------------------------------------------


def test_exclude_class_filter_counts():
    """_filter_predictions_by_category drops exactly the matching rows."""
    preds = [
        {"image_id": 1, "category_id": 10, "bbox": [0,0,1,1], "score": 0.9},
        {"image_id": 1, "category_id": 11, "bbox": [0,0,1,1], "score": 0.8},
        {"image_id": 2, "category_id": 10, "bbox": [0,0,1,1], "score": 0.7},
    ]
    kept, total, dropped = excl._filter_predictions_by_category(preds, [11])
    assert total == 3
    assert len(kept) == 2
    assert dropped == 1
    assert all(r["category_id"] != 11 for r in kept)


def test_exclude_class_resolves_names_and_ids():
    """resolve_exclude_coco_ids handles name lookup + unknown entries."""
    from pycocotools.coco import COCO
    with tempfile.TemporaryDirectory() as d:
        gt_path = os.path.join(d, "gt.json")
        json.dump(_make_mock_gt(), open(gt_path, "w"))
        coco = COCO(gt_path)
        ids, names = excl.resolve_exclude_coco_ids(
            coco, ["Urine-HGUC", 21, "DOES_NOT_EXIST"]
        )
        # Urine-HGUC -> 20, 21 -> Urine-NHGUC; unknown silently dropped
        assert ids == [20, 21]
        assert names == ["Urine-HGUC", "Urine-NHGUC"]


def test_exclude_class_end_to_end():
    """COCOeval restricted to non-excluded cat_ids; predictions filtered too."""
    from pycocotools.coco import COCO
    preds = [
        {"image_id": i, "category_id": cid, "bbox": [10,10,20,20], "score": 0.9}
        for i, cid in [(1,10),(2,11),(3,20),(4,21)]
    ]
    with tempfile.TemporaryDirectory() as d:
        gt_path = os.path.join(d, "gt.json")
        json.dump(_make_mock_gt(), open(gt_path, "w"))
        coco_gt = COCO(gt_path)
        out = os.path.join(d, "out.json")
        coco_eval = excl.evaluate_predictions_on_coco_with_exclusion(
            coco_gt=coco_gt,
            coco_results=preds,
            json_result_file=out,
            iou_type="bbox",
            exclude_coco_ids=[21],  # drop Urine-NHGUC
            classwise=False,
        )
        assert 21 not in coco_eval.params.catIds
        assert set(coco_eval.params.catIds) == {10, 11, 20}
        kept = json.load(open(out))
        assert all(r["category_id"] != 21 for r in kept)
        assert len(kept) == 3


# --------------------------------------------------------------------------
# organ_restricted_coco_eval
# --------------------------------------------------------------------------


def test_organ_path_resolver_primary_rule():
    """``<prefix>__<rest>`` lookup against DEFAULT_PREFIX_TO_ORGAN."""
    fn = orcm.make_default_path_to_organ_fn(
        orcm.DEFAULT_PREFIX_TO_ORGAN,
        ["respiratory tract", "Serous effusion", "Thyroid gland", "Urine", "TCT_CCD"],
    )
    cases = {
        "Thyroid_gland__abc.jpg":         "Thyroid gland",
        "Urine__xyz.jpg":                 "Urine",
        "TCT_CCD__001.jpg":               "TCT_CCD",
        "respiratory_tract__lung.jpg":    "respiratory tract",
        "/data/Serous_effusion__01.jpg":  "Serous effusion",
    }
    for path, expected in cases.items():
        assert fn(path) == expected, f"{path}: got {fn(path)!r}"


def test_organ_path_resolver_segment_fallback():
    """When the basename prefix doesn't match, fall back to path-segment match."""
    fn = orcm.make_default_path_to_organ_fn(
        {"Thyroid_gland": "Thyroid gland"},
        ["Thyroid gland", "Urine"],
    )
    # Directory segment containing the organ name → match
    assert fn("/data/Thyroid gland/img_42.jpg") == "Thyroid gland"
    # Underscore variant in a directory segment → also matches
    assert fn("/data/Thyroid_gland/img_42.jpg") == "Thyroid gland"
    # Random substring within a segment must NOT match — guards short organ names
    # (e.g. organ "GI" shouldn't catch "GI_normal_data").
    fn_gi = orcm.make_default_path_to_organ_fn({}, ["GI"])
    assert fn_gi("/data/GI/x.jpg") == "GI"            # full segment ✓
    assert fn_gi("/data/GI_normal/x.jpg") is None     # substring inside segment ✗


def test_organ_path_resolver_unknown_returns_none():
    fn = orcm.make_default_path_to_organ_fn(
        orcm.DEFAULT_PREFIX_TO_ORGAN,
        list(orcm.DEFAULT_PREFIX_TO_ORGAN.values()),
    )
    assert fn("weird_no_prefix.jpg") is None
    assert fn("/random/path/file.png") is None


def test_organ_prior_filter_counts():
    """8 mock preds (4 correct-organ + 4 wrong-organ) → exactly 4 survive."""
    from pycocotools.coco import COCO
    preds = [
        # image 1 (Thyroid)
        {"image_id": 1, "category_id": 10, "bbox": [10,10,20,20], "score": 0.95},  # ✓
        {"image_id": 1, "category_id": 20, "bbox": [10,10,20,20], "score": 0.85},  # ✗
        # image 2 (Thyroid)
        {"image_id": 2, "category_id": 11, "bbox": [10,10,20,20], "score": 0.95},  # ✓
        {"image_id": 2, "category_id": 21, "bbox": [10,10,20,20], "score": 0.85},  # ✗
        # image 3 (Urine)
        {"image_id": 3, "category_id": 20, "bbox": [10,10,20,20], "score": 0.95},  # ✓
        {"image_id": 3, "category_id": 10, "bbox": [10,10,20,20], "score": 0.85},  # ✗
        # image 4 (Urine)
        {"image_id": 4, "category_id": 21, "bbox": [10,10,20,20], "score": 0.95},  # ✓
        {"image_id": 4, "category_id": 11, "bbox": [10,10,20,20], "score": 0.85},  # ✗
    ]
    with tempfile.TemporaryDirectory() as d:
        tax = os.path.join(d, "tax.json"); json.dump(MOCK_TAXONOMY, open(tax, "w"))
        gt  = os.path.join(d, "gt.json"); json.dump(_make_mock_gt(), open(gt, "w"))
        coco_gt = COCO(gt)
        out = os.path.join(d, "out"); os.makedirs(out)

        summary = orcm.evaluate_with_organ_prior(
            coco_gt=coco_gt, raw_predictions=preds,
            taxonomy_path=tax, output_folder=out, apply_prior=True,
        )
        assert summary["filter_stats"] == {
            "total": 8, "kept": 4, "dropped_wrong_organ": 4,
        }
        assert summary["per_organ"]["Thyroid gland"]["n_classes"] == 2
        assert summary["per_organ"]["Thyroid gland"]["n_instances"] == 2
        assert summary["per_organ"]["Thyroid gland"]["mAP"] == 1.0
        assert summary["per_organ"]["Urine"]["mAP"] == 1.0
        # Filenames: with apply_prior=True they use the ``organ_prior`` suffix.
        assert os.path.isfile(os.path.join(out, "bbox.organ_prior.json"))
        assert os.path.isfile(os.path.join(out, "bbox.organ_prior.summary.json"))


def test_organ_prior_output_filenames_disambiguate_modes():
    """apply_prior=True/False must write to distinct files so they coexist."""
    from pycocotools.coco import COCO
    preds = [{"image_id": 1, "category_id": 10, "bbox": [10,10,20,20], "score": 0.9}]
    with tempfile.TemporaryDirectory() as d:
        tax = os.path.join(d, "tax.json"); json.dump(MOCK_TAXONOMY, open(tax, "w"))
        gt  = os.path.join(d, "gt.json"); json.dump(_make_mock_gt(), open(gt, "w"))
        coco_gt = COCO(gt)
        out = os.path.join(d, "out"); os.makedirs(out)

        orcm.evaluate_with_organ_prior(
            coco_gt=coco_gt, raw_predictions=preds,
            taxonomy_path=tax, output_folder=out, apply_prior=True,
        )
        orcm.evaluate_with_organ_prior(
            coco_gt=coco_gt, raw_predictions=preds,
            taxonomy_path=tax, output_folder=out, apply_prior=False,
        )
        # Both modes' files coexist — no clobber.
        for fname in (
            "bbox.organ_prior.json",
            "bbox.organ_prior.summary.json",
            "bbox.organ_prior_off.json",
            "bbox.organ_prior_off.summary.json",
        ):
            assert os.path.isfile(os.path.join(out, fname)), f"missing {fname}"


def test_organ_prior_loads_both_taxonomy_formats():
    """load_taxonomy dispatches on .json vs .pt and produces the same mapping."""
    json_path = "/home/25_liwenjie/code/WeDetect/data/texts/tct_ngc_taxonomy.json"
    pt_path = "/home/25_liwenjie/code/WeDetect/data/texts/tct_ngc_class_organ_mask_base30.pt"
    if not (os.path.isfile(json_path) and os.path.isfile(pt_path)):
        # Optional: only runs when WeDetect is co-located.
        return
    organs_j, c2o_j = orcm.load_taxonomy(json_path)
    organs_p, c2o_p = orcm.load_taxonomy(pt_path)
    assert organs_j == organs_p
    overlap = set(c2o_j) & set(c2o_p)
    for k in overlap:
        assert c2o_j[k] == c2o_p[k]


# --------------------------------------------------------------------------
# Standalone runner
# --------------------------------------------------------------------------


def _all_tests():
    return sorted(
        (name, obj) for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )


def main():
    failures = 0
    for name, fn in _all_tests():
        try:
            fn()
            print(f"  PASS {name}")
        except Exception as e:
            failures += 1
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
    total = len(_all_tests())
    print(f"\n{total - failures}/{total} passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
