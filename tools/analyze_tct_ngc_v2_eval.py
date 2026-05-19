#!/usr/bin/env python
"""Per-class analysis over the TCT_NGC V2 dev30 evaluation artifacts.

Consumes the eval outputs produced by the standard GLIP eval pipeline +
``tools/summarize_tct_ngc_v2_eval.py`` (called twice for the base split: full
30-class and ``--cat-ids-subset`` for the no-negative view; once each for
novel and hard). Writes:

    eval/report/per_class_unified.csv
    eval/report/per_class_sorted_by_ap.md
    eval/report/prompt_drift_analysis.json
    eval/report/novel_recall_analysis.json
    eval/report/TCT_NGC_V2_ANALYSIS.md

No re-running of COCOeval; per-class AP values are reused from the existing
CSVs byte-for-byte. The base_no_negative view is a strict subset of
base_with_negative (same predictions, same GT, same images) — we assert this
at load time via the ``eval_subset_cat_ids`` field.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_ROOT = REPO_ROOT / "OUTPUT" / "tct_ngc_v2_base_dev30_glip_tiny_goldg_ccsbu" / "eval"
DEFAULT_DATASET_ROOT = Path("/root/commonfile/TCT_NGC")

SMALL_SAMPLE_TRAIN_THRESHOLD = 500
LOW_AP_THRESHOLD = 0.05
OK_AP_THRESHOLD = 0.25


# -------------------------- IO helpers --------------------------------------

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_per_class_csv(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cid = int(row["category_id"])
            rows[cid] = {
                "category_id": cid,
                "category_name": row["category_name"],
                "AP": float(row["AP"]),
                "AP50": float(row["AP50"]),
                "AP75": float(row["AP75"]),
            }
    return rows


def load_task_yaml_prompts(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    overrides = cfg.get("DATASETS", {}).get("OVERRIDE_CATEGORY", []) or []
    return {int(entry["id"]): entry["name"] for entry in overrides}


# -------------------------- IoU helpers -------------------------------------

def bbox_iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def class_agnostic_recall(gt_anns, preds, iou_thresh=0.5, top_k_per_image=100):
    gts_by_image = defaultdict(list)
    for ann in gt_anns:
        gts_by_image[ann["image_id"]].append(ann)
    preds_by_image = defaultdict(list)
    for p in preds:
        preds_by_image[p["image_id"]].append(p)
    matched = 0
    total = 0
    for image_id, gts in gts_by_image.items():
        total += len(gts)
        preds_here = sorted(preds_by_image.get(image_id, []), key=lambda p: -p["score"])[:top_k_per_image]
        used = [False] * len(preds_here)
        for gt in gts:
            best_iou = 0.0
            best_idx = -1
            for i, p in enumerate(preds_here):
                if used[i]:
                    continue
                iou = bbox_iou_xywh(gt["bbox"], p["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou >= iou_thresh and best_idx >= 0:
                used[best_idx] = True
                matched += 1
    return {"matched": matched, "total": total, "recall": matched / total if total else 0.0}


# -------------------------- diagnosis ---------------------------------------

def diagnose(split, n_train, n_test, ap, ap_with_neg=None):
    if split == "novel" or split == "hard":
        if ap < LOW_AP_THRESHOLD:
            return "zero_shot_fail", f"{split} zero-shot: AP={ap:.3f}; prompt likely unaligned"
        return "zero_shot_ok", f"{split} zero-shot: AP={ap:.3f}"

    if n_train < SMALL_SAMPLE_TRAIN_THRESHOLD:
        return "small_sample", f"n_train={n_train} (< {SMALL_SAMPLE_TRAIN_THRESHOLD}); AP={ap:.3f}"

    if ap < LOW_AP_THRESHOLD and n_train >= SMALL_SAMPLE_TRAIN_THRESHOLD:
        return "model_issue", f"n_train={n_train} sufficient but AP={ap:.3f}"

    if split == "base_no_neg" and ap_with_neg is not None:
        delta = ap - ap_with_neg
        if delta < -0.02 and ap_with_neg >= 0.05:
            return "prompt_drift", f"ΔAP={delta:+.3f} vs with_negative"

    if ap >= OK_AP_THRESHOLD:
        return "ok", f"n_train={n_train}, AP={ap:.3f}"

    return "moderate", f"n_train={n_train}, AP={ap:.3f}"


# -------------------------- main --------------------------------------------

def main():
    eval_root = DEFAULT_EVAL_ROOT
    report_dir = eval_root / "report"
    artifacts_dir = eval_root / "artifacts"
    report_dir.mkdir(parents=True, exist_ok=True)

    eval_metadata = load_json(artifacts_dir / "eval_metadata.json")
    annotations_root = DEFAULT_DATASET_ROOT / "annotations"

    # --- summaries ---
    with_neg_summary = load_json(report_dir / "base_with_negative_summary.json")
    no_neg_summary = load_json(report_dir / "base_no_negative_summary.json")
    novel_summary = load_json(report_dir / "novel_summary.json")
    hard_summary = (report_dir / "hard_summary.json")
    hard_summary = load_json(hard_summary) if hard_summary.exists() else None

    # provenance: no_negative must be a strict subset of with_negative on the
    # same predictions+GT. Same images, same num_annotations, smaller cat set.
    assert with_neg_summary["num_images"] == no_neg_summary["num_images"], (
        f"Image-set drift between base views: with_neg={with_neg_summary['num_images']} "
        f"vs no_neg={no_neg_summary['num_images']}. base_no_negative must be a "
        f"post-hoc COCOeval-catIds view, not a different annotation file."
    )
    assert with_neg_summary["pred_json"] == no_neg_summary["pred_json"], (
        "Both base views must derive from the same predictions JSON."
    )
    no_neg_subset = no_neg_summary.get("eval_subset_cat_ids")
    assert no_neg_subset is not None, "base_no_negative summary missing eval_subset_cat_ids."
    with_neg_full = with_neg_summary.get("eval_subset_cat_ids")
    assert with_neg_full is None, "base_with_negative summary should not have eval_subset_cat_ids set."

    expected_subset = set(eval_metadata.get("no_negative_subset_cat_ids", []))
    actual_subset = set(no_neg_subset)
    assert actual_subset == expected_subset, (
        "base_no_negative summary's eval_subset_cat_ids does not match "
        "eval_metadata.no_negative_subset_cat_ids — the summarize call used "
        "the wrong --cat-ids-subset.\n"
        f"  expected: {sorted(expected_subset)}\n"
        f"  got:      {sorted(actual_subset)}\n"
        f"  diff (expected - got): {sorted(expected_subset - actual_subset)}\n"
        f"  diff (got - expected): {sorted(actual_subset - expected_subset)}"
    )

    # --- per-class CSVs ---
    with_neg_per = load_per_class_csv(report_dir / "base_with_negative_per_class.csv")
    no_neg_per = load_per_class_csv(report_dir / "base_no_negative_per_class.csv")
    novel_per = load_per_class_csv(report_dir / "novel_per_class.csv")
    hard_per_path = report_dir / "hard_per_class.csv"
    hard_per = load_per_class_csv(hard_per_path) if hard_per_path.exists() else {}

    assert set(no_neg_per).issubset(set(with_neg_per)), (
        f"base_no_negative cats not a subset of base_with_negative: extras="
        f"{sorted(set(no_neg_per) - set(with_neg_per))}"
    )

    # --- GT files for n_train / n_test counts ---
    train_gt = load_json(annotations_root / "instances_train_dev_dev30.json")
    base_test_gt = load_json(annotations_root / "instances_test_base_dev30.json")
    novel_gt = load_json(annotations_root / "instances_test_novel.json")
    hard_gt_path = annotations_root / "instances_hard_test.json"
    hard_gt = load_json(hard_gt_path) if hard_gt_path.exists() else None

    train_counts = Counter(a["category_id"] for a in train_gt["annotations"])
    test_counts_base = Counter(a["category_id"] for a in base_test_gt["annotations"])
    test_counts_novel = Counter(a["category_id"] for a in novel_gt["annotations"])
    test_counts_hard = Counter(a["category_id"] for a in hard_gt["annotations"]) if hard_gt else Counter()

    # --- task yaml prompts ---
    base_prompts = load_task_yaml_prompts(REPO_ROOT / "configs" / "tct_ngc" / "tct_ngc_v2_base.yaml")
    novel_prompts = load_task_yaml_prompts(artifacts_dir / "novel_eval.yaml")
    hard_prompts = load_task_yaml_prompts(artifacts_dir / "hard_eval.yaml") if (artifacts_dir / "hard_eval.yaml").exists() else {}

    # --- predictions (single base bbox.json shared across both views) ---
    base_preds = load_json(Path(with_neg_summary["pred_json"]))
    novel_preds = load_json(Path(novel_summary["pred_json"]))
    hard_preds = load_json(Path(hard_summary["pred_json"])) if hard_summary else []

    def agg_preds(preds):
        by_cat = defaultdict(list)
        for p in preds:
            by_cat[p["category_id"]].append(p["score"])
        return {
            cid: {
                "n_predictions": len(scores),
                "mean_score": mean(scores) if scores else 0.0,
            }
            for cid, scores in by_cat.items()
        }

    base_pred_stats = agg_preds(base_preds)
    novel_pred_stats = agg_preds(novel_preds)
    hard_pred_stats = agg_preds(hard_preds)

    # --- build unified rows ---
    rows = []

    for cid, per in sorted(with_neg_per.items()):
        n_train = train_counts.get(cid, 0)
        n_test = test_counts_base.get(cid, 0)
        pred_stat = base_pred_stats.get(cid, {"n_predictions": 0, "mean_score": 0.0})
        label, note = diagnose("base_with_neg", n_train, n_test, per["AP"])
        rows.append({
            "split": "base_with_neg",
            "cat_id": cid,
            "name": per["category_name"],
            "prompt_name": base_prompts.get(cid, per["category_name"]),
            "n_train_anns": n_train,
            "n_test_anns": n_test,
            "AP": per["AP"],
            "AP50": per["AP50"],
            "AP75": per["AP75"],
            "n_predictions": pred_stat["n_predictions"],
            "mean_score": pred_stat["mean_score"],
            "diagnosis_label": label,
            "note": note,
        })

    for cid, per in sorted(no_neg_per.items()):
        n_train = train_counts.get(cid, 0)
        n_test = test_counts_base.get(cid, 0)
        # predictions are SHARED with with_negative — same bbox.json — so we
        # filter here just to record the per-cat stat for this view.
        pred_stat = base_pred_stats.get(cid, {"n_predictions": 0, "mean_score": 0.0})
        ap_with_neg = with_neg_per[cid]["AP"]
        label, note = diagnose("base_no_neg", n_train, n_test, per["AP"], ap_with_neg=ap_with_neg)
        rows.append({
            "split": "base_no_neg",
            "cat_id": cid,
            "name": per["category_name"],
            "prompt_name": base_prompts.get(cid, per["category_name"]),
            "n_train_anns": n_train,
            "n_test_anns": n_test,
            "AP": per["AP"],
            "AP50": per["AP50"],
            "AP75": per["AP75"],
            "n_predictions": pred_stat["n_predictions"],
            "mean_score": pred_stat["mean_score"],
            "diagnosis_label": label,
            "note": note,
        })

    for cid, per in sorted(novel_per.items()):
        n_test = test_counts_novel.get(cid, 0)
        pred_stat = novel_pred_stats.get(cid, {"n_predictions": 0, "mean_score": 0.0})
        label, note = diagnose("novel", 0, n_test, per["AP"])
        rows.append({
            "split": "novel",
            "cat_id": cid,
            "name": per["category_name"],
            "prompt_name": novel_prompts.get(cid, per["category_name"]),
            "n_train_anns": 0,
            "n_test_anns": n_test,
            "AP": per["AP"],
            "AP50": per["AP50"],
            "AP75": per["AP75"],
            "n_predictions": pred_stat["n_predictions"],
            "mean_score": pred_stat["mean_score"],
            "diagnosis_label": label,
            "note": note,
        })

    for cid, per in sorted(hard_per.items()):
        n_test = test_counts_hard.get(cid, 0)
        pred_stat = hard_pred_stats.get(cid, {"n_predictions": 0, "mean_score": 0.0})
        label, note = diagnose("hard", 0, n_test, per["AP"])
        rows.append({
            "split": "hard",
            "cat_id": cid,
            "name": per["category_name"],
            "prompt_name": hard_prompts.get(cid, per["category_name"]),
            "n_train_anns": 0,
            "n_test_anns": n_test,
            "AP": per["AP"],
            "AP50": per["AP50"],
            "AP75": per["AP75"],
            "n_predictions": pred_stat["n_predictions"],
            "mean_score": pred_stat["mean_score"],
            "diagnosis_label": label,
            "note": note,
        })

    for r in rows:
        for k in ("AP", "AP50", "AP75"):
            v = r[k]
            assert isinstance(v, float) and v == v, f"Missing/NaN {k} for row {r}"

    # --- per_class_unified.csv ---
    csv_path = report_dir / "per_class_unified.csv"
    fieldnames = [
        "split", "cat_id", "name", "prompt_name",
        "n_train_anns", "n_test_anns",
        "AP", "AP50", "AP75",
        "n_predictions", "mean_score",
        "diagnosis_label", "note",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # --- per_class_sorted_by_ap.md ---
    sorted_rows = sorted(rows, key=lambda r: r["AP"])
    md_lines = [
        "# TCT_NGC V2 dev30 — Per-class AP (sorted ascending)",
        "",
        f"All {len(rows)} rows across the 4 eval splits, sorted by AP.",
        "",
        "| # | Split | ID | Class | n_train | n_test | AP | AP50 | AP75 | n_preds | mean_score | Label |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, r in enumerate(sorted_rows, 1):
        md_lines.append(
            f"| {i} | {r['split']} | {r['cat_id']} | {r['name']} | {r['n_train_anns']} | {r['n_test_anns']} | "
            f"{r['AP']:.4f} | {r['AP50']:.4f} | {r['AP75']:.4f} | {r['n_predictions']} | "
            f"{r['mean_score']:.3f} | {r['diagnosis_label']} |"
        )
    (report_dir / "per_class_sorted_by_ap.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # --- prompt drift: same predictions; ΔAP arises from COCOeval scope alone ---
    shared_ids = sorted(set(with_neg_per) & set(no_neg_per))
    drift_entries = []
    for cid in shared_ids:
        ap_with = with_neg_per[cid]["AP"]
        ap_no = no_neg_per[cid]["AP"]
        preds_here = [p for p in base_preds if p["category_id"] == cid]
        drift_entries.append({
            "cat_id": cid,
            "name": with_neg_per[cid]["category_name"],
            "AP_with_negative": ap_with,
            "AP_no_negative": ap_no,
            "delta_AP": ap_no - ap_with,
            "n_preds": len(preds_here),
            "mean_score": mean(p["score"] for p in preds_here) if preds_here else 0.0,
        })
    drift_entries.sort(key=lambda e: e["delta_AP"])
    drift_payload = {
        "shared_category_count": len(shared_ids),
        "negative_class_ids_excluded": eval_metadata.get("negative_class_ids_for_view", []),
        "note": "Both AP_with_negative and AP_no_negative come from the SAME bbox.json + GT. The delta reflects only COCOeval scope (params.catIds), not different image sets or different inference runs.",
        "entries": drift_entries,
    }
    dump_json(report_dir / "prompt_drift_analysis.json", drift_payload)

    # --- novel class-agnostic recall ---
    print("Computing class-agnostic recall for novel split (may take ~15s)...")
    novel_recall = class_agnostic_recall(novel_gt["annotations"], novel_preds)
    true_ar100 = novel_summary["metrics"].get("AR@100")
    if true_ar100 is None:
        ar_line = [
            ln for ln in novel_summary["coco_summary"].splitlines()
            if "maxDets=100" in ln and "Recall" in ln and "area=   all" in ln
        ]
        if ar_line:
            true_ar100 = float(ar_line[0].split("=")[-1].strip())

    novel_recall_payload = {
        "class_agnostic_recall_iou50": novel_recall["recall"],
        "class_agnostic_matched": novel_recall["matched"],
        "class_agnostic_total": novel_recall["total"],
        "true_AR_100_all": true_ar100,
        "classification_error_fraction": (novel_recall["recall"] - (true_ar100 or 0.0)),
        "per_class_preds": [
            {
                "cat_id": cid,
                "name": novel_per[cid]["category_name"],
                "n_preds": novel_pred_stats.get(cid, {}).get("n_predictions", 0),
                "mean_score": novel_pred_stats.get(cid, {}).get("mean_score", 0.0),
                "AP": novel_per[cid]["AP"],
            }
            for cid in sorted(novel_per)
        ],
    }
    dump_json(report_dir / "novel_recall_analysis.json", novel_recall_payload)
    if true_ar100 is not None:
        assert novel_recall["recall"] + 1e-6 >= true_ar100, (
            f"class-agnostic recall {novel_recall['recall']:.3f} < true AR@100 {true_ar100:.3f}"
        )

    # --- comprehensive markdown report ---
    def fmt(x, prec=4):
        return f"{x:.{prec}f}"

    summary_table = [
        "| Split | Images | Categories | AP | AP50 | AP75 | APs | APm | APl |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    splits_for_table = [
        ("base_with_negative", with_neg_summary),
        ("base_no_negative", no_neg_summary),
        ("novel", novel_summary),
    ]
    if hard_summary is not None:
        splits_for_table.append(("hard", hard_summary))
    for name, summary in splits_for_table:
        m = summary["metrics"]
        summary_table.append(
            f"| {name} | {summary['num_images']} | {summary['num_categories']} | "
            f"{fmt(m['AP'])} | {fmt(m['AP50'])} | {fmt(m['AP75'])} | "
            f"{fmt(m['APs'])} | {fmt(m['APm'])} | {fmt(m['APl'])} |"
        )

    per_class_table = [
        "| Split | ID | Class | n_train | n_test | AP | AP50 | AP75 | n_preds | mean_score | Label |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        per_class_table.append(
            f"| {r['split']} | {r['cat_id']} | {r['name']} | {r['n_train_anns']} | {r['n_test_anns']} | "
            f"{fmt(r['AP'])} | {fmt(r['AP50'])} | {fmt(r['AP75'])} | {r['n_predictions']} | "
            f"{fmt(r['mean_score'], 3)} | {r['diagnosis_label']} |"
        )

    drift_worst = [e for e in drift_entries if e["delta_AP"] < 0][:5]
    drift_section = []
    for e in drift_worst:
        drift_section.append(
            f"- **id={e['cat_id']} `{e['name']}`** — "
            f"ΔAP={e['delta_AP']:+.4f} "
            f"(with={fmt(e['AP_with_negative'])} → no={fmt(e['AP_no_negative'])}), "
            f"n_preds={e['n_preds']}, mean_score={fmt(e['mean_score'], 3)}"
        )
    if not drift_section:
        drift_section.append("- 无类的 ΔAP < 0（去掉 negative 类后 base AP 全部上升或持平）")

    n_negative = len(eval_metadata.get("negative_class_ids_for_view", []))
    n_no_neg_classes = len(no_neg_per)

    analysis_md = []
    analysis_md.append("# TCT_NGC V2 dev30 — Per-class Analysis")
    analysis_md.append("")
    analysis_md.append("生成脚本: `tools/analyze_tct_ngc_v2_eval.py`")
    analysis_md.append("")
    analysis_md.append(
        "本报告不重新跑任何评测，只对已有的 `bbox.json` + `per_class.csv` + `summary.json` 做二次分析。"
    )
    analysis_md.append(
        f"`base_with_negative` 与 `base_no_negative` 来自**同一个 bbox.json + 同一个 GT**，"
        f"区别仅在于 COCOeval 的 `params.catIds` 是否限制到 {n_no_neg_classes} 个非负类（去掉 {n_negative} 个 ontology=negative 类）。"
    )
    analysis_md.append("")
    analysis_md.append("## 1. 总体指标汇总")
    analysis_md.append("")
    analysis_md.extend(summary_table)
    analysis_md.append("")
    analysis_md.append("## 2. Per-class 完整表")
    analysis_md.append("")
    analysis_md.append(f"全 {len(rows)} 行，每行来自对应 split 的 `*_per_class.csv`。")
    analysis_md.append("`diagnosis_label` 规则：")
    analysis_md.append(f"- `small_sample`: n_train_anns < {SMALL_SAMPLE_TRAIN_THRESHOLD}")
    analysis_md.append(f"- `model_issue`: 训练样本充足但 AP < {LOW_AP_THRESHOLD}")
    analysis_md.append("- `prompt_drift`: base_no_neg 相比 base_with_neg 的 ΔAP < -0.02 且基线可用")
    analysis_md.append(f"- `ok`: AP ≥ {OK_AP_THRESHOLD}")
    analysis_md.append("- `moderate`: 介于上述之间")
    analysis_md.append("- `zero_shot_fail` / `zero_shot_ok`: novel / hard 专用")
    analysis_md.append("")
    analysis_md.extend(per_class_table)
    analysis_md.append("")
    analysis_md.append("> 排序版见 [per_class_sorted_by_ap.md](per_class_sorted_by_ap.md)。")
    analysis_md.append("")
    analysis_md.append("## 3. Prompt Drift（同推理结果，仅 COCOeval 缩范围）")
    analysis_md.append("")
    analysis_md.append(f"在 {len(shared_ids)} 个共同类中 ΔAP 最负的 5 类：")
    analysis_md.append("")
    analysis_md.extend(drift_section)
    analysis_md.append("")
    analysis_md.append(
        "由于本视图未改变图像集和预测结果，ΔAP 的来源**仅是 COCOeval 评估范围的缩小**："
        "去掉负类 prompt 后，模型对剩余类的精度估计可能因背景类别变少而抬升，"
        "也可能因负类预测被算作 FP 而下降——这是 GLIP 类别口径相关的纯统计现象。"
    )
    analysis_md.append("")
    analysis_md.append("## 4. Novel Zero-shot recall 分析")
    analysis_md.append("")
    novel_cls_agnostic = novel_recall_payload["class_agnostic_recall_iou50"]
    analysis_md.append(f"- 官方 AP = **{fmt(novel_summary['metrics']['AP'])}** (AP50 {fmt(novel_summary['metrics']['AP50'])})")
    if true_ar100 is not None:
        analysis_md.append(f"- 官方 AR@100 = **{true_ar100:.3f}**")
    analysis_md.append(
        f"- **类别无关 recall @ IoU=0.5** = **{novel_cls_agnostic:.3f}** "
        f"({novel_recall['matched']}/{novel_recall['total']} GT 框被任意类别预测覆盖)"
    )
    analysis_md.append("")
    analysis_md.append(
        "类别无关 recall 与 AR@100 的差距可视为模型『检到框但分错类』的比例上界，"
        "也是 novel prompt 与预训练词汇空间不对齐带来的分类损失。"
    )
    analysis_md.append("")

    (report_dir / "TCT_NGC_V2_ANALYSIS.md").write_text("\n".join(analysis_md) + "\n", encoding="utf-8")

    print(f"Wrote report tree under: {report_dir}")


if __name__ == "__main__":
    main()
