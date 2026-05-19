#!/usr/bin/env python
"""Render the human-facing markdown report for a TCT_NGC v2 dev30 evaluation.

Consumes the per-split summaries produced by ``summarize_tct_ngc_v2_eval.py``
and the asset metadata produced by ``prepare_tct_ngc_v2_eval_assets.py``.

The base eval is presented as **two views over the same inference run**:
``base_with_negative`` (full 30-class scoring) and ``base_no_negative``
(post-hoc COCOeval ``params.catIds`` filter dropping the
``ontology=negative`` classes). Both views share the same predictions JSON
and the same image set; only the scored category subset differs.
"""

import argparse
import json
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-json", required=True,
                        help="eval_metadata.json from prepare_tct_ngc_v2_eval_assets.py")
    parser.add_argument("--base-with-negative-summary-json", required=True)
    parser.add_argument("--base-no-negative-summary-json", required=True)
    parser.add_argument("--novel-summary-json", required=True)
    parser.add_argument("--hard-summary-json", default=None,
                        help="Optional hard-eval summary.json")
    parser.add_argument("--probe-json", default=None,
                        help="Optional batch/worker probe artifact JSON.")
    parser.add_argument("--out-markdown", required=True)
    parser.add_argument("--out-json", required=True)
    return parser


def fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_overall_table(splits):
    lines = [
        "| Split | Images | Categories | AP | AP50 | AP75 | APs | APm | APl |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in splits:
        m = summary["metrics"]
        lines.append(
            f"| {label} | {summary['num_images']} | {summary['num_categories']} | "
            f"{fmt(m['AP'])} | {fmt(m['AP50'])} | {fmt(m['AP75'])} | "
            f"{fmt(m['APs'])} | {fmt(m['APm'])} | {fmt(m['APl'])} |"
        )
    return "\n".join(lines)


def render_speed_table(splits):
    lines = [
        "| Split | 8 卡 Batch | Workers | FP16 | Wallclock(s) | Img/s (wallclock) | s/img/device |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for label, summary in splits:
        runtime = summary.get("runtime", {})
        lines.append(
            f"| {label} | {runtime.get('test_batch', '-')} | {runtime.get('num_workers', '-')} | "
            f"{runtime.get('use_amp', '-')} | {fmt(runtime.get('elapsed_seconds', 0))} | "
            f"{fmt(runtime.get('images_per_second_wallclock', 0.0))} | "
            f"{fmt(runtime.get('seconds_per_image_per_device', 0.0))} |"
        )
    return "\n".join(lines)


def main():
    args = build_parser().parse_args()

    metadata = json.loads(Path(args.metadata_json).read_text(encoding="utf-8"))
    base_with_neg = json.loads(Path(args.base_with_negative_summary_json).read_text(encoding="utf-8"))
    base_no_neg = json.loads(Path(args.base_no_negative_summary_json).read_text(encoding="utf-8"))
    novel = json.loads(Path(args.novel_summary_json).read_text(encoding="utf-8"))
    hard = (
        json.loads(Path(args.hard_summary_json).read_text(encoding="utf-8"))
        if args.hard_summary_json else None
    )
    probe = (
        json.loads(Path(args.probe_json).read_text(encoding="utf-8"))
        if args.probe_json else None
    )

    # Sanity: provenance of the post-hoc view.
    assert base_with_neg["num_images"] == base_no_neg["num_images"], (
        f"Image-set drift between base views: {base_with_neg['num_images']} vs "
        f"{base_no_neg['num_images']}. base_no_negative must be a COCOeval-catIds "
        "view, not a different annotation file."
    )
    assert base_with_neg["pred_json"] == base_no_neg["pred_json"], (
        "Both base views must derive from the same predictions JSON."
    )

    n_negative = len(metadata.get("negative_class_ids_for_view", []))
    n_no_neg_classes = base_no_neg["num_categories"]
    n_with_neg_classes = base_with_neg["num_categories"]

    splits_for_overall = [
        (f"base_with_negative ({n_with_neg_classes} cats)", base_with_neg),
        (f"base_no_negative ({n_no_neg_classes} cats, view)", base_no_neg),
        (f"novel ({novel['num_categories']} cats)", novel),
    ]
    splits_for_speed = [
        ("base", base_with_neg),
        ("novel", novel),
    ]
    if hard is not None:
        splits_for_overall.append((f"hard ({hard['num_categories']} cats)", hard))
        splits_for_speed.append(("hard", hard))

    overall_table = render_overall_table(splits_for_overall)
    speed_table = render_speed_table(splits_for_speed)

    probe_section = ""
    if probe is not None:
        probe_section = "\n".join([
            "- Batch / Worker 探测结果：",
            f"  - 选中 global batch: {probe.get('selected_batch_size', '-')}",
            f"  - 选中 dataloader workers: {probe.get('selected_num_workers', '-')}",
            f"  - 探测批次数: {probe.get('subset_batches', '-')}",
        ])

    base_command = Path(base_with_neg["runtime"]["command_file"]).resolve() \
        if base_with_neg.get("runtime", {}).get("command_file") else "(未记录)"
    novel_command = Path(novel["runtime"]["command_file"]).resolve() \
        if novel.get("runtime", {}).get("command_file") else "(未记录)"

    report = f"""# TCT_NGC V2 dev30 Evaluation Report

## 1. 实验目的

对 TCT_NGC V2 dev30 的最终训练模型执行正式评估：

- Base 评估：在 `{Path(metadata['splits']['base_with_negative']['ann']).name}` 上完成一次 8 卡推理；从同一份预测同时给出 `base_with_negative`（{n_with_neg_classes} 类）与 `base_no_negative`（{n_no_neg_classes} 类）两个视图。
- Novel 评估：在 `{Path(metadata['splits']['novel']['ann']).name}` 上完成 novel-only 评估（{novel['num_categories']} 类）。
{f"- Hard 评估：在 `{Path(metadata['splits']['hard']['ann']).name}` 上完成 hard-only 评估（{hard['num_categories']} 类）。" if hard is not None else ""}

## 2. 模型与 Checkpoint

- Checkpoint: `{metadata["weight"]}`
- 训练配置快照: `{metadata["checkpoint_config"]}`

## 3. 数据与划分

- 数据根目录: `{metadata["dataset_root"]}`
- Base 评估注释文件: `{metadata["splits"]["base_with_negative"]["ann"]}`
- Novel 评估注释文件: `{metadata["splits"]["novel"]["ann"]}`
{f"- Hard 评估注释文件: `{metadata['splits']['hard']['ann']}`" if hard is not None else ""}
- Base 评估任务配置: `{metadata["splits"]["base_with_negative"]["yaml"]}`
- Novel 评估任务配置: `{metadata["splits"]["novel"]["yaml"]}`

### 3.1 Base 的 `with_negative` / `no_negative` 两个视图

`base_no_negative` **不再单独跑一次推理，也不再修改注释文件**。它是同一份 `bbox.json` + 同一个 GT 上，通过把 COCOeval 的 `params.catIds` 限制到去掉 {n_negative} 个 `ontology=negative` 类（保留 {n_no_neg_classes} 类）后的 post-hoc 视图。这样：

- 两个视图的图像集完全相同（{base_with_neg["num_images"]} 张），任务定义没有偏移；
- 两个视图都能给出标准 COCO 指标（AP/AP50/AP75/APs/APm/APl），可直接对比；
- 想要新的子集视图，只需把对应 cat-id 列表传给 `summarize_tct_ngc_v2_eval.py --cat-ids-subset`，无需再次推理。

被排除出 `no_negative` 视图的负类（GT cat id）：

{json.dumps(metadata.get("negative_class_names_for_view", []), ensure_ascii=False, indent=2)}

## 4. Novel / Hard 评估方案

Novel 与 Hard 沿用各自的原始 `OVERRIDE_CATEGORY` 长 prompt（`metadata/category_prompt_name_map_v2.json` 中 `prompt_name`）。这些类没有训练样本，AP 反映 GLIP 的开放词汇 zero-shot 能力。

## 5. 8 卡评估加速策略

- 启动方式：`torchrun --nproc_per_node=8`
- 评估模型构建配置：使用训练完成后的 `ft_task_1/config.yml`
- 精度策略：开启 FP16 (`TEST.USE_AMP=True`)
{probe_section}

## 6. 运行命令

Base 与 Novel 评估命令均保存到各自输出目录下的 `command.sh`：

- Base command: `{base_command}`
- Novel command: `{novel_command}`

## 7. 实验结果

### 7.1 总体指标

{overall_table}

### 7.2 速度信息

{speed_table}

## 8. 结果分析

- `base_with_negative` 与 `base_no_negative` 在同一份预测上做 COCOeval，差异**仅来自评估范围（catIds）**。两者 AP 的差值反映的是该 checkpoint 在去掉 {n_negative} 个负类干扰后的相对表现，不存在图像集变化或重新推理带来的偏置。
- `novel`{"/`hard`" if hard is not None else ""} 评估反映模型在零样本 prompt 上的开放词汇能力。AP 偏低时建议结合 `prompt_drift_analysis.json` 与 `novel_recall_analysis.json` 区分『检测能力不足』与『分类不对齐』。

## 9. 最终结论

- Base 评估完成（with_negative + no_negative 双视图来自同一推理）
- Novel 单独评估完成{("，Hard 单独评估完成" if hard is not None else "")}
- 评估均使用 8 GPUs 执行；结果、日志、命令与文档已落盘可复现
"""

    out_markdown = Path(args.out_markdown)
    out_markdown.parent.mkdir(parents=True, exist_ok=True)
    out_markdown.write_text(report, encoding="utf-8")

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": metadata["weight"],
        "checkpoint_config": metadata["checkpoint_config"],
        "splits": {
            "base_with_negative": base_with_neg,
            "base_no_negative": base_no_neg,
            "novel": novel,
        },
        "negative_class_ids_for_view": metadata.get("negative_class_ids_for_view", []),
        "negative_class_names_for_view": metadata.get("negative_class_names_for_view", []),
    }
    if hard is not None:
        payload["splits"]["hard"] = hard
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {out_markdown}\nWrote: {out_json}")


if __name__ == "__main__":
    main()
