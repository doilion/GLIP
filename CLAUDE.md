# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

GLIP (Grounded Language-Image Pre-training) — CVPR 2022. Vision-language object detection / phrase grounding model. Code is a fork of `maskrcnn_benchmark` extended with a language branch and deep vision-language fusion. Configs and weights cover GLIP-T (A/B/C) and GLIP-L variants; downstream targets are COCO, LVIS, Flickr30K, and ODinW (13 / 35 task benchmarks).

## Setup

The repo ships a custom CUDA extension (`maskrcnn_benchmark._C`), so a one-time build is required after any environment change:

```bash
pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo transformers
python setup.py build develop --user
```

`setup.py` compiles CPU/CUDA sources from `maskrcnn_benchmark/csrc/` and installs the package in-place (the resulting `.so` already lives at `maskrcnn_benchmark/_C.cpython-38-x86_64-linux-gnu.so`). Requires PyTorch ≥ 1.9 with a matching CUDA toolchain; the README recommends the `pengchuanzhang/maskrcnn` or `pengchuanzhang/pytorch` Docker images.

Backbone checkpoints (Swin) are expected under `MODEL/`. Datasets are expected under `DATASET/` (see [DATA.md](DATA.md) — Objects365, Flickr30K, MixedGrounding, COCO, LVIS, ODinW each have distinct layouts; Objects365 in particular uses a TSV format).

## Common commands

All training/eval scripts are dispatched from `tools/` and consume a YAML config plus a list of `KEY VALUE` overrides interpreted by `yacs`. Pass `--help` to any tool for the full flag list — the patterns below are the ones the README/configs actually exercise.

### Pre-training (distributed)
```bash
python -m torch.distributed.launch --nnodes 2 --nproc_per_node=16 tools/train_net.py \
    --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml \
    --skip-test --use-tensorboard --override_output_dir {output_dir}
```

### Zero-shot / standard evaluation
```bash
# COCO
python tools/test_grounding_net.py --config-file {config} --weight {ckpt} \
    TEST.IMS_PER_BATCH 1 MODEL.DYHEAD.SCORE_AGG "MEAN" \
    TEST.EVAL_TASK detection MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
    OUTPUT_DIR {output_dir}

# LVIS — uses MDETR FixedAP; runs distributed with chunked class evaluation
python -m torch.distributed.launch --nproc_per_node=4 tools/test_grounding_net.py \
    --config-file {config} --task_config configs/lvis/minival.yaml --weight {ckpt} \
    TEST.EVAL_TASK detection TEST.CHUNKED_EVALUATION 40 \
    TEST.MDETR_STYLE_AGGREGATE_CLASS_NUM 3000 ...

# Flickr30K phrase grounding
python tools/test_grounding_net.py --config-file {config} \
    --task_config configs/flickr/test.yaml,configs/flickr/val.yaml \
    --weight {ckpt} TEST.EVAL_TASK grounding \
    MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False ...
```

### ODinW / custom COCO-format dataset
Per-dataset yaml lives in `configs/odinw_13/` and `configs/odinw_35/`. Three downstream regimes (selected via `SOLVER.TUNING_HIGHLEVEL_OVERRIDE`): `full`, `language_prompt_v2` (prompt tuning), `linear_prob` (linear probing).

`{custom_shot_and_epoch_and_general_copy}` encodes few-shot regime: `1_200_8`, `3_200_4`, `5_200_2`, `10_200_1`, or `0_200_1` for full-data. For full-data also set `SOLVER.STEP_PATIENCE 2 SOLVER.AUTO_TERMINATE_PATIENCE 4`.

```bash
python -m torch.distributed.launch --nproc_per_node=4 tools/finetune.py \
    --config-file {config_file} --ft-tasks {odinw_configs} --skip-test \
    --custom_shot_and_epoch_and_general_copy {regime} \
    --evaluate_only_best_on_test --push_both_val_and_test \
    MODEL.WEIGHT {ckpt} ...
```

Download all 35 ODinW datasets in one shot: `python odinw/download_datasets.py`.

Knowledge-augmented inference uses `knowledge/odinw_benchmark35_knowledge_and_gpt3.yaml` together with `GLIPKNOW.*` overrides — see the README "Knowledge-Augmented Inference" section for the exact invocation.

## Architecture

### Two detection meta-architectures
`maskrcnn_benchmark/modeling/detector/__init__.py` registers exactly two top-level models, picked by `cfg.MODEL.META_ARCHITECTURE`:
- `GeneralizedRCNN` — classical detector (no language branch).
- `GeneralizedVLRCNN` ([generalized_vl_rcnn.py](maskrcnn_benchmark/modeling/detector/generalized_vl_rcnn.py)) — the GLIP model. Composes a visual `backbone` + `language_backbone` + `rpn` + (optional) `roi_heads`. Forward takes both images and a tokenized caption; an optional `random_word` helper supports MLM masking on the language stream with a `greenlight_map` that controls which tokens may be masked / contribute to the MLM loss.

GLIP configs use `META_ARCHITECTURE: GeneralizedVLRCNN`. The RPN module of choice is `VLDyHead` ([rpn/vldyhead.py](maskrcnn_benchmark/modeling/rpn/vldyhead.py)) which performs deep cross-modal fusion; ATSS / FCOS / Retina / classic RPN heads also live alongside it in [rpn/](maskrcnn_benchmark/modeling/rpn/).

### Language branch
`maskrcnn_benchmark/modeling/language_backbone/` supplies BERT (`bert_model.py`, via HuggingFace `transformers`), CLIP (`clip_model.py` + `simple_tokenizer.py`), and a small RNN encoder. `build.py` selects which one based on `cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE`.

### Config system
`yacs`-based. The full default tree is in [maskrcnn_benchmark/config/defaults.py](maskrcnn_benchmark/config/defaults.py); dataset path resolution goes through [maskrcnn_benchmark/config/paths_catalog.py](maskrcnn_benchmark/config/paths_catalog.py). Configs in `configs/pretrain/` define training recipes; `configs/lvis/`, `configs/flickr/`, `configs/odinw_*` are *task* configs passed via `--task_config` and merged on top. Trailing CLI `KEY VALUE` pairs are last-write-wins overrides — this is how scripts toggle e.g. `MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS`, `TEST.EVAL_TASK`, `SOLVER.TUNING_HIGHLEVEL_OVERRIDE`.

### Engine / data
- Training loops: `maskrcnn_benchmark/engine/trainer.py` (standard) plus specialized variants (`alter_trainer.py`, `singlepath_trainer.py`, `stage_trainer.py`, `evolution.py`).
- Inference + the grounding-style predictor live in `engine/inference.py` and `engine/predictor_glip.py`.
- Data pipeline: `maskrcnn_benchmark/data/build.py` is the entry point; concrete datasets are under `data/datasets/`, transforms under `data/transforms/`, and samplers (including chunked / distributed flavors used for LVIS large-vocabulary eval) under `data/samplers/`.

### Entry-point scripts
- `tools/train_net.py` — pre-training and standard fine-tuning. Sets `CUDA_LAUNCH_BLOCKING=1` at module load.
- `tools/finetune.py` — ODinW / few-shot fine-tuning loop with auto-step LR, early termination, and the `--custom_shot_and_epoch_and_general_copy` shorthand.
- `tools/test_grounding_net.py` — evaluation for COCO / LVIS / Flickr / ODinW. Picks behavior off `TEST.EVAL_TASK` (`detection` vs `grounding`).
- `tools/test_grounding_net_exclude_negative.py` — same flow as `test_grounding_net.py`, plus re-evaluates the dumped `bbox.json` with a configurable set of "negative" categories removed from both predictions and `COCOeval.params.catIds`. Defaults to `DEFAULT_NEGATIVE_CLASS_NAMES` (cytology / TCT line, mirroring WeDetect's `test_exclude_negative.py`); override via `--exclude-class-names "name1,name2"` or `--exclude-coco-ids "5,7"`. `--skip-inference` re-evals existing dumps. Writes `bbox.exclude_negative.json` and `bbox.exclude_negative.summary.json` next to the original. Implementation in [maskrcnn_benchmark/data/datasets/evaluation/coco/exclude_class_coco_eval.py](maskrcnn_benchmark/data/datasets/evaluation/coco/exclude_class_coco_eval.py).
- `tools/test_grounding_net_organ_prior.py` — applies a per-image organ / tissue prior at eval time. Each image's organ is parsed from its filename (default rule: `<prefix>__<rest>`, with prefixes like `Thyroid_gland`, `Urine`, `TCT_CCD` mapped to WeDetect's organ taxonomy; override via `--path-prefix-map '{"prefix": "Organ name"}'`). Predictions whose class belongs to a different organ are dropped before COCOeval, then per-organ AP + macro + instance-weighted aggregates are reported alongside the all-class flat number. `--compare-baseline` additionally runs a prior-OFF pass (written as `bbox.organ_prior_off.*` alongside the `bbox.organ_prior.*` files; YOLOE-style A/B). `--taxonomy` accepts WeDetect's `tct_ngc_taxonomy.json` schema or the `.pt` mask file built by `tools/build_class_organ_mask.py`. Mirrors WeDetect's `OrganRestrictedCocoMetric` and YOLOE's `eval_domain_prior_infer_tct_ngc.py` but operates post-hoc. Implementation in [maskrcnn_benchmark/data/datasets/evaluation/coco/organ_restricted_coco_eval.py](maskrcnn_benchmark/data/datasets/evaluation/coco/organ_restricted_coco_eval.py); regression tests in [tests/test_eval_priors.py](tests/test_eval_priors.py) (run standalone: `python tests/test_eval_priors.py`).
- `tools/test_net.py`, `tools/eval_all.py`, `tools/visualize_grounding_net.py` — additional utilities.

## Gotchas

- The compiled extension is pinned to `cpython-38-x86_64-linux-gnu`. Switching Python versions requires rebuilding via `python setup.py build develop --user`.
- Many model-zoo download links in the README are expired; current checkpoints live at https://huggingface.co/GLIPModel/GLIP and https://huggingface.co/harold/GLIP.
- For ODinW custom datasets you must strip the `id:0` background category from the COCO annotation JSON, and set `MODEL.*.NUM_CLASSES` to `num_real_categories + 1` (background still counted in this field).
- LVIS evaluation requires `TEST.MDETR_STYLE_AGGREGATE_CLASS_NUM 3000` and chunked evaluation; without these large-vocabulary numbers will be wrong.
- Distributed launches use the legacy `torch.distributed.launch` API.
