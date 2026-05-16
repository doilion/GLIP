"""Evaluate GLIP on a COCO-format task with a configurable list of
"negative" classes excluded from the mAP.

Port of WeDetect's ``test_exclude_negative.py``. Two modes:

1. **Default (full pipeline)**: behaves like ``tools/test_grounding_net.py`` —
   builds the model, runs inference, dumps ``bbox.json`` — and then
   additionally re-evaluates the predictions with the excluded categories
   removed from both the predictions and ``COCOeval.params.catIds``.
   Original ``bbox.json`` is left intact; the filtered evaluation writes
   ``bbox.exclude_negative.json`` next to it.

2. **``--skip-inference``**: re-evaluate one or more previously-dumped
   ``bbox.json`` files against the GT annotation file resolved from the
   config, without rebuilding the model. Cheap, deterministic.

Examples
--------

Full run on an ODinW task (defaults to the cytology / TCT negative-class list
defined in ``exclude_class_coco_eval.DEFAULT_NEGATIVE_CLASS_NAMES``)::

    python tools/test_grounding_net_exclude_negative.py \
        --config-file {config_file} --weight {model_checkpoint} \
        --task_config {odinw_configs} \
        TEST.IMS_PER_BATCH 1 TEST.EVAL_TASK detection \
        DATASETS.USE_OVERRIDE_CATEGORY True DATASETS.USE_CAPTION_PROMPT True

Custom exclude list (overrides the default)::

    ... --exclude-class-names "Urine-Negative,Thyroid gland-Negative samples"

Re-eval an existing dump::

    python tools/test_grounding_net_exclude_negative.py \
        --config-file {config_file} --weight {model_checkpoint} \
        --task_config {odinw_configs} \
        --skip-inference
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (do not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa: F401  isort:skip

import argparse
import datetime
import io
import json
import logging
import os
from contextlib import redirect_stdout

import torch
import torch.distributed as dist

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.data.datasets.evaluation.coco.exclude_class_coco_eval import (
    DEFAULT_NEGATIVE_CLASS_NAMES,
    evaluate_predictions_on_coco_with_exclusion,
    resolve_exclude_coco_ids,
)
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.comm import get_rank, is_main_process, synchronize
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir


def init_distributed_mode(args):
    """Same init as ``tools/test_grounding_net.py``."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(0, 7200),
    )
    dist.barrier()


def parse_exclude_args(args, logger):
    """Resolve --exclude-class-names / --exclude-coco-ids / defaults into a list."""
    inputs = []
    if args.exclude_class_names is not None:
        inputs.extend(
            n.strip() for n in args.exclude_class_names.split(",") if n.strip()
        )
    if args.exclude_coco_ids is not None:
        for token in args.exclude_coco_ids.split(","):
            token = token.strip()
            if token:
                inputs.append(int(token))

    if not inputs:
        logger.info(
            "exclude-class: no CLI overrides given — using DEFAULT_NEGATIVE_CLASS_NAMES "
            "(%d entries)",
            len(DEFAULT_NEGATIVE_CLASS_NAMES),
        )
        inputs = list(DEFAULT_NEGATIVE_CLASS_NAMES)
    return inputs


def run_exclude_eval_for_dataset(dataset, output_folder, exclude_inputs, iou_types, logger):
    """Re-evaluate the dumped ``{iou_type}.json`` files with class exclusion."""
    if dataset.coco is None:
        logger.warning(
            "exclude-class: dataset %r has no coco api (TSVDataset?) — skipping",
            type(dataset).__name__,
        )
        return

    resolved_ids, resolved_names = resolve_exclude_coco_ids(
        dataset.coco, exclude_inputs, logger
    )
    if not resolved_ids:
        logger.warning("exclude-class: no valid excluded categories — skipping")
        return

    logger.info(
        "exclude-class: dataset %r — excluding %d categories %s (coco_ids=%s)",
        getattr(dataset, "ann_file", "?"),
        len(resolved_ids),
        resolved_names,
        resolved_ids,
    )

    for iou_type in iou_types:
        bbox_json = os.path.join(output_folder, f"{iou_type}.json")
        if not os.path.isfile(bbox_json):
            logger.warning(
                "exclude-class: expected %s not found — skipping %s",
                bbox_json,
                iou_type,
            )
            continue

        with open(bbox_json, "r") as f:
            preds = json.load(f)

        out_json = os.path.join(output_folder, f"{iou_type}.exclude_negative.json")

        # Capture COCOeval.summarize() output so it lands in the logger /
        # log file instead of only on stdout.
        buf = io.StringIO()
        with redirect_stdout(buf):
            coco_eval = evaluate_predictions_on_coco_with_exclusion(
                coco_gt=dataset.coco,
                coco_results=preds,
                json_result_file=out_json,
                iou_type=iou_type,
                exclude_coco_ids=resolved_ids,
                classwise=True,
                logger=logger,
            )
        summary = buf.getvalue().strip()
        if summary:
            logger.info("[%s] COCOeval summary (excluded negatives):\n%s", iou_type, summary)

        if coco_eval is not None:
            stats = list(coco_eval.stats)
            logger.info(
                "[%s] mAP_copypaste (excluded negatives): "
                + " ".join(f"{s:.4f}" for s in stats[:6]),
                iou_type,
            )
            # Persist the headline numbers in a tiny JSON for downstream
            # logging / dashboards.
            meta_path = os.path.join(
                output_folder, f"{iou_type}.exclude_negative.summary.json"
            )
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "excluded_class_names": resolved_names,
                        "excluded_coco_ids": resolved_ids,
                        "stats": stats,
                        "stat_keys": [
                            "AP",
                            "AP50",
                            "AP75",
                            "APs",
                            "APm",
                            "APl",
                            "ARmax1",
                            "ARmax10",
                            "ARmax100",
                            "ARs",
                            "ARm",
                            "ARl",
                        ],
                    },
                    f,
                    indent=2,
                )


def main():
    parser = argparse.ArgumentParser(
        description="GLIP grounding-net evaluation with negative classes excluded"
    )
    parser.add_argument("--config-file", required=True, metavar="FILE")
    parser.add_argument("--weight", default=None, metavar="FILE")
    parser.add_argument("--task_config", default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world-size", default=1, type=int)
    parser.add_argument("--dist-url", default="env://")

    parser.add_argument(
        "--exclude-class-names",
        default=None,
        help="Comma-separated category names to exclude. Replaces the default "
        "DEFAULT_NEGATIVE_CLASS_NAMES list. Combine with --exclude-coco-ids.",
    )
    parser.add_argument(
        "--exclude-coco-ids",
        default=None,
        help="Comma-separated COCO category IDs to exclude (additive with "
        "--exclude-class-names).",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Do not run the model. Re-evaluate existing {iou_type}.json files "
        "under {output_dir}/eval/{weight}/inference/{dataset}/ instead.",
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line (yacs KEY VALUE pairs)",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    if distributed:
        init_distributed_mode(args)

    cfg.local_rank = args.local_rank
    cfg.num_gpus = num_gpus

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    log_dir = cfg.OUTPUT_DIR
    if args.weight:
        log_dir = os.path.join(
            log_dir, "eval", os.path.splitext(os.path.basename(args.weight))[0]
        )
    if log_dir:
        mkdir(log_dir)

    logger = setup_logger("maskrcnn_benchmark", log_dir, get_rank())
    logger.info(args)
    logger.info("Using %d GPUs", num_gpus)
    logger.info(cfg)

    # ---------- model & checkpoint ----------
    model = None
    if not args.skip_inference:
        model = build_detection_model(cfg)
        model.to(cfg.MODEL.DEVICE)
        checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
        if args.weight:
            checkpointer.load(args.weight, force=True)
        else:
            checkpointer.load(cfg.MODEL.WEIGHT)

    exclude_inputs = parse_exclude_args(args, logger)

    # ---------- iterate over task configs (mirrors test_grounding_net.py) ----------
    task_configs = args.task_config.split(",") if args.task_config else [None]
    for task_config in task_configs:
        if task_config is not None:
            cfg_ = cfg.clone()
            cfg_.defrost()
            cfg_.merge_from_file(task_config)
            cfg_.merge_from_list(args.opts)
        else:
            cfg_ = cfg

        iou_types = ("bbox",)
        if cfg_.MODEL.MASK_ON:
            iou_types = iou_types + ("segm",)
        if cfg_.MODEL.KEYPOINT_ON:
            iou_types = iou_types + ("keypoints",)

        dataset_names = cfg_.DATASETS.TEST
        if isinstance(dataset_names[0], (list, tuple)):
            dataset_names = [d for group in dataset_names for d in group]

        output_folders = []
        for dataset_name in dataset_names:
            of = os.path.join(log_dir, "inference", dataset_name) if log_dir else None
            if of is not None:
                mkdir(of)
            output_folders.append(of)

        data_loaders_val = make_data_loader(
            cfg_, is_train=False, is_distributed=distributed
        )

        for output_folder, dataset_name, data_loader_val in zip(
            output_folders, dataset_names, data_loaders_val
        ):
            if not args.skip_inference:
                inference(
                    model,
                    data_loader_val,
                    dataset_name=dataset_name,
                    iou_types=iou_types,
                    box_only=cfg_.MODEL.RPN_ONLY
                    and (
                        cfg_.MODEL.RPN_ARCHITECTURE == "RPN"
                        or cfg_.DATASETS.CLASS_AGNOSTIC
                    ),
                    device=cfg_.MODEL.DEVICE,
                    expected_results=cfg_.TEST.EXPECTED_RESULTS,
                    expected_results_sigma_tol=cfg_.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                    output_folder=output_folder,
                    cfg=cfg_,
                )
                synchronize()

            # Only rank 0 runs the exclusion eval — pycocotools is single-process
            # and the prediction JSON has already been gathered.
            if is_main_process() and output_folder is not None:
                run_exclude_eval_for_dataset(
                    dataset=data_loader_val.dataset,
                    output_folder=output_folder,
                    exclude_inputs=exclude_inputs,
                    iou_types=iou_types,
                    logger=logger,
                )

            synchronize()


if __name__ == "__main__":
    main()
