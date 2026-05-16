"""Evaluate GLIP with an organ / tissue prior applied per image.

Medical patches in TCT_NGC and similar datasets come with a known organ
(encoded in the filename, e.g. ``Thyroid_gland__xxx.jpg``). At test time we
should only score classes belonging to that organ — predictions for classes
of a different organ are dropped.

This tool runs the standard GLIP inference (like ``tools/test_grounding_net.py``)
and then re-evaluates the dumped ``bbox.json`` with the prior applied, producing:

  - ``bbox.organ_prior.json`` — per-image-organ-filtered predictions
  - ``bbox.organ_prior.summary.json`` — all-class flat + per-organ AP + macro / instance-weighted

The implementation mirrors WeDetect's ``OrganRestrictedCocoMetric`` and
YOLOE's ``eval_domain_prior_infer_tct_ngc.py`` but operates entirely post-hoc
at the COCOeval level.

Examples
--------

Full pipeline against an ODinW-style COCO task with the WeDetect taxonomy::

    python tools/test_grounding_net_organ_prior.py \
        --config-file {config} --weight {ckpt} --task_config {odinw_yaml} \
        --taxonomy /path/to/tct_ngc_taxonomy.json \
        TEST.IMS_PER_BATCH 1 TEST.EVAL_TASK detection \
        DATASETS.USE_OVERRIDE_CATEGORY True DATASETS.USE_CAPTION_PROMPT True

Compare with-prior vs without (writes both summaries)::

    ... --compare-baseline

Re-eval an existing dump (no GPU needed)::

    python tools/test_grounding_net_organ_prior.py \
        --config-file {config} --weight {ckpt} --task_config {odinw_yaml} \
        --taxonomy /path/to/mask.pt --skip-inference
"""

from maskrcnn_benchmark.utils.env import setup_environment  # noqa: F401  isort:skip

import argparse
import datetime
import json
import os

import torch
import torch.distributed as dist

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.data.datasets.evaluation.coco.organ_restricted_coco_eval import (
    DEFAULT_PREFIX_TO_ORGAN,
    evaluate_with_organ_prior,
)
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.comm import get_rank, is_main_process, synchronize
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir


def init_distributed_mode(args):
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
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(0, 7200),
    )
    dist.barrier()


def _parse_extra_prefix_map(s):
    if s is None:
        return None
    extra = json.loads(s)
    if not isinstance(extra, dict):
        raise ValueError("--path-prefix-map must be a JSON object {prefix: organ_name}")
    return extra


def run_one_dataset(dataset, output_folder, args, logger):
    """Re-evaluate ``{output_folder}/bbox.json`` with the organ prior."""
    if getattr(dataset, "coco", None) is None:
        logger.warning("organ-prior: dataset has no coco api — skipping")
        return

    bbox_json = os.path.join(output_folder, "bbox.json")
    if not os.path.isfile(bbox_json):
        logger.warning("organ-prior: expected %s not found — skipping", bbox_json)
        return

    with open(bbox_json, "r") as f:
        preds = json.load(f)

    extra_prefix_to_organ = _parse_extra_prefix_map(args.path_prefix_map)

    # Primary run: prior on
    evaluate_with_organ_prior(
        coco_gt=dataset.coco,
        raw_predictions=preds,
        taxonomy_path=args.taxonomy,
        output_folder=output_folder,
        extra_prefix_to_organ=extra_prefix_to_organ,
        apply_prior=True,
        logger=logger,
    )

    # Baseline: prior off — distinct filenames (bbox.organ_prior_off.*) so it
    # safely lands in the same output_folder.
    if args.compare_baseline:
        evaluate_with_organ_prior(
            coco_gt=dataset.coco,
            raw_predictions=preds,
            taxonomy_path=args.taxonomy,
            output_folder=output_folder,
            extra_prefix_to_organ=extra_prefix_to_organ,
            apply_prior=False,
            logger=logger,
        )


def main():
    parser = argparse.ArgumentParser(
        description="GLIP grounding-net eval with per-image organ prior"
    )
    parser.add_argument("--config-file", required=True, metavar="FILE")
    parser.add_argument("--weight", default=None, metavar="FILE")
    parser.add_argument("--task_config", default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world-size", default=1, type=int)
    parser.add_argument("--dist-url", default="env://")

    parser.add_argument(
        "--taxonomy",
        required=True,
        help="Path to taxonomy .json (e.g. WeDetect's tct_ngc_taxonomy.json) "
        "OR organ-mask .pt (built by WeDetect's tools/build_class_organ_mask.py).",
    )
    parser.add_argument(
        "--path-prefix-map",
        default=None,
        help='JSON object mapping additional filename-prefix → organ_name, '
        "merged on top of DEFAULT_PREFIX_TO_ORGAN. "
        f"Defaults already cover: {sorted(DEFAULT_PREFIX_TO_ORGAN)}",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also run a prior-OFF pass. Writes bbox.organ_prior_off.* "
        "alongside the bbox.organ_prior.* files. Equivalent to YOLOE's "
        "prior_OFF/prior_ON comparison.",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Do not run the model — re-evaluate existing bbox.json files.",
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

    model = None
    if not args.skip_inference:
        model = build_detection_model(cfg)
        model.to(cfg.MODEL.DEVICE)
        checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
        if args.weight:
            checkpointer.load(args.weight, force=True)
        else:
            checkpointer.load(cfg.MODEL.WEIGHT)

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

            if is_main_process() and output_folder is not None:
                run_one_dataset(data_loader_val.dataset, output_folder, args, logger)
            synchronize()


if __name__ == "__main__":
    main()
