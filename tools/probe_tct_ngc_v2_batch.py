#!/usr/bin/env python

import argparse
import gc
import os

import torch
from torch.cuda.amp import GradScaler, autocast

# Set up custom environment before nearly anything else is imported.
from maskrcnn_benchmark.utils.env import setup_environment  # noqa: F401
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer


def build_cfg(args, per_gpu_batch):
    cfg_local = cfg.clone()
    cfg_local.merge_from_file(args.config_file)
    cfg_local.merge_from_file(args.task_config)
    extra_opts = [
        "MODEL.WEIGHT", args.weight,
        "MODEL.DEVICE", "cuda",
        "SOLVER.IMS_PER_BATCH", str(per_gpu_batch),
        "DATASETS.TRAIN_DATASETNAME_SUFFIX", "_grounding",
        "DATASETS.USE_OVERRIDE_CATEGORY", "True",
        "DATASETS.DISABLE_SHUFFLE", "True",
        "MODEL.BACKBONE.USE_CHECKPOINT", "True",
        "MODEL.DYHEAD.USE_CHECKPOINT", "True",
        "MODEL.LANGUAGE_BACKBONE.USE_CHECKPOINT", "True",
        "SOLVER.USE_AMP", "True",
        "DATALOADER.NUM_WORKERS", str(args.num_workers),
    ]
    if args.train_min_sizes:
        extra_opts.extend(["AUGMENT.MULT_MIN_SIZE_TRAIN", args.train_min_sizes])
    if args.train_max_size:
        extra_opts.extend(["INPUT.MAX_SIZE_TRAIN", str(args.train_max_size), "INPUT.MAX_SIZE_TEST", str(args.train_max_size)])
    cfg_local.merge_from_list(extra_opts)
    cfg_local.local_rank = 0
    cfg_local.num_gpus = 1
    cfg_local.freeze()
    return cfg_local


def run_probe(cfg_local):
    model = build_detection_model(cfg_local)
    device = torch.device("cuda")
    model.to(device)
    model.train()

    checkpointer = DetectronCheckpointer(cfg_local, model, save_dir="/tmp/glip_probe")
    checkpointer.load(cfg_local.MODEL.WEIGHT, force=True)

    data_loader = make_data_loader(cfg_local, is_train=True, is_distributed=False, start_iter=0)
    images, targets, _, positive_map, _, greenlight_map = next(iter(data_loader))

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-6)
    scaler = GradScaler(enabled=cfg_local.SOLVER.USE_AMP)

    images = images.to(device)
    targets = [t.to(device) for t in targets]
    captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)

    with autocast(enabled=cfg_local.SOLVER.USE_AMP):
        loss_dict = model(images, targets, captions, positive_map, greenlight_map=greenlight_map)
        losses = sum(loss for loss in loss_dict.values())

    scaler.scale(losses).backward()
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    del images, targets, positive_map, greenlight_map, captions, loss_dict, losses, model, optimizer, scaler, data_loader
    gc.collect()
    torch.cuda.empty_cache()
    return peak_mb


def main():
    parser = argparse.ArgumentParser(description="Probe max per-GPU batch size for TCT_NGC V2 base training.")
    parser.add_argument("--config-file", default="configs/pretrain/glip_Swin_T_O365.yaml")
    parser.add_argument("--task-config", default="configs/tct_ngc/tct_ngc_v2_base.yaml")
    parser.add_argument("--weight", default="MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth")
    parser.add_argument("--candidates", default="1,2,3")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-min-sizes", default="(800,)")
    parser.add_argument("--train-max-size", type=int, default=1333)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for batch probing.")

    print("probing on", torch.cuda.get_device_name(0))
    print("weight", args.weight)

    for item in args.candidates.split(","):
        per_gpu_batch = int(item)
        try:
            cfg_local = build_cfg(args, per_gpu_batch)
            peak_mb = run_probe(cfg_local)
            print(f"per_gpu_batch={per_gpu_batch} ok peak_mem_mb={peak_mb:.2f} suggested_global_batch={per_gpu_batch * 8}")
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"per_gpu_batch={per_gpu_batch} oom")
                torch.cuda.empty_cache()
            else:
                raise


if __name__ == "__main__":
    main()
