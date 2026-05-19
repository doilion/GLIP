#!/usr/bin/env python

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description="Probe efficient 8-GPU eval config for TCT_NGC V2.")
    parser.add_argument(
        "--config-file",
        default="OUTPUT/tct_ngc_v2_base_dev30_glip_tiny_goldg_ccsbu/ft_task_1/config.yml",
    )
    parser.add_argument(
        "--task-config",
        required=True,
    )
    parser.add_argument(
        "--weight",
        default="OUTPUT/tct_ngc_v2_base_dev30_glip_tiny_goldg_ccsbu/ft_task_1/model_final.pth",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-candidates", default="32,24,16,8")
    parser.add_argument("--worker-candidates", default="8,4")
    parser.add_argument("--subset-batches", type=int, default=20)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    return parser


def run_probe(args, batch_size, num_workers, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    command = ["bash", "tools/eval_tct_ngc_v2_8gpu.sh"]
    env_full = dict(os.environ)
    env_full.update({
        "CONFIG_FILE": str(Path(args.config_file).resolve()),
        "TASK_CONFIG": str(Path(args.task_config).resolve()),
        "WEIGHT_PATH": str(Path(args.weight).resolve()),
        "OUTPUT_DIR": str(output_dir.resolve()),
        "TEST_BATCH": str(batch_size),
        "NUM_WORKERS": str(num_workers),
        "USE_AMP": "True",
        "SUBSET": str(args.subset_batches),
        "NPROC_PER_NODE": str(args.nproc_per_node),
    })
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=Path.cwd(), env=env_full)
    elapsed = time.perf_counter() - start
    success = proc.returncode == 0
    approx_images = args.subset_batches * batch_size
    throughput = approx_images / elapsed if success and elapsed > 0 else 0.0
    return {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "success": success,
        "elapsed_seconds": elapsed,
        "approx_images": approx_images,
        "approx_images_per_second": throughput,
        "output_dir": str(output_dir),
    }


def main():
    args = build_parser().parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    batch_candidates = [int(item) for item in args.batch_candidates.split(",")]
    worker_candidates = [int(item) for item in args.worker_candidates.split(",")]

    results = []
    chosen_batch = None
    default_workers = worker_candidates[0]
    successful_batch_probe = None

    for batch_size in batch_candidates:
        probe_dir = output_root / "batch_search" / f"batch_{batch_size}_workers_{default_workers}"
        result = run_probe(args, batch_size, default_workers, probe_dir)
        result["phase"] = "batch_search"
        results.append(result)
        if result["success"]:
            chosen_batch = batch_size
            successful_batch_probe = result
            break

    if chosen_batch is None:
        raise RuntimeError("No stable evaluation batch size found.")

    worker_results = []
    for num_workers in worker_candidates:
        if num_workers == default_workers and successful_batch_probe is not None:
            result = dict(successful_batch_probe)
            result["phase"] = "worker_search_reused"
        else:
            probe_dir = output_root / "worker_search" / f"batch_{chosen_batch}_workers_{num_workers}"
            result = run_probe(args, chosen_batch, num_workers, probe_dir)
            result["phase"] = "worker_search"
        worker_results.append(result)
        results.append(result)

    stable_worker_results = [item for item in worker_results if item["success"]]
    if not stable_worker_results:
        raise RuntimeError("No stable worker configuration found for chosen batch size.")
    best_worker = max(stable_worker_results, key=lambda item: item["approx_images_per_second"])

    summary = {
        "selected_batch_size": chosen_batch,
        "selected_num_workers": best_worker["num_workers"],
        "subset_batches": args.subset_batches,
        "results": results,
    }
    (output_root / "probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
