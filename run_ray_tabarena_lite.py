"""
Distributed TabArena Lite Benchmark runner using Ray.
Automatically distributes dataset and fold evaluation jobs across all available GPUs in the Ray cluster.
"""
from __future__ import annotations

import os
import sys
import argparse
import time
import dataclasses
from pathlib import Path
from typing import Any

# Polyfill flax.nnx.dataclass for Ray workers
try:
    import flax.nnx as _flax_nnx
    if not hasattr(_flax_nnx, "dataclass"):
        _flax_nnx.dataclass = dataclasses.dataclass
except Exception:
    pass

# Add local package source directory to sys.path and PYTHONPATH
pkg_src = str((Path(__file__).parent / "packages" / "tabarena" / "src").resolve())
if pkg_src not in sys.path:
    sys.path.insert(0, pkg_src)
os.environ["PYTHONPATH"] = f"{pkg_src}:{os.environ.get('PYTHONPATH', '')}"

import ray
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext
from tabarena.models.zstabfm.info import zstabfm_info
from tabarena.models.zsisab.info import zsisab_info


@ray.remote(num_gpus=1)
def run_job_on_ray_worker(job: Any, expname: str, debug_mode: bool = True) -> list[dict[str, Any]]:
    """Runs a single TabArena Job on an assigned Ray GPU worker."""
    import sys
    import os
    import dataclasses
    from pathlib import Path
    
    # Polyfill flax.nnx inside the remote Ray worker process
    try:
        import flax.nnx as _flax_nnx
        if not hasattr(_flax_nnx, "dataclass"):
            _flax_nnx.dataclass = dataclasses.dataclass
    except Exception:
        pass
    
    # Ensure worker has pythonpath set
    pkg_path = "/kaggle/working/tabarena/packages/tabarena/src"
    if os.path.exists(pkg_path) and pkg_path not in sys.path:
        sys.path.insert(0, pkg_path)
        
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    worker_gpus = ray.get_gpu_ids()
    task_name = getattr(job, "task_id", "job")
    print(f"[*] [Ray Worker GPU {worker_gpus}] Starting {task_name}...", flush=True)

    start_time = time.time()
    try:
        from tabarena.contexts import TabArenaContext
        context = TabArenaContext()
        results = context.run_job(job, expname=expname, register=False, debug_mode=debug_mode)
        elapsed = time.time() - start_time
        print(f"[+] [Ray Worker GPU {worker_gpus}] Completed {task_name} in {elapsed:.1f}s", flush=True)
        return results
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[!] [Ray Worker GPU {worker_gpus}] Failed {task_name} after {elapsed:.1f}s: {e}", flush=True)
        raise e


def main():
    parser = argparse.ArgumentParser(description="Run Distributed TabArena benchmark via Ray.")
    parser.add_argument("--models", type=str, default="zstabfm", choices=["zstabfm", "zsisab"], help="Model to benchmark.")
    parser.add_argument("--subset", type=str, default="tiny", help="Dataset subset (e.g., 'tiny', 'all').")
    parser.add_argument("--num-gpus-per-task", type=float, default=1.0, help="Number of GPUs per Ray task.")
    args = parser.parse_args()

    print("=" * 75)
    print(f"RUNNING DISTRIBUTED TABARENA BENCHMARK VIA RAY FOR {args.models.upper()}")
    print("=" * 75)

    runtime_env = {
        "env_vars": {
            "PYTHONPATH": pkg_src,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    }

    try:
        ray.init(address="auto", runtime_env=runtime_env, ignore_reinit_error=True)
        print("[+] Attached to live Ray cluster with runtime_env configured.")
    except Exception:
        print("[*] No existing Ray cluster found. Initializing local Ray instance...")
        ray.init(runtime_env=runtime_env, ignore_reinit_error=True)

    cluster_resources = ray.cluster_resources()
    total_cpus = cluster_resources.get("CPU", 0)
    total_gpus = cluster_resources.get("GPU", 0)
    print(f"[+] Active Ray Resources: {int(total_cpus)} CPUs | {int(total_gpus)} GPUs")

    output_dir = Path(__file__).parent / f"tabarena_{args.subset}_results"
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.models == "zstabfm":
        target_info = zstabfm_info
    else:
        target_info = zsisab_info

    experiments = TabArenaV0pt1ExperimentBundle(
        models=[
            (target_info.search_space, 0),
        ],
    ).build_experiments()

    if "," in args.subset:
        subset_filter = [s.strip() for s in args.subset.split(",")]
    elif args.subset in ("tiny", "tiny_lite", "lite_tiny"):
        subset_filter = ["tiny", "lite"]
    else:
        subset_filter = args.subset

    print(f"[+] Active task filters: {subset_filter}")

    context = TabArenaContext()
    print("[*] Building benchmark execution plan...")
    jobs = context.build_jobs(
        experiments,
        subset=subset_filter,
    )
    print(f"[+] Total jobs to execute across cluster: {len(jobs)}")

    if not jobs:
        print("No jobs found for the specified configuration.")
        return

    print(f"\n[*] Dispatching {len(jobs)} jobs across {int(total_gpus)} GPUs in parallel...")
    start_total = time.time()

    remote_task_fn = run_job_on_ray_worker.options(num_gpus=args.num_gpus_per_task)
    futures = [
        remote_task_fn.remote(job, str(cache_dir), debug_mode=True)
        for job in jobs
    ]

    job_results_nested = ray.get(futures)
    all_results: list[dict[str, Any]] = []
    for res_list in job_results_nested:
        if res_list:
            all_results.extend(res_list)

    total_time = time.time() - start_total
    print(f"\n[+] All {len(jobs)} jobs finished successfully in {total_time:.1f}s ({total_time/60:.2f} mins)!")

    print(f"[*] Registering {len(all_results)} split results into TabArenaContext...")
    context.register(all_results, new_result_prefix="[New] ")

    print(f"\n[*] Generating TabArena {args.subset.upper()} official leaderboard...")
    leaderboard = context.compare(output_dir=output_dir / "eval")
    website_lb = context.leaderboard_to_website_format(leaderboard=leaderboard)

    print("\n" + "=" * 75)
    print("OFFICIAL TABARENA LEADERBOARD OUTPUT:")
    print("=" * 75)
    print(website_lb.to_markdown(index=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "tabarena_leaderboard.md", "w", encoding="utf-8") as f:
        f.write(website_lb.to_markdown(index=False))
    website_lb.to_csv(output_dir / "tabarena_leaderboard.csv", index=False)
    print(f"\n[+] Saved final leaderboard outputs to {output_dir}")


if __name__ == "__main__":
    main()
