"""
Official TabArena evaluation script.
Uses TabArenaV0pt1ExperimentBundle and TabArenaContext.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

# Add source directory
sys.path.insert(0, str(Path(__file__).parent / "packages" / "tabarena" / "src"))

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext
from tabarena.models.zstabfm.info import zstabfm_info
from tabarena.models.zsisab.info import zsisab_info


def main():
    parser = argparse.ArgumentParser(description="Run TabArena benchmark.")
    parser.add_argument("--models", type=str, default="zstabfm", choices=["zstabfm", "zsisab"], help="Model to benchmark.")
    parser.add_argument("--subset", type=str, default="tiny", help="Dataset subset (e.g., 'tiny', 'all').")
    args = parser.parse_args()

    print("=" * 70)
    print(f"RUNNING OFFICIAL TABARENA BENCHMARK FOR {args.models.upper()} (SUBSET: {args.subset.upper()})")
    print("=" * 70)

    output_dir = Path(__file__).parent / f"tabarena_{args.subset}_results"

    if args.models == "zstabfm":
        target_info = zstabfm_info
    else:
        target_info = zsisab_info

    experiments = TabArenaV0pt1ExperimentBundle(
        models=[
            (target_info.search_space, 0),
        ],
    ).build_experiments()

    # Handle composite subset filters (e.g. 'tiny' + 'lite' for the 18-dataset fast benchmark)
    if "," in args.subset:
        subset_filter = [s.strip() for s in args.subset.split(",")]
    elif args.subset in ("tiny", "tiny_lite", "lite_tiny"):
        subset_filter = ["tiny", "lite"]
    else:
        subset_filter = args.subset

    print(f"Active task filters: {subset_filter}")

    context = TabArenaContext()
    context.build_and_run_jobs(
        experiments,
        expname=str(output_dir / "cache"),
        subset=subset_filter,
        new_result_prefix="[New] ",
        debug_mode=True,
    )

    print(f"\nGenerating TabArena {args.subset.upper()} official leaderboard...")
    leaderboard = context.compare(output_dir=output_dir / "eval")
    website_lb = context.leaderboard_to_website_format(leaderboard=leaderboard)

    print("\n" + "=" * 70)
    print("OFFICIAL TABARENA LEADERBOARD OUTPUT:")
    print("=" * 70)
    print(website_lb.to_markdown(index=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "tabarena_leaderboard.md", "w") as f:
        f.write(website_lb.to_markdown(index=False))
    website_lb.to_csv(output_dir / "tabarena_leaderboard.csv", index=False)
    print(f"\nSaved leaderboard outputs to {output_dir}")


if __name__ == "__main__":
    main()
