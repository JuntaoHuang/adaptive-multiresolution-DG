#!/usr/bin/env python3
"""Plot profile1D_*.txt in one figure, sorted by time in file name."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


TIME_RE = re.compile(r"^profile1D_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\.txt$")


def parse_time(file_path: Path) -> float | None:
    match = TIME_RE.match(file_path.name)
    if match is None:
        return None
    return float(match.group(1))


def save_multi_formats(output_name: str) -> None:
    output_path = Path(output_name)
    stem_path = output_path.with_suffix("")

    targets = [
        output_path,
        stem_path.with_suffix(".eps"),
        stem_path.with_suffix(".jpg"),
        stem_path.with_suffix(".pdf"),
    ]
    saved: list[str] = []
    for target in targets:
        if target.suffix.lower() == ".jpg":
            plt.savefig(target, dpi=300)
        else:
            plt.savefig(target, dpi=200)
        saved.append(str(target))

    print("Saved figures: " + ", ".join(saved))


def main() -> None:
    examples = """Examples:
  python3 scripts/plot_profile1d.py --show
  python3 scripts/plot_profile1d.py --time-interval 0.02 --tmin 0.1 --tmax 0.2 --show
  python3 scripts/plot_profile1d.py --final --input-file profile1D_final.txt --show
"""
    parser = argparse.ArgumentParser(
        description=(
            "Plot Burgers 1D profiles.\n"
            "Default mode plots all profile1D_*.txt in one figure by time order."
        ),
        epilog=examples,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="Plot a single final profile file (same style as plot_profile1d.py)",
    )
    parser.add_argument(
        "--input-file",
        default="profile1D_final.txt",
        help="Input file for --final mode (default: profile1D_final.txt)",
    )
    parser.add_argument(
        "--pattern",
        default="profile1D_*.txt",
        help="Glob pattern for profile files (default: profile1D_*.txt)",
    )
    parser.add_argument(
        "--output",
        default="profile1D.png",
        help="Output image file (default: profile1D.png)",
    )
    parser.add_argument(
        "--time-interval",
        type=float,
        default=0.0,
        help="Minimum time gap between plotted snapshots (default: 0, plot all)",
    )
    parser.add_argument(
        "--tmin",
        type=float,
        default=None,
        help="Minimum time to include (default: no lower bound)",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=None,
        help="Maximum time to include (default: no upper bound)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figure window after saving",
    )
    args = parser.parse_args()

    if args.final:
        input_path = Path(args.input_file)
        data = np.loadtxt(input_path)
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError("Expected at least 3 columns: x, y-num, y-exa")

        x = data[:, 0]
        y_num = data[:, 1]
        y_exa = data[:, 2]

        plt.figure(figsize=(8, 5))
        plt.plot(x, y_num, "o", label="y-num", markersize=3)
        plt.plot(x, y_exa, "-", label="y-exa", linewidth=2)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Profile from {input_path.name}")
        plt.legend()
        plt.tight_layout()
        save_multi_formats(args.output)
        if args.show:
            plt.show()
        return

    all_files = sorted(Path(".").glob(args.pattern))
    timed_files: list[tuple[float, Path]] = []
    for f in all_files:
        t = parse_time(f)
        if t is not None:
            timed_files.append((t, f))
    timed_files.sort(key=lambda x: x[0])

    if not timed_files:
        raise FileNotFoundError(
            f"No time-stamped files found for pattern: {args.pattern}"
        )

    # Optional time-range filtering.
    if args.tmin is not None:
        timed_files = [item for item in timed_files if item[0] >= args.tmin]
    if args.tmax is not None:
        timed_files = [item for item in timed_files if item[0] <= args.tmax]

    # Keep snapshots with at least `time_interval` spacing.
    if args.time_interval > 0:
        filtered: list[tuple[float, Path]] = []
        last_t: float | None = None
        for t, f in timed_files:
            if last_t is None or t - last_t >= args.time_interval:
                filtered.append((t, f))
                last_t = t
        timed_files = filtered

    if not timed_files:
        raise ValueError("No files left after applying time filters.")

    cmap = plt.cm.viridis
    n = len(timed_files)
    plt.figure(figsize=(10, 6))

    for i, (t, file_path) in enumerate(timed_files):
        data = np.loadtxt(file_path)
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(
                f"{file_path} must contain at least 3 columns: x, y-num, y-exa"
            )

        x = data[:, 0]
        y_num = data[:, 1]
        y_exa = data[:, 2]
        color = cmap(i / max(n - 1, 1))
        label_num = f"num t={t:g}"
        label_exa = f"exa t={t:g}"

        plt.plot(x, y_num, "o", color=color, markersize=2, alpha=0.9, label=label_num)
        plt.plot(x, y_exa, "-", color=color, linewidth=1.2, alpha=0.8, label=label_exa)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("profile1D over time")
    plt.grid(True, alpha=0.2)
    if n <= 10:
        plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    save_multi_formats(args.output)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
