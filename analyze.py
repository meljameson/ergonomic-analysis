#!/usr/bin/env python3
"""Summarize and compare wrist-angle trials recorded by wrist_angle.py.

Example:
  python analyze.py direct_grip.csv handle_v1.csv --threshold 15
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="+", help="CSV files from wrist_angle.py")
    p.add_argument("--labels", nargs="+", help="legend labels (default: file names)")
    p.add_argument("--threshold", type=float, default=15.0, help="report time spent beyond +/- this angle (deg)")
    p.add_argument("--smooth", type=int, default=5, help="median filter window in frames (1 = off)")
    p.add_argument("--plot", help="save the plot to this file instead of showing it")
    return p.parse_args()


def load_trials(path):
    """Return {trial: (time_s, angle_deg)} with untracked frames as NaN."""
    trials = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            t, a = trials.setdefault(int(row["trial"]), ([], []))
            t.append(float(row["time_s"]))
            a.append(float(row["angle_deg"]) if row["angle_deg"] else np.nan)
    return {k: (np.array(t), np.array(a)) for k, (t, a) in trials.items()}


def median_filter(values, window):
    if window <= 1:
        return values
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    return np.array([np.nanmedian(padded[i:i + window]) if not np.isnan(values[i]) else np.nan
                     for i in range(len(values))])


def summarize(time_s, angle, threshold):
    tracked = ~np.isnan(angle)
    a = angle[tracked]
    if a.size == 0:
        return None
    dt = np.diff(time_s, append=time_s[-1])[tracked]
    return {
        "duration_s": time_s[-1] - time_s[0],
        "tracked_pct": 100 * tracked.mean(),
        "peak_ext_deg": a.max(),
        "peak_flex_deg": a.min(),
        "mean_abs_deg": np.abs(a).mean(),
        "time_beyond_s": dt[np.abs(a) > threshold].sum(),
    }


def main():
    args = parse_args()
    labels = args.labels or [Path(p).stem for p in args.csv]
    if len(labels) != len(args.csv):
        raise SystemExit("--labels must have one entry per CSV file")

    header = f"{'trial':<28}{'dur s':>7}{'tracked%':>10}{'peak +':>9}{'peak -':>9}{'mean|a|':>9}{f'>{args.threshold:g}deg s':>11}"
    print(header)
    print("-" * len(header))

    fig, ax = plt.subplots(figsize=(10, 5))
    for path, label in zip(args.csv, labels):
        for trial, (t, a) in load_trials(path).items():
            a = median_filter(a, args.smooth)
            name = f"{label} #{trial}"
            s = summarize(t, a, args.threshold)
            if s is None:
                print(f"{name:<28}  (no tracked frames)")
                continue
            print(f"{name:<28}{s['duration_s']:>7.1f}{s['tracked_pct']:>10.0f}{s['peak_ext_deg']:>9.1f}"
                  f"{s['peak_flex_deg']:>9.1f}{s['mean_abs_deg']:>9.1f}{s['time_beyond_s']:>11.1f}")
            ax.plot(t, a, label=name)

    for y in (args.threshold, -args.threshold):
        ax.axhline(y, color="grey", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("wrist angle relative to neutral (deg)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if args.plot:
        fig.savefig(args.plot, dpi=150)
        print(f"Saved plot to {args.plot}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
