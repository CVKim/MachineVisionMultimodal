"""Produce productivity charts from a saved activity summary.

Reads ``outputs/activity/<stem>/<stem>__summary.json`` and writes:
    <stem>__productivity_buckets.png  — stacked-area of state mix over time
    <stem>__productivity_per_track.png — bar chart of (working+lifting)/total
                                          per track
    <stem>__productivity_summary.csv   — per-bucket CSV usable in Excel

Run:
    python scripts/plot_productivity.py \\
        --summary outputs/activity/store_aisle/store_aisle__summary.json \\
        --bucket-seconds 5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mvmm.tracking.pose_activity import STATE_COLORS_BGR, STATE_ORDER


def _state_color_mpl(state: str) -> tuple[float, float, float]:
    b, g, r = STATE_COLORS_BGR.get(state, STATE_COLORS_BGR["unknown"])
    return (r / 255.0, g / 255.0, b / 255.0)


def reconstruct_state_log(summary: dict) -> dict[int, list[tuple[float, str]]]:
    """The summary stores time-in-state totals but not the full state log.

    For bucket plotting we instead re-read the per-frame state CSV that
    sits next to the summary.
    """
    pass  # implemented by main below


def load_state_csv(states_csv: Path) -> dict[int, list[tuple[float, str]]]:
    out: dict[int, list[tuple[float, str]]] = {}
    with states_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = int(row["track_id"])
            t = float(row["timestamp_s"])
            s = row["state"]
            out.setdefault(tid, []).append((t, s))
    return out


def bucketize(
    state_logs: dict[int, list[tuple[float, str]]],
    bucket_seconds: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Aggregate seconds per state per bucket across ALL tracks.

    Returns:
        (bucket_centers, {state_name: per_bucket_seconds_array})
    """
    max_t = 0.0
    for log in state_logs.values():
        if log:
            max_t = max(max_t, log[-1][0])
    n = max(int(np.ceil(max_t / bucket_seconds)), 1) if max_t > 0 else 1
    per_state = {s: np.zeros(n, dtype=np.float64) for s in STATE_ORDER}
    for log in state_logs.values():
        for i in range(1, len(log)):
            t_prev, s_prev = log[i - 1]
            t_cur, _ = log[i]
            t = float(t_prev)
            end = float(t_cur)
            while t < end:
                b = min(int(t // bucket_seconds), n - 1)
                b_end = min((b + 1) * bucket_seconds, end)
                per_state[s_prev][b] += b_end - t
                t = b_end
    centers = (np.arange(n) + 0.5) * bucket_seconds
    return centers, per_state


def plot_buckets(
    centers: np.ndarray, per_state: dict[str, np.ndarray], path: Path, bucket_seconds: float
) -> None:
    fig, ax = plt.subplots(figsize=(11, 4), dpi=140)
    bottom = np.zeros_like(centers, dtype=np.float64)
    for s in STATE_ORDER:
        vals = per_state[s]
        ax.bar(
            centers,
            vals,
            width=0.85 * bucket_seconds,
            bottom=bottom,
            color=_state_color_mpl(s),
            label=s,
            edgecolor="none",
        )
        bottom += vals
    ax.set_xlabel("video time (seconds)")
    ax.set_ylabel(f"person-seconds per {bucket_seconds:.0f}s bucket")
    ax.set_title(f"Activity mix over time  ({bucket_seconds:.0f}s buckets)")
    ax.legend(loc="upper right", fontsize=8, ncol=len(STATE_ORDER))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_per_track_productivity(state_logs: dict[int, list[tuple[float, str]]], path: Path) -> None:
    working = {"working", "lifting"}
    rows: list[tuple[int, float, float]] = []
    for tid, log in state_logs.items():
        totals = dict.fromkeys(STATE_ORDER, 0.0)
        for i in range(1, len(log)):
            t_prev, s_prev = log[i - 1]
            t_cur, _ = log[i]
            totals[s_prev] = totals.get(s_prev, 0.0) + max(t_cur - t_prev, 0.0)
        total = sum(totals.values())
        if total <= 0:
            continue
        prod = sum(totals.get(s, 0.0) for s in working) / total
        rows.append((tid, prod, total))
    if not rows:
        print("  [skip] no tracks for per-track chart")
        return
    rows.sort(key=lambda x: -x[1])
    ids = [str(r[0]) for r in rows]
    prods = [r[1] * 100 for r in rows]
    totals = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(rows)), 4), dpi=140)
    bars = ax.bar(ids, prods, color="#2ca02c", edgecolor="black", linewidth=0.5)
    for bar, total in zip(bars, totals, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{total:.1f}s",
            ha="center",
            fontsize=7,
        )
    ax.set_ylabel("productive time  (working + lifting) / total  (%)")
    ax.set_xlabel("track id")
    ax.set_title("Per-track productivity")
    ax.set_ylim(0, max(100, max(prods) + 10))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def write_buckets_csv(
    centers: np.ndarray,
    per_state: dict[str, np.ndarray],
    path: Path,
    bucket_seconds: float,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bucket_start_s", "bucket_end_s", *STATE_ORDER, "productive_pct"])
        for i, c in enumerate(centers):
            start = c - bucket_seconds / 2
            end = c + bucket_seconds / 2
            row = [round(start, 3), round(end, 3)]
            secs_per_state = []
            for s in STATE_ORDER:
                secs_per_state.append(round(float(per_state[s][i]), 3))
            row.extend(secs_per_state)
            total = sum(secs_per_state)
            prod = sum(per_state[s][i] for s in ("working", "lifting")) / total if total > 0 else 0.0
            row.append(round(prod * 100, 2))
            w.writerow(row)
    print(f"  saved {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True, help="Path to <stem>__summary.json")
    p.add_argument(
        "--states", type=Path, default=None, help="Path to <stem>__states.csv (defaults to sibling)."
    )
    p.add_argument(
        "--bucket-seconds", type=float, default=5.0, help="Bucket width: 60 = minute, 3600 = hour."
    )
    p.add_argument("--out", type=Path, default=None, help="Output directory (defaults to summary's dir).")
    args = p.parse_args()

    if not args.summary.exists():
        sys.exit(f"summary not found: {args.summary}")
    # We anchor on the summary path to discover the sibling CSV; the
    # JSON body itself is not needed by this script.
    stem = args.summary.stem.replace("__summary", "")
    states_csv = args.states or args.summary.with_name(f"{stem}__states.csv")
    if not states_csv.exists():
        sys.exit(f"states csv not found: {states_csv}")
    out_dir = args.out or args.summary.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[productivity] reading {states_csv}")
    state_logs = load_state_csv(states_csv)
    centers, per_state = bucketize(state_logs, args.bucket_seconds)

    plot_buckets(
        centers,
        per_state,
        out_dir / f"{stem}__productivity_buckets.png",
        args.bucket_seconds,
    )
    plot_per_track_productivity(
        state_logs,
        out_dir / f"{stem}__productivity_per_track.png",
    )
    write_buckets_csv(
        centers,
        per_state,
        out_dir / f"{stem}__productivity_summary.csv",
        args.bucket_seconds,
    )
    print(
        f"\n[productivity] done - total {len(centers)} bucket(s) of "
        f"{args.bucket_seconds:.1f}s spanning {centers[-1] + args.bucket_seconds / 2:.1f}s"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
