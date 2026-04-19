"""Full system-evolution chart — baseline → DR → LSTM → representation → + masking.

Five deterministic-mode bars per metric so the audience can read the entire
research trajectory of the project in one image. The story the chart tells:
four training-time interventions all landed at the same deterministic return
(−3.36) and the same argmax collapse (100% INTERACT), and only the
inference-time mask moved the needle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def _mean(d: dict, key: str) -> float:
    if key == "success_rate":
        return 100.0 * float(d["success_rate"])
    if key == "interact_share":
        return 100.0 * float(d["action_distribution"].get("INTERACT", 0.0))
    return float(d[key]["mean"])


def _std(d: dict, key: str) -> float:
    if key in ("success_rate", "interact_share"):
        return 0.0
    return float(d[key]["std"])


STAGES = [
    ("baseline", "tab:red"),
    ("+ DR", "tab:orange"),
    ("+ LSTM", "tab:green"),
    ("+ repr.", "tab:purple"),
    ("+ masking", "tab:blue"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-det", type=Path, required=True)
    parser.add_argument("--dr-det", type=Path, required=True)
    parser.add_argument("--lstm-det", type=Path, required=True)
    parser.add_argument("--repr-det", type=Path, required=True)
    parser.add_argument("--masked-det", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = [
        _load(args.baseline_det),
        _load(args.dr_det),
        _load(args.lstm_det),
        _load(args.repr_det),
        _load(args.masked_det),
    ]
    labels = [s[0] for s in STAGES]
    colors = [s[1] for s in STAGES]

    panels = [
        ("episode_return", "episode return (deterministic)"),
        ("success_rate", "success rate (%, deterministic)"),
        ("interact_share", "INTERACT share (%, deterministic)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (key, title) in zip(axes, panels, strict=True):
        vals = [_mean(d, key) for d in data]
        errs = [_std(d, key) for d in data]
        x = list(range(len(data)))
        ax.bar(x, vals, 0.6, yerr=errs, capsize=3, color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
        ax.set_title(title, fontsize=10)
        ymax = max(abs(v) for v in vals) or 1.0
        for i, v in enumerate(vals):
            pad = 0.04 * ymax if v >= 0 else -0.08 * ymax
            ax.text(i, v + pad, f"{v:+.2f}", ha="center", fontsize=9)

    fig.suptitle(
        "System evolution — deterministic behavior across five milestones",
        fontsize=11,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
