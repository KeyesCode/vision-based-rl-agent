"""Baseline (grayscale 84, no aux) vs representation upgrade (RGB 128 + aux loss).

Renders a 4-panel bar chart: stochastic return, deterministic return, success
rate, deterministic-INTERACT share. The deterministic panels are the key signal
— the hypothesis is that richer input + supervised adjacency breaks the argmax
collapse that neither the feedforward nor the recurrent variant could fix.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-stochastic", type=Path, required=True)
    parser.add_argument("--baseline-deterministic", type=Path, required=True)
    parser.add_argument("--repr-stochastic", type=Path, required=True)
    parser.add_argument("--repr-deterministic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    b_s = _load(args.baseline_stochastic)
    b_d = _load(args.baseline_deterministic)
    r_s = _load(args.repr_stochastic)
    r_d = _load(args.repr_deterministic)

    panels = [
        ("episode_return", "episode return (stochastic)", b_s, r_s),
        ("episode_return", "episode return (deterministic)", b_d, r_d),
        ("success_rate", "success rate (%, stochastic)", b_s, r_s),
        ("interact_share", "INTERACT share (%, deterministic)", b_d, r_d),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    width = 0.55
    for ax, (key, title, a, b) in zip(axes, panels, strict=True):
        vals = [_mean(a, key), _mean(b, key)]
        errs = [_std(a, key), _std(b, key)]
        ax.bar(
            [0, 1],
            vals,
            width,
            yerr=errs,
            capsize=4,
            color=["tab:red", "tab:blue"],
            alpha=0.85,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            [
                "baseline\n(84×84 grayscale)",
                "upgrade\n(128×128 RGB + aux)",
            ],
            fontsize=9,
        )
        ax.set_title(title, fontsize=10)
        ymax = max(abs(v) for v in vals) or 1.0
        for i, v in enumerate(vals):
            ax.text(i, v + (0.04 * ymax if v >= 0 else -0.08 * ymax), f"{v:+.2f}",
                    ha="center", fontsize=9)

    fig.suptitle(
        "Representation upgrade: RGB 128×128 + adjacency auxiliary loss "
        "vs grayscale 84×84 baseline",
        fontsize=11,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
