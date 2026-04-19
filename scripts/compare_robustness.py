"""Render the 2×2 robustness chart used in the README.

Takes four evaluation JSONs — baseline policy on baseline env, baseline policy on
randomized env, DR policy on baseline env, DR policy on randomized env — and
produces a grouped bar chart showing how each policy holds up across the two
visual regimes. The distance between the two bars of the *same* policy is the
robustness proxy: a smaller drop means the policy relies less on the specific
training-time visuals.

Usage:
    python scripts/compare_robustness.py \
        --baseline-on-baseline runs/ppo_woodcutting_v2/eval_trained.json \
        --baseline-on-dr       runs/robustness/v2_on_dr.json \
        --dr-on-baseline       runs/robustness/dr_on_baseline.json \
        --dr-on-dr             runs/robustness/dr_on_dr.json \
        --output docs/results/robustness.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _metric(d: dict, key: str) -> tuple[float, float]:
    if key == "success_rate":
        return 100.0 * float(d["success_rate"]), 0.0
    block = d[key]
    return float(block["mean"]), float(block["std"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-on-baseline", type=Path, required=True)
    parser.add_argument("--baseline-on-dr", type=Path, required=True)
    parser.add_argument("--dr-on-baseline", type=Path, required=True)
    parser.add_argument("--dr-on-dr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    b_b = _load(args.baseline_on_baseline)
    b_d = _load(args.baseline_on_dr)
    d_b = _load(args.dr_on_baseline)
    d_d = _load(args.dr_on_dr)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    metrics = [
        ("episode_return", "episode return", None),
        ("trees_chopped", "trees chopped / episode", None),
        ("success_rate", "success rate (%)", None),
    ]
    labels = ("baseline env", "randomized env")
    width = 0.38
    x = np.arange(len(labels))

    for ax, (key, title, _) in zip(axes, metrics, strict=True):
        b_vals = [_metric(b_b, key)[0], _metric(b_d, key)[0]]
        d_vals = [_metric(d_b, key)[0], _metric(d_d, key)[0]]
        if key != "success_rate":
            b_err = [_metric(b_b, key)[1], _metric(b_d, key)[1]]
            d_err = [_metric(d_b, key)[1], _metric(d_d, key)[1]]
        else:
            b_err = d_err = [0.0, 0.0]

        ax.bar(x - width / 2, b_vals, width, yerr=b_err, capsize=3,
               label="baseline policy (no DR)", color="tab:red", alpha=0.8)
        ax.bar(x + width / 2, d_vals, width, yerr=d_err, capsize=3,
               label="domain-randomized policy", color="tab:blue", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)

    axes[-1].legend(loc="lower right", fontsize=9)
    fig.suptitle(
        "Robustness under visual domain randomization "
        "— smaller drop between bars = more transferable policy",
        fontsize=11,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
