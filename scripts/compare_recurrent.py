"""Feedforward vs recurrent PPO comparison chart.

Renders a 4-panel bar chart — stochastic return, deterministic return, success
rate, idle ratio — for the two policies. The deterministic-return panel is the
key portfolio signal: if recurrent memory actually helps the argmax policy make
coherent action sequences, the blue (LSTM) bar should be materially larger.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _mean(d: dict, key: str) -> float:
    if key == "success_rate":
        return 100.0 * float(d["success_rate"])
    if key in ("idle_ratio", "invalid_action_ratio"):
        return 100.0 * float(d[key])
    return float(d[key]["mean"])


def _std(d: dict, key: str) -> float:
    if key in ("success_rate", "idle_ratio", "invalid_action_ratio"):
        return 0.0
    return float(d[key]["std"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ff-stochastic", type=Path, required=True)
    parser.add_argument("--ff-deterministic", type=Path, required=True)
    parser.add_argument("--lstm-stochastic", type=Path, required=True)
    parser.add_argument("--lstm-deterministic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ff_s = _load(args.ff_stochastic)
    ff_d = _load(args.ff_deterministic)
    rn_s = _load(args.lstm_stochastic)
    rn_d = _load(args.lstm_deterministic)

    panels = [
        ("episode_return", "episode return (stochastic)", ff_s, rn_s),
        ("episode_return", "episode return (deterministic)", ff_d, rn_d),
        ("success_rate", "success rate (%, stochastic)", ff_s, rn_s),
        ("idle_ratio", "idle ratio (%, stochastic)", ff_s, rn_s),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    width = 0.55
    for ax, (key, title, ff, rn) in zip(axes, panels, strict=True):
        vals = [_mean(ff, key), _mean(rn, key)]
        errs = [_std(ff, key), _std(rn, key)]
        colors = ["tab:red", "tab:blue"]
        ax.bar([0, 1], vals, width, yerr=errs, capsize=4, color=colors, alpha=0.85)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["feedforward", "recurrent (LSTM)"])
        ax.set_title(title, fontsize=10)
        # Annotate values on top of bars
        ymax = max(vals) if max(vals) != 0 else 1.0
        for i, v in enumerate(vals):
            ax.text(i, v + 0.04 * abs(ymax), f"{v:+.2f}", ha="center", fontsize=9)

    fig.suptitle(
        "Feedforward PPO vs recurrent (LSTM) PPO — same budget, same hyperparams",
        fontsize=11,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
