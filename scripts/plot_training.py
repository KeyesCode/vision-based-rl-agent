"""Turn a training run's TensorBoard logs into README-ready PNG charts.

Usage:
    python scripts/plot_training.py --run-dir runs/ppo_woodcutting \
        [--baseline-json eval_random.json] \
        [--trained-json eval_trained.json] \
        [--progression-json runs/ppo_woodcutting/checkpoint_progression.json] \
        [--output-dir runs/ppo_woodcutting/plots]

Plots produced:
    - reward_over_time.png           charts/episode_return (+ eval overlay, + baseline)
    - episode_length.png             charts/episode_length
    - success_rate.png               charts/success_rate
    - trees_chopped.png              charts/trees_chopped
    - losses.png                     policy / value / entropy
    - explained_variance.png         value-function diagnostic
    - action_distribution.png        bar chart: random vs trained policy (needs JSONs)
    - checkpoint_progression.png     clean deterministic-eval curve over checkpoints
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

plt.rcParams.update(
    {
        "figure.figsize": (8.0, 4.5),
        "figure.dpi": 110,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
    }
)


@dataclass
class Series:
    steps: np.ndarray
    values: np.ndarray

    @classmethod
    def empty(cls) -> "Series":
        return cls(np.array([]), np.array([]))

    def is_empty(self) -> bool:
        return self.steps.size == 0

    def smooth(self, window: int) -> np.ndarray:
        if window <= 1 or self.values.size == 0:
            return self.values
        w = min(window, self.values.size)
        kernel = np.ones(w) / w
        return np.convolve(self.values, kernel, mode="valid")


def load_scalars(run_dir: Path) -> dict[str, Series]:
    """Load all scalar series from a TB event file (picks the largest one)."""
    event_files = list(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TB event files in {run_dir}")
    event_path = max(event_files, key=lambda p: p.stat().st_size)
    ea = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    ea.Reload()
    series: dict[str, Series] = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        steps = np.asarray([e.step for e in events])
        values = np.asarray([e.value for e in events])
        series[tag] = Series(steps, values)
    return series


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_line(
    ax,
    series: Series,
    label: str,
    smooth_window: int = 5,
    raw_alpha: float = 0.25,
    color: str | None = None,
) -> None:
    if series.is_empty():
        return
    ax.plot(series.steps, series.values, alpha=raw_alpha, color=color)
    if smooth_window > 1 and series.values.size > smooth_window:
        smoothed = series.smooth(smooth_window)
        offset = (series.values.size - smoothed.size) // 2
        x = series.steps[offset : offset + smoothed.size]
        ax.plot(x, smoothed, label=label, linewidth=2, color=color)
    else:
        ax.plot(series.steps, series.values, label=label, linewidth=2, color=color)


def plot_reward(
    scalars: dict[str, Series],
    output: Path,
    baseline_return: float | None = None,
) -> None:
    fig, ax = plt.subplots()
    _plot_line(ax, scalars.get("charts/episode_return", Series.empty()), "training (rolling mean)", color="tab:blue")
    eval_series = scalars.get("eval/episode_return", Series.empty())
    if not eval_series.is_empty():
        ax.scatter(eval_series.steps, eval_series.values, color="tab:orange", label="eval (deterministic)", zorder=3, s=32)
    if baseline_return is not None:
        ax.axhline(baseline_return, color="tab:red", linestyle="--", label=f"random baseline ({baseline_return:+.2f})")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode return")
    ax.set_title("Episode return over training")
    ax.legend(loc="best")
    _save(fig, output)


def plot_single(
    scalars: dict[str, Series],
    tag: str,
    title: str,
    ylabel: str,
    output: Path,
    baseline: float | None = None,
    baseline_label: str | None = None,
    color: str = "tab:green",
    y_percent: bool = False,
) -> None:
    series = scalars.get(tag, Series.empty())
    if series.is_empty():
        return
    fig, ax = plt.subplots()
    _plot_line(ax, series, "training", color=color)
    if baseline is not None:
        ax.axhline(
            baseline,
            color="tab:red",
            linestyle="--",
            label=baseline_label or f"random ({baseline:.2f})",
        )
        ax.legend()
    ax.set_xlabel("environment steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if y_percent:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    _save(fig, output)


def plot_losses(scalars: dict[str, Series], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (tag, title) in zip(
        axes,
        [
            ("losses/policy_loss", "policy loss"),
            ("losses/value_loss", "value loss"),
            ("losses/entropy", "policy entropy"),
        ],
    ):
        series = scalars.get(tag, Series.empty())
        _plot_line(ax, series, title, color="tab:purple")
        ax.set_xlabel("steps")
        ax.set_title(title)
    fig.suptitle("Optimization diagnostics")
    _save(fig, output)


def plot_action_distribution(baseline: dict, trained: dict, output: Path) -> None:
    """Grouped bar chart contrasting random vs trained policy action distributions."""
    b = baseline["action_distribution"]
    t = trained["action_distribution"]
    # Preserve the same action order across both series.
    actions = list(b.keys())
    b_vals = np.asarray([b[a] for a in actions]) * 100
    t_vals = np.asarray([t.get(a, 0.0) for a in actions]) * 100

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(actions))
    width = 0.4
    ax.bar(x - width / 2, b_vals, width, label="random baseline", color="tab:red", alpha=0.75)
    ax.bar(x + width / 2, t_vals, width, label="trained PPO", color="tab:blue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(actions, rotation=20, ha="right")
    ax.set_ylabel("action share (%)")
    ax.set_title("Action distribution: random baseline vs trained PPO")
    ax.legend()
    _save(fig, output)


def plot_checkpoint_progression(
    progression: dict,
    output: Path,
    baseline: dict | None = None,
) -> None:
    """3-panel progression: return / success / trees, one point per checkpoint."""
    records = progression["records"]
    if not records:
        return
    steps = np.asarray([r["env_steps"] for r in records])
    ret = np.asarray([r["episode_return_mean"] for r in records])
    ret_std = np.asarray([r["episode_return_std"] for r in records])
    succ = np.asarray([r["success_rate"] for r in records]) * 100
    trees = np.asarray([r["trees_chopped_mean"] for r in records])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(steps, ret, "o-", color="tab:blue", linewidth=2, markersize=5)
    axes[0].fill_between(steps, ret - ret_std, ret + ret_std, alpha=0.18, color="tab:blue")
    axes[0].set_title(f"episode return (stochastic, {progression['episodes']} ep/ckpt)")
    axes[0].set_ylabel("return")

    axes[1].plot(steps, succ, "o-", color="tab:orange", linewidth=2, markersize=5)
    axes[1].set_title("success rate")
    axes[1].set_ylabel("success (%)")

    axes[2].plot(steps, trees, "o-", color="tab:brown", linewidth=2, markersize=5)
    axes[2].set_title("trees chopped per episode")
    axes[2].set_ylabel("trees")

    if baseline is not None:
        axes[0].axhline(baseline["episode_return"]["mean"], color="tab:red", ls="--", label="random")
        axes[1].axhline(baseline["success_rate"] * 100, color="tab:red", ls="--", label="random")
        axes[2].axhline(baseline["trees_chopped"]["mean"], color="tab:red", ls="--", label="random")
        for ax in axes:
            ax.legend(loc="best", fontsize=9)

    for ax in axes:
        ax.set_xlabel("environment steps")

    fig.suptitle("Checkpoint progression — deterministic cadence, honest stochastic eval")
    _save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run directory (contains TB events).")
    parser.add_argument("--baseline-json", type=Path, default=None, help="Random-baseline eval JSON for overlay lines.")
    parser.add_argument("--trained-json", type=Path, default=None, help="Trained-agent eval JSON (for action-distribution chart).")
    parser.add_argument("--progression-json", type=Path, default=None, help="Checkpoint-progression JSON (for checkpoint_progression.png).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to save PNGs (default: <run_dir>/plots).")
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    output_dir: Path = args.output_dir if args.output_dir is not None else run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    scalars = load_scalars(run_dir)
    print(f"Loaded {len(scalars)} scalar tags from {run_dir}")

    baseline: dict | None = None
    if args.baseline_json is not None and args.baseline_json.exists():
        baseline = json.loads(args.baseline_json.read_text())
        print(f"Loaded random baseline: {args.baseline_json}")

    baseline_return = baseline["episode_return"]["mean"] if baseline else None
    baseline_success = baseline["success_rate"] if baseline else None
    baseline_trees = baseline["trees_chopped"]["mean"] if baseline else None
    baseline_length = baseline["episode_length"]["mean"] if baseline else None

    plot_reward(scalars, output_dir / "reward_over_time.png", baseline_return=baseline_return)
    plot_single(
        scalars,
        tag="charts/episode_length",
        title="Episode length over training",
        ylabel="steps per episode",
        output=output_dir / "episode_length.png",
        baseline=baseline_length,
        baseline_label=f"random ({baseline_length:.0f})" if baseline_length is not None else None,
    )
    plot_single(
        scalars,
        tag="charts/success_rate",
        title="Success rate (inventory filled) over training",
        ylabel="success rate",
        output=output_dir / "success_rate.png",
        baseline=baseline_success,
        baseline_label=f"random ({baseline_success*100:.1f}%)" if baseline_success is not None else None,
        color="tab:orange",
        y_percent=True,
    )
    plot_single(
        scalars,
        tag="charts/trees_chopped",
        title="Trees chopped per episode",
        ylabel="trees per episode",
        output=output_dir / "trees_chopped.png",
        baseline=baseline_trees,
        baseline_label=f"random ({baseline_trees:.2f})" if baseline_trees is not None else None,
        color="tab:brown",
    )
    plot_single(
        scalars,
        tag="losses/explained_variance",
        title="Value-function explained variance",
        ylabel="explained variance",
        output=output_dir / "explained_variance.png",
        color="tab:purple",
    )
    plot_losses(scalars, output_dir / "losses.png")

    if args.trained_json is not None and args.trained_json.exists() and baseline is not None:
        trained = json.loads(args.trained_json.read_text())
        plot_action_distribution(baseline, trained, output_dir / "action_distribution.png")
        print("Rendered action_distribution.png")
    if args.progression_json is not None and args.progression_json.exists():
        progression = json.loads(args.progression_json.read_text())
        plot_checkpoint_progression(
            progression,
            output_dir / "checkpoint_progression.png",
            baseline=baseline,
        )
        print("Rendered checkpoint_progression.png")

    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
