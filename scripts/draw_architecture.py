"""Render the architecture diagram used in the README.

Outputs ``docs/architecture.png``. Kept as a script (not a hand-made SVG) so the
diagram stays in sync with the code — re-run whenever the pipeline changes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BOX_STYLE = "round,pad=0.12,rounding_size=0.08"
ARROW_STYLE = "->,head_length=5,head_width=3"


def _box(ax, xy, w, h, text, color="#e8eef7", edge="#3b5a8a", font=9, bold=False):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=BOX_STYLE,
        linewidth=1.2,
        edgecolor=edge,
        facecolor=color,
    )
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=font,
        weight=weight,
        wrap=True,
    )


def _arrow(ax, start, end, color="#444", lw=1.2, style=None):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style or ARROW_STYLE,
        linewidth=lw,
        color=color,
        mutation_scale=1.0,
    )
    ax.add_patch(arrow)


def main() -> None:
    out = Path("docs/architecture.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    # -- Perception row (top-left & top-right)
    _box(ax, (0.3, 5.6), 2.4, 0.9, "2D Simulator\n(MockOSRSClient)", color="#e7f0e4", edge="#4c7a3d", bold=True)
    _box(ax, (9.3, 5.6), 2.4, 0.9, "Live Screen Capture\n(mss)", color="#fde9d9", edge="#b96b1c", bold=True)

    # -- Preprocessing
    _box(ax, (4.6, 5.6), 2.8, 0.9, "Preprocessing\ngrayscale · resize 84 · framestack 4", color="#eef2f6", edge="#3b5a8a")

    # Arrows from perception into preprocessing
    _arrow(ax, (2.7, 6.05), (4.6, 6.05))
    _arrow(ax, (9.3, 6.05), (7.4, 6.05))

    # -- Gym env + rewards
    _box(ax, (4.1, 3.9), 3.8, 1.0, "Gymnasium env (OSRSEnv)\n+ CompositeReward", color="#e8eef7", edge="#3b5a8a", bold=True)
    _arrow(ax, (6.0, 5.6), (6.0, 4.9))

    # -- PPO policy
    _box(ax, (4.1, 2.0), 3.8, 1.2, "PPO policy (from scratch)\nNature-CNN  →  actor + critic heads\nGAE · clip · entropy · LR-anneal",
         color="#eae0f4", edge="#5b3a82", bold=True)
    _arrow(ax, (6.0, 3.9), (6.0, 3.2))

    # Return value path
    _arrow(ax, (7.9, 4.4), (7.9, 3.2))
    ax.text(7.95, 3.5, "action int", fontsize=8, rotation=-90, va="center", color="#444")
    _arrow(ax, (4.1, 3.2), (4.1, 4.4))
    ax.text(4.0, 3.7, "obs / reward", fontsize=8, rotation=90, va="center", ha="right", color="#444")

    # -- Checkpoints / TB (right side of policy row)
    _box(ax, (9.3, 2.1), 2.4, 0.9, "TensorBoard +\nCheckpoints", color="#fef6e4", edge="#b99421")
    _arrow(ax, (7.9, 2.6), (9.3, 2.6))

    # -- ActionDecoder
    _box(ax, (4.1, 0.4), 3.8, 0.9, "ActionDecoder\ndiscrete int → typed Action", color="#eef2f6", edge="#3b5a8a")
    _arrow(ax, (6.0, 2.0), (6.0, 1.3))

    # -- Simulator step (left below)
    _box(ax, (0.3, 0.4), 3.1, 0.9, "Simulator step\n(fast training)", color="#e7f0e4", edge="#4c7a3d")
    _arrow(ax, (4.1, 0.85), (3.4, 0.85))

    # -- Safety gate + Controller (right below)
    _box(ax, (8.5, 0.4), 3.2, 0.9, "SafetyGate  →  Input Controller\nbbox · rate-limit · kill-switch",
         color="#fde9d9", edge="#b96b1c")
    _arrow(ax, (7.9, 0.85), (8.5, 0.85))

    # Bidirectional world arrows (simulator ↔ perception, input ↔ live)
    _arrow(ax, (1.5, 1.3), (1.5, 5.6), color="#4c7a3d", lw=1.1, style="-")
    ax.text(1.3, 3.5, "world state", fontsize=8, rotation=90, va="center", ha="right", color="#4c7a3d")
    _arrow(ax, (10.1, 1.3), (10.1, 5.6), color="#b96b1c", lw=1.1, style="-")
    ax.text(10.28, 3.5, "OS input", fontsize=8, rotation=90, va="center", color="#b96b1c")

    # -- Title + caption
    fig.suptitle("OSRS-RL — Architecture", fontsize=13, weight="bold")
    fig.text(
        0.5,
        0.01,
        "Same Gymnasium env, same policy, same checkpoint — only the injected GameClient differs between training (simulator) and live evaluation.",
        ha="center",
        fontsize=9,
        style="italic",
        color="#555",
    )

    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
