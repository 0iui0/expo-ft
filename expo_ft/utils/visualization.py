"""Online RL visualization: Q-function, sampling candidates, and edit actions.

Three plot families logged to wandb at episode boundaries:

1. **Q Visualization** — histogram and time-series of Q-values for selected
   actions on successful vs failed episode steps.

2. **Sampling Visualization** — candidate action trajectories (N base VLA
   proposals + residual edits) with the argmax-Q selection highlighted.

3. **Edit Visualization** — before/after comparison of base vs residual-edited
   action chunks over the replan horizon.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
#  Matplotlib setup (lazy import so the module loads even if no display)
# ──────────────────────────────────────────────────────────────────────────────

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# Dimension group labels for CR5AF action (16-D: eef_9d + joint_pos + gripper_pos)
_ACTION_GROUPS = {
    "eef_9d":  (0, 9,   ["x", "y", "z", "qx", "qy", "qz", "qw", "grip_width", "force"]),
    "joint_pos": (9, 15, [f"j{i}" for i in range(6)]),
    "gripper_pos": (15, 16, ["gripper"]),
}


def _action_group_name(dim: int) -> str:
    for name, (start, end, _) in _ACTION_GROUPS.items():
        if start <= dim < end:
            return name
    return "other"


# ──────────────────────────────────────────────────────────────────────────────
#  Data capture  (called once per policy step, stored in EpisodeState)
# ──────────────────────────────────────────────────────────────────────────────

def capture_step_data(
    agent_sample_info: dict,
) -> dict:
    """Extract visualization-relevant fields from ``sample_actions`` info dict.

    Returns a flat dict with numpy arrays that the plot functions consume.
    Returns an empty dict when ``only_base_actions`` was used (no Q values).
    """
    si = agent_sample_info
    if not si or "qs" not in si:
        return {}

    qs = np.asarray(si["qs"])                                    # (n_cand,)
    all_chunks = np.asarray(si["all_chunks"])                      # (n_cand, H, D)
    selected_idx = int(np.asarray(si["selected_idx"]))             # scalar
    n_edit = int(si.get("n_edit", 0))

    n_cand = qs.shape[0]
    is_edit = np.zeros(n_cand, dtype=bool)
    if n_edit > 0:
        is_edit[-n_edit:] = True  # last n_edit are residual-edited chunks

    return {
        "qs": qs,                           # (n_cand,)
        "all_chunks": all_chunks,            # (n_cand, H, D)
        "selected_idx": selected_idx,        # scalar
        "is_edit": is_edit,                  # (n_cand,) bool mask
        "n_edit": n_edit,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Plot generators
# ──────────────────────────────────────────────────────────────────────────────

def plot_q_visualization(
    step_data_list: list,
    success: bool,
    save_dir: str | Path,
    prefix: str = "q_viz",
) -> Optional[str]:
    """Histogram + time-series of Q-values for the selected action.

    ``step_data_list`` is a list of captured dicts (one per policy step).
    Returns the path to the saved PNG, or None if insufficient data.
    """
    plt = _mpl()
    selected_qs = []
    for sd in step_data_list:
        if sd and "qs" in sd:
            q_selected = float(sd["qs"][sd["selected_idx"]])
            selected_qs.append(q_selected)

    if len(selected_qs) < 3:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- Left: histogram of selected-action Q values ---
    axes[0].hist(selected_qs, bins=min(20, len(selected_qs) // 2),
                 color="#4C72B0", alpha=0.8, edgecolor="white")
    axes[0].axvline(np.mean(selected_qs), color="#C44E52", linestyle="--",
                    label=f"mean={np.mean(selected_qs):.3f}")
    axes[0].set_xlabel("Q-value")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Q-values of selected actions ({'SUCCESS' if success else 'FAILURE'})")
    axes[0].legend()

    # --- Right: Q-value time series ---
    xs = np.arange(len(selected_qs))
    axes[1].plot(xs, selected_qs, color="#4C72B0", linewidth=1.5, marker=".", markersize=3)
    axes[1].axhline(0, color="gray", linestyle=":", linewidth=0.5)
    axes[1].set_xlabel("Policy step")
    axes[1].set_ylabel("Q-value of selected action")
    axes[1].set_title("Q-value time series")
    axes[1].set_xlim(0, len(selected_qs) - 1)

    fig.suptitle(f"Episode {'SUCCESS' if success else 'FAILURE'}", fontsize=14, y=1.02)
    fig.tight_layout()
    save_path = Path(save_dir) / f"{prefix}_{'success' if success else 'failure'}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Q visualization: %s", save_path)
    return str(save_path)


def plot_candidate_trajectories(
    step_data: dict,
    step_idx: int,
    save_dir: str | Path,
    prefix: str = "sampling",
) -> Optional[str]:
    """Plot candidate action trajectories for one policy step.

    Shows all N base VLA proposals (thin gray) + residual edits (dashed blue)
    with the argmax-Q selection in bold red. One panel per action group.
    """
    plt = _mpl()
    if not step_data or "all_chunks" not in step_data:
        return None

    chunks = step_data["all_chunks"]       # (n_cand, H, D)
    H, D = chunks.shape[1], chunks.shape[2]
    selected_idx = step_data["selected_idx"]
    is_edit = step_data["is_edit"]
    n_cand = chunks.shape[0]

    # Determine groups from D
    groups = []
    offset = 0
    for name, (start, end, labels) in _ACTION_GROUPS.items():
        if offset + start < D:
            s = offset + start
            e = min(offset + end, D)
            if e > s:
                groups.append((name, s, e, labels[:e - s] if isinstance(labels, (list, tuple)) else labels))

    if not groups:
        # fallback: single panel
        groups = [("action", 0, D, [f"d{d}" for d in range(D)])]

    fig, axes = plt.subplots(len(groups), 1, figsize=(14, 3 * len(groups)),
                             squeeze=False)
    ts = np.arange(H)

    for row, (gname, gs, ge, glabels) in enumerate(groups):
        ax = axes[row, 0]
        # Plot each candidate dim as a thin line
        for ci in range(n_cand):
            style = "--" if is_edit[ci] else "-"
            alpha = 0.25 if is_edit[ci] else 0.12
            color = "#4878D0" if is_edit[ci] else "#AAAAAA"
            lw = 0.5
            for dim_in_group in range(ge - gs):
                ax.plot(ts, chunks[ci, :, gs + dim_in_group],
                        style, color=color, alpha=alpha, linewidth=lw)

        # Highlight selected trajectory (bold red)
        sel = chunks[selected_idx]
        for dim_in_group in range(ge - gs):
            ax.plot(ts, sel[:, gs + dim_in_group], "-", color="#C44E52",
                    linewidth=1.8, alpha=0.9)

        # Annotate which type was selected
        sel_type = "EDIT" if is_edit[selected_idx] else "BASE"
        ax.set_title(f"{gname}  |  selected index={selected_idx} ({sel_type})")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Action value")

    fig.suptitle(f"Candidate trajectories at policy step {step_idx}",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    save_path = Path(save_dir) / f"{prefix}_step{step_idx}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def plot_edit_comparison(
    step_data: dict,
    step_idx: int,
    save_dir: str | Path,
    prefix: str = "edit",
) -> Optional[str]:
    """Base vs edited action comparison for one policy step.

    When an edit action was selected, show across the replan horizon:
    - Base action trajectory (thin) vs edited (bold) for each edited candidate
    - Panel per action-dimension group
    """
    plt = _mpl()
    if not step_data or "all_chunks" not in step_data:
        return None

    chunks = step_data["all_chunks"]        # (n_cand, H, D)
    H, D = chunks.shape[1], chunks.shape[2]
    is_edit = step_data["is_edit"]
    n_edit = step_data["n_edit"]
    selected_idx = step_data["selected_idx"]

    if n_edit == 0:
        return None  # no editing active in this rollout

    n_base = chunks.shape[0] - n_edit

    # Groups
    groups = []
    offset = 0
    for name, (start, end, labels) in _ACTION_GROUPS.items():
        if offset + start < D:
            s = offset + start
            e = min(offset + end, D)
            if e > s:
                groups.append((name, s, e, labels[:e - s]))

    fig, axes = plt.subplots(len(groups), 1, figsize=(14, 3 * len(groups)),
                             squeeze=False)
    ts = np.arange(H)

    for row, (gname, gs, ge, glabels) in enumerate(groups):
        ax = axes[row, 0]

        # Base version of the selected candidate (before edit)
        # For edit candidates, the base version is the first n_base chunks at same index
        base_idx = selected_idx if selected_idx < n_base else selected_idx - n_edit
        for dim_in_group in range(ge - gs):
            base_line = chunks[base_idx, :, gs + dim_in_group]
            ax.plot(ts, base_line, "-", color="#AAAAAA", linewidth=1.0, alpha=0.6)

        if is_edit[selected_idx]:
            # Edited version of selected candidate
            for dim_in_group in range(ge - gs):
                edit_line = chunks[selected_idx, :, gs + dim_in_group]
                ax.plot(ts, edit_line, "-", color="#C44E52", linewidth=1.8, alpha=0.9)

            # Show all edit deltas in background
            for ei in range(n_edit):
                edit_ci = n_base + ei
                base_ci = ei if ei < n_base else 0
                for dim_in_group in range(ge - gs):
                    delta = chunks[edit_ci, :, gs + dim_in_group] - chunks[base_ci, :, gs + dim_in_group]
                    ax.plot(ts, delta, ":", color="#E5A262", linewidth=0.4, alpha=0.3)

            ax.set_title(f"{gname}  |  base (gray) → edit (red), deltas (dots)")
        else:
            ax.set_title(f"{gname}  |  base trajectory (base was selected)")

        ax.set_xlabel("Time step (replan horizon)")
        ax.set_ylabel("Action value")

    fig.suptitle(f"Edit comparison at policy step {step_idx}",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    save_path = Path(save_dir) / f"{prefix}_step{step_idx}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


# ──────────────────────────────────────────────────────────────────────────────
#  Top-level orchestrator called at episode end
# ──────────────────────────────────────────────────────────────────────────────

def log_episode_visualizations(
    episode_log,  # EpisodeState with step_data_history list
    success: bool,
    save_dir: str | Path,
    *,
    max_sampling_plots: int = 3,
) -> list[tuple[str, str]]:
    """Generate and log all three visualization families.

    Returns a list of ``(tag, filepath)`` tuples suitable for wandb.log:
    ``wandb.log({tag: wandb.Image(path)})``
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    step_data_list = getattr(episode_log, "step_data_history", []) or []
    # Filter out empty entries (non-policy steps or no Q data)
    step_data_list = [sd for sd in step_data_list if sd and "qs" in sd]
    if not step_data_list:
        logger.info("No Q data captured for episode visualizations")
        return []

    images = []

    # 1. Q visualization (always)
    q_path = plot_q_visualization(step_data_list, success, save_dir)
    if q_path:
        images.append(("vis/q_histogram", q_path))

    # 2. Sampling & edit visualizations (sample a few steps evenly)
    n_steps = len(step_data_list)
    n_plots = min(max_sampling_plots, n_steps)
    indices = np.linspace(0, n_steps - 1, n_plots, dtype=int).tolist()

    for si in indices:
        sd = step_data_list[si]
        samp_path = plot_candidate_trajectories(sd, si, save_dir)
        if samp_path:
            images.append((f"vis/sampling_step{si}", samp_path))

        edit_path = plot_edit_comparison(sd, si, save_dir)
        if edit_path:
            images.append((f"vis/edit_step{si}", edit_path))

    logger.info("Generated %d visualization images for episode", len(images))
    return images
