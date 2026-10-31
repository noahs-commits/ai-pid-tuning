"""Matplotlib helpers for visualising simulation runs."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def plot_trajectory(
    positions_list: Sequence[float],
    targets: Sequence[float] | None = None,
    *,
    times: Sequence[float] | None = None,
    dt: float | None = None,
    velocities: Sequence[float] | None = None,
    actions: Sequence[float] | None = None,
    title: str = "1D PID simulation",
    save_path: str | None = None,
    show: bool = True,
) -> tuple[Any, Any]:
    """Plot position and setpoint over time.

    Parameters
    ----------
    positions, targets:
        Position and setpoint samples.  ``targets`` may be ``None``.
    times:
        Time for each sample.  If omitted, ``dt`` is used; if both are
        omitted the sample index is used.
    velocities, actions:
        Optional extra traces.  When either is supplied a three-panel figure
        (position/setpoint, velocity, action) is created.
    save_path:
        Save the figure to this path.
    show:
        Call ``matplotlib.pyplot.show()`` (useful in scripts/notebooks).

    Returns
    -------
    (fig, axes)
        The matplotlib figure and its axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "matplotlib is required for plotting; install it with "
            "`pip install matplotlib`"
        ) from exc

    positions = np.asarray(positions_list, dtype=np.float64)
    if positions.ndim != 1 or positions.size == 0:
        raise ValueError("positions must be a non-empty 1D sequence")

    targets_arr = None if targets is None else np.asarray(targets, dtype=np.float64)

    if times is None:
        if dt is None:
            times_arr = np.arange(positions.size, dtype=np.float64)
        else:
            times_arr = np.arange(positions.size, dtype=np.float64) * dt
    else:
        times_arr = np.asarray(times, dtype=np.float64)

    if times_arr.size != positions.size:
        raise ValueError("times must have the same length as positions")
    if targets_arr is not None and targets_arr.size != positions.size:
        raise ValueError("targets must have the same length as positions")

    has_extra = velocities is not None or actions is not None
    axes: Any
    if has_extra:
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 7))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(9, 4))

    ax_pos = axes[0] if has_extra else axes
    ax_pos.plot(times_arr, positions, label="position", color="tab:blue")
    if targets_arr is not None:
        ax_pos.plot(
            times_arr,
            targets_arr,
            "--",
            label="setpoint",
            color="tab:orange",
        )
    ax_pos.set_ylabel("position (m)")
    ax_pos.set_title(title)
    ax_pos.legend()
    ax_pos.grid(True)

    if has_extra:
        if velocities is not None:
            axes[1].plot(times_arr, velocities, color="tab:green")
        axes[1].set_ylabel("velocity (m/s)")
        axes[1].grid(True)

        if actions is not None:
            axes[2].plot(times_arr, actions, color="tab:red")
        axes[2].set_ylabel("action")
        axes[2].set_xlabel("time (s)")
        axes[2].grid(True)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
    if show:
        plt.show()

    return fig, axes
