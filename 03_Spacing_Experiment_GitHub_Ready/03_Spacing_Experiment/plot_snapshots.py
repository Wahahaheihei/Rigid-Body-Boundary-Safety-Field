# -*- coding: utf-8 -*-
"""Recompute the two representative Scenario 2 boundary-risk snapshots."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors

import core_model as core
import plot_util as util


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT_TIMES = (9.7, 14.6)


def get_steps():
    shape = core.AircraftShape(core.CONFIG["SHAPE"])
    trajs = core.build_parallel_opposite_trajectories(core.CONFIG["PARALLEL_OPPOSITE_SCENE"])
    steps = []

    for t in SNAPSHOT_TIMES:
        objects, states = core.build_scene_objects_at_time(t, trajs)
        step = core.compute_single_time_step_multi(shape, objects)
        step["objects"] = objects
        step["A_center"] = states["A"]["center"]
        step["A_vel"] = states["A"]["vel"]
        step["B_center"] = states["B"]["center"]
        step["B_vel"] = states["B"]["vel"]
        steps.append(step)

    return shape, trajs, steps


def save():
    font = util.set_english_font()
    shape, trajs, steps = get_steps()

    all_adsi = np.concatenate([s["ADSI"] for s in steps])
    vmin = float(np.percentile(all_adsi, 2))
    vmax = float(np.percentile(all_adsi, 98))
    if vmax - vmin < 1e-12:
        vmax = vmin + 1e-12

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.2))

    for idx, (ax, step) in enumerate(zip(axes, steps), start=1):
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*core.CONFIG["X_LIM"])
        ax.set_ylim(*core.CONFIG["Y_LIM"])
        ax.grid(True, alpha=0.12)
        core.draw_parallel_route_guides(ax, trajs)

        polygons = [
            shape.world_polys(np.asarray(obj["center"]), float(obj["heading"]))
            for obj in step["objects"]
        ]

        for i in range(len(step["objects"])):
            mask = step["labels"] == i
            core.draw_light_band(
                ax,
                polygons[i],
                step["object_points"][i],
                step["ADSI"][mask],
                core.CONFIG["RENDER"]["CMAP"],
                vmin,
                vmax,
            )

        core.draw_aircraft(
            ax, polygons[0], color="white", alpha=1.0,
            edge_width=1.2, zorder=6, edge_color="royalblue",
        )
        core.draw_aircraft(
            ax, polygons[1], color="white", alpha=1.0,
            edge_width=1.2, zorder=6, edge_color="crimson",
        )
        core.draw_velocity_arrow(ax, step["A_center"], step["A_vel"], "royalblue", scale=4.8, zorder=8)
        core.draw_velocity_arrow(ax, step["B_center"], step["B_vel"], "crimson", scale=4.8, zorder=8)

        ax.set_title(f"({chr(96 + idx)}) Snapshot {idx} (t={SNAPSHOT_TIMES[idx - 1]:.1f} s)", fontsize=14)
        ax.tick_params(labelbottom=False, labelleft=False)

    scalar = cm.ScalarMappable(
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
        cmap=core.CONFIG["RENDER"]["CMAP"],
    )
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=list(axes), fraction=0.025, pad=0.018)
    colorbar.set_label("Boundary ADSI", fontsize=12)

    out_path = OUT_DIR / "scenario2_representative_risk_snapshots.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    print(f"font={font}")
    print(f"Snapshot color scale: vmin={vmin:.8f}, vmax={vmax:.8f}")


if __name__ == "__main__":
    save()
