# -*- coding: utf-8 -*-
"""Reproduce the Scenario 2 time-series comparison panels.

The full 0-29.2 s archived time history is retained in the data files. The
publication panel displays only the active interaction interval 6-17 s and
samples bars at 0.25 s.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import plot_util as util


ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "data" / "raw_aspe_timeseries.csv"
REFERENCE_CSV = ROOT / "data" / "comparison_geometry_timeseries.csv"
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.06
E_REF = 516.4152141159
T_REF = 1.0

DISPLAY_T_MIN = 6.0
DISPLAY_T_MAX = 17.0
BAR_SAMPLE_DT = 0.25
SNAPSHOT_TIMES = (9.7, 14.6)


def save_figure(fig, png_path):
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def first_last_min_indices(values):
    values = np.asarray(values, dtype=float)
    minimum = np.nanmin(values)
    idx = np.where(np.isclose(values, minimum))[0]
    return float(minimum), int(idx[0]), int(idx[-1])


def sample_indices(times, target_dt):
    times = np.asarray(times, dtype=float)
    dt = float(np.median(np.diff(times)))
    step = max(1, int(round(target_dt / max(dt, 1e-12))))
    idx = np.arange(0, len(times), step, dtype=int)
    if idx[-1] != len(times) - 1:
        idx = np.append(idx, len(times) - 1)
    return idx


def load_data():
    reference = pd.read_csv(REFERENCE_CSV)
    raw = pd.read_csv(RAW_CSV)

    if len(reference) != len(raw) or not np.allclose(
        reference["time"].to_numpy(float),
        raw["time"].to_numpy(float),
        atol=1e-10,
    ):
        raw = reference[["time"]].merge(raw, on="time", how="left", validate="one_to_one")

    aspe = raw["A_total_ASPE_raw"].to_numpy(float)
    aspe_rate = raw["A_total_ASPEdot_raw"].to_numpy(float)
    aspe_star = aspe / E_REF
    aspe_rate_star = (T_REF / E_REF) * aspe_rate
    adsi = ALPHA * aspe_star + (1.0 - ALPHA) * aspe_rate_star

    out = reference.copy()
    out["total_ADSI"] = adsi
    out["ASPE_star"] = aspe_star
    out["ASPEdot_star"] = aspe_rate_star
    out.to_csv(
        ROOT / "data" / "scenario2_timeseries_alpha006.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return out


def build():
    font = util.set_english_font()
    df = load_data()

    times = df["time"].to_numpy(float)
    adsi = df["total_ADSI"].to_numpy(float)
    following = df["following_model_probability"].to_numpy(float)
    clearance = df["minimum_boundary_clearance_exact"].to_numpy(float)

    min_clearance, i0, i1 = first_last_min_indices(clearance)
    t0, t1 = times[i0], times[i1]

    mask = (times >= DISPLAY_T_MIN - 1e-12) & (times <= DISPLAY_T_MAX + 1e-12)
    td = times[mask]
    ad = adsi[mask]
    fd = following[mask]
    cd = clearance[mask]

    idx = sample_indices(td, BAR_SAMPLE_DT)
    tb, ab, fb, cb = td[idx], ad[idx], fd[idx], cd[idx]
    dtb = float(np.median(np.diff(tb)))
    main_width = 0.34 * dtb
    clearance_width = 0.52 * dtb

    snapshot_idx = [int(np.argmin(abs(times - t))) for t in SNAPSHOT_TIMES]

    fig = plt.figure(figsize=(17.6, 11.2))
    gs = GridSpec(3, 1, height_ratios=[0.72, 1.05, 1.05], hspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    ax.bar(
        tb, cb, width=clearance_width,
        color="#4C8D89", edgecolor="#356C69", linewidth=0.35, alpha=0.90,
        label="Minimum boundary clearance", zorder=3,
    )
    ax.set_xlim(DISPLAY_T_MIN, DISPLAY_T_MAX)
    ax.set_ylim(0, max(10.0, float(np.nanmax(cb)) * 1.08))
    ax.set_ylabel("Minimum boundary clearance (m)", fontsize=12.5)
    ax.set_title("(a) Independent geometric reference: minimum boundary clearance", fontsize=13.2, pad=7)
    ax.grid(True, axis="y", alpha=0.18, zorder=0)
    ax.tick_params(axis="both", labelsize=10.5)
    ax.tick_params(axis="x", labelbottom=False)
    ax.legend(loc="upper right", fontsize=9.6, framealpha=0.93)
    ax.axvspan(t0, t1, color="#4C8D89", alpha=0.11, zorder=1)
    info = (
        f"Minimum clearance: {min_clearance:.2f} m   |   "
        f"Closest-approach interval: {t0:.1f}-{t1:.1f} s"
    )
    ax.text(
        0.015, 0.08, info, transform=ax.transAxes, ha="left", va="bottom",
        fontsize=9.7, color="#356C69",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#356C69", alpha=0.90),
        zorder=5,
    )
    for j in snapshot_idx:
        ax.axvline(times[j], ls="--", lw=0.90, color=".62", alpha=0.66, zorder=2)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
    ax2.bar(
        tb, ab, width=main_width, color="royalblue", edgecolor="royalblue",
        alpha=0.92, label="Aircraft A total ADSI", zorder=3,
    )
    ax2.set_xlim(DISPLAY_T_MIN, DISPLAY_T_MAX)
    ax2.set_ylim(0, max(0.12, float(np.nanmax(ab)) * 1.12))
    ax2.set_ylabel("Total ADSI", fontsize=12.6)
    ax2.set_title("(b) Total ADSI", fontsize=13.2, pad=7)
    ax2.grid(True, axis="y", alpha=0.18, zorder=0)
    ax2.tick_params(axis="both", labelsize=10.6)
    ax2.tick_params(axis="x", labelbottom=False)
    ax2.legend(loc="upper right", fontsize=9.6, framealpha=0.93)

    for k, j in enumerate(snapshot_idx):
        tx, y = times[j], adsi[j]
        ax2.axvline(tx, ls="--", lw=0.95, color=".58", alpha=0.72, zorder=2)
        ax2.scatter([tx], [y], s=38, color="royalblue", zorder=6)
        label = f"Snapshot {k + 1}"
        xytext = (-6, 20) if k == 0 else (6, -24)
        va = "bottom" if k == 0 else "top"
        ax2.annotate(
            label, xy=(tx, y), xytext=xytext, textcoords="offset points",
            ha="center", va=va, fontsize=9.6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.93, ec=".55"),
            arrowprops=dict(arrowstyle="->", lw=0.8, color=".50"), zorder=7,
        )

    ax3 = fig.add_subplot(gs[2, 0], sharex=ax)
    ax3.bar(
        tb, fb, width=main_width, color="magenta", edgecolor="magenta",
        alpha=0.82, label="Following-model collision probability", zorder=4,
    )
    ax3.set_xlim(DISPLAY_T_MIN, DISPLAY_T_MAX)
    ax3.set_ylim(0, max(0.16, float(np.nanmax(fb)) * 1.10))
    ax3.set_xlabel("Time (s)", fontsize=12.6)
    ax3.set_ylabel("Following-model collision probability", fontsize=11.8)
    ax3.set_title("(c) Following-model collision probability", fontsize=13.2, pad=7)
    ax3.grid(True, axis="y", alpha=0.18, zorder=0)
    ax3.tick_params(axis="both", labelsize=10.0)
    ax3.legend(loc="upper right", fontsize=9.6, framealpha=0.93)

    for k, j in enumerate(snapshot_idx):
        tx, y = times[j], following[j]
        ax3.axvline(tx, ls="--", lw=0.95, color=".58", alpha=0.72, zorder=2)
        ax3.annotate(
            f"Snapshot {k + 1}", xy=(tx, y), xytext=(-6 if k == 0 else 6, 20),
            textcoords="offset points", ha="center", va="bottom", fontsize=9.6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.93, ec=".55"),
            arrowprops=dict(arrowstyle="->", lw=0.8, color=".50"), zorder=7,
        )

    out_path = OUT_DIR / "scenario2_three_bar_panels.png"
    save_figure(fig, out_path)

    peak_idx = int(np.argmax(adsi))
    print(f"font={font}")
    print(f"Peak Total ADSI = {adsi[peak_idx]:.9f} at t = {times[peak_idx]:.1f} s")
    print(f"Minimum clearance = {min_clearance:.2f} m over {t0:.1f}-{t1:.1f} s")


if __name__ == "__main__":
    build()
