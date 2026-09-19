# -*- coding: utf-8 -*-
"""Reproduce the public Scenario 2 example figures."""
from plot_timeseries import build as build_timeseries
from plot_snapshots import save as build_snapshots


if __name__ == "__main__":
    print("=" * 72)
    print("Scenario 2 representative reproduction")
    print("alpha=0.06, E_ref=516.4152141159, T_ref=1.0 s, k3=45 m/s")
    print("=" * 72)

    build_timeseries()
    print("\nRecomputing the two representative boundary-risk snapshots...")
    build_snapshots()

    print("\nDone. See ./output")
