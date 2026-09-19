# Representative Scenario 2 implementation

This directory contains a cleaned representative implementation of the rigid-body
boundary-source safety-field model used for the Scenario 2 parallel
opposite-direction taxiing encounter in the manuscript.

The release is intentionally limited to the numerical core, the Scenario 2
definition, archived baseline time-series data, and scripts for reproducing the
representative time-series panels and boundary-risk snapshots.

## Baseline model parameters

| Symbol | Value | Unit/status | Role |
|---|---:|---|---|
| `K` | 0.5 | scaling coefficient | field-strength scale |
| `k1` | 1.2 | dimensionless | distance-attenuation exponent |
| `k3` | 45 | m/s | phenomenological speed-scale parameter in the dynamic correction |
| `d0` | 1.6 | m | safety-distance correction |
| `eps` | 0.35 | m | small-separation smoothing parameter |
| `alpha` | 0.06 | dimensionless | weight of normalized instantaneous ASPE |
| `N` | 384 | points/aircraft | boundary discretization |
| `lambda_M` | 1.0e6 | numerical weight | aircraft-level source-strength target |
| `lambda_L` | 3.0e-3 | numerical weight | adjacency smoothness regularization |
| `M` | 1 | normalized | aircraft-level total source strength |
| `E_ref` | 516.4152141159 | reference ASPE scale | maximum aircraft-level ASPE in the deterministic nominal Scenario 2 encounter |
| `T_ref` | 1.0 | s | reference time scale |

### Interpretation of `k3`

The dynamic correction retains a Doppler-inspired mathematical form adapted from
Wang et al. (2016), *Transportation Research Part C*, 72, 306-324.
In the present aircraft model, `k3` is treated as a **phenomenological speed-scale
parameter**, not as a physical wave-propagation speed. The numerical baseline
value is 45 and is expressed in the same velocity unit used for local aircraft
boundary velocities in the implementation (m/s). Its influence is assessed
separately in the manuscript's parameter-sensitivity analysis.

No calibrated universal "high-risk" threshold is used in this public example.

## Scenario 2 baseline encounter

- Lane/route separation: 21 m.
- Aircraft A: starts at `(-76, 10.5)` m, travels toward `+x` at 5.2 m/s,
  heading 0 deg.
- Aircraft B: starts at `(76, -10.5)` m, travels toward `-x` at 5.2 m/s,
  heading 180 deg.
- Time step of the archived baseline series: 0.1 s.
- The interaction terminates when an aircraft reaches its prescribed route endpoint.
- Publication display interval: 6.0-17.0 s.
- Bar sampling interval in the displayed panels: 0.25 s.
- Representative snapshots: 9.7 s and 14.6 s.

## Files

- `core_model.py` — cleaned numerical core for boundary sampling, dynamic
  influence construction, constrained virtual-mass estimation, ASPE/ASPE-rate,
  ADSI, and representative snapshot rendering.
- `plot_timeseries.py` — reconstructs `alpha=0.06` ADSI from archived raw
  ASPE/ASPE-rate data and produces the three time-series panels.
- `plot_snapshots.py` — recomputes the two representative boundary-risk snapshots
  using the numerical core.
- `run_reproduce.py` — runs both reproduction scripts.
- `data/raw_aspe_timeseries.csv` — archived raw aircraft-level ASPE and ASPE-rate
  quantities plus state variables; obsolete derived ADSI columns from earlier
  development versions have been removed.
- `data/comparison_geometry_timeseries.csv` — Following-model comparison series
  and independent minimum-boundary-clearance series used in the published
  comparison.
- `data/normalization_reference.csv` — definition and value of `E_ref` and
  `T_ref`.
- `data/summary.csv` — key deterministic values.
- `reference_output/` — reference PNGs from the retained final plotting version.

The Following-model series is included as archived comparison data. It is not
part of the proposed boundary-source model core.

## Run

Python 3.9+ is recommended.

```bash
pip install -r requirements.txt
python run_reproduce.py
```

Generated figures are written to `output/`.

The time-series script uses the archived raw ASPE/ASPE-rate values so that it
does not repeat the expensive full 293-state constrained inversion. The snapshot
script performs fresh model evaluations at 9.7 s and 14.6 s.

## Expected deterministic checks

The retained final baseline gives approximately:

- Peak Total ADSI: `0.0987117` at `13.1 s`.
- Minimum boundary clearance: `1.20 m`.
- Closest-boundary-approach interval: `15.1-15.5 s`.
- Following-model peak time: `12.9 s`.

Small rendering differences can occur when Times New Roman is unavailable;
the numerical data are unaffected.
