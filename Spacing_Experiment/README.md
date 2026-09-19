Representative Scenario 2 implementation
This directory provides a representative implementation of the rigid-body
boundary-source safety-field model used for the Scenario 2 parallel
opposite-direction taxiing encounter in the manuscript.
It includes the numerical core, baseline model parameters, Scenario 2
definition, archived time-series data, and scripts for reproducing the
representative time-series panels and boundary-risk snapshots.
Baseline model parameters
Symbol	Value	Unit/status	Role
`K`	0.5	scaling coefficient	field-strength scale
`k1`	1.2	dimensionless	distance-attenuation exponent
`k3`	45	m/s	speed-scale parameter in the dynamic correction
`d0`	1.6	m	safety-distance correction
`eps`	0.35	m	small-separation smoothing parameter
`alpha`	0.06	dimensionless	weight of normalized instantaneous ASPE
`N`	384	points/aircraft	boundary discretization
`lambda\_M`	1.0e6	numerical weight	aircraft-level source-strength target
`lambda\_L`	3.0e-3	numerical weight	adjacency smoothness regularization
`M`	1	normalized	aircraft-level total source strength
`E\_ref`	516.4152141159	reference ASPE scale	maximum aircraft-level ASPE in the deterministic nominal Scenario 2 encounter
`T\_ref`	1.0	s	reference time scale
The parameters above correspond to the baseline configuration used in the
representative Scenario 2 results. Their selection and sensitivity are discussed
in the manuscript.
Scenario 2 baseline encounter
Lane/route separation: 21 m.
Aircraft A: starts at `(-76, 10.5)` m, travels toward `+x` at 5.2 m/s,
heading 0 deg.
Aircraft B: starts at `(76, -10.5)` m, travels toward `-x` at 5.2 m/s,
heading 180 deg.
Time step of the archived baseline series: 0.1 s.
The interaction terminates when an aircraft reaches its prescribed route endpoint.
Publication display interval: 6.0-17.0 s.
Bar sampling interval in the displayed panels: 0.25 s.
Representative snapshots: 9.7 s and 14.6 s.
Files
`core\_model.py` — numerical core for boundary sampling, dynamic influence
construction, constrained virtual-mass estimation, ASPE/ASPE-rate, ADSI,
and representative snapshot rendering.
`plot\_timeseries.py` — reconstructs the baseline ADSI time series from the
archived ASPE/ASPE-rate data and produces the three time-series panels.
`plot\_snapshots.py` — recomputes the two representative boundary-risk snapshots
using the numerical core.
`run\_reproduce.py` — runs both reproduction scripts.
`data/raw\_aspe\_timeseries.csv` — archived aircraft-level ASPE and ASPE-rate
quantities together with the corresponding state variables.
`data/comparison\_geometry\_timeseries.csv` — Following-model comparison series
and independent minimum-boundary-clearance series.
`data/normalization\_reference.csv` — normalization reference values.
`data/summary.csv` — key deterministic values.
`reference\_output/` — reference figures generated from the retained baseline
configuration.
Run
Python 3.9+ is recommended.
```bash
pip install -r requirements.txt
python run\_reproduce.py
```
Generated figures are written to `output/`.
The time-series script uses the archived ASPE/ASPE-rate values to reproduce the
reported baseline series efficiently, while the snapshot script performs fresh
model evaluations at 9.7 s and 14.6 s.
Expected deterministic checks
The baseline configuration gives approximately:
Peak Total ADSI: `0.0987117` at `13.1 s`.
Minimum boundary clearance: `1.20 m`.
Closest-boundary-approach interval: `15.1-15.5 s`.
Following-model peak time: `12.9 s`.
Small rendering differences may occur across operating systems or font
configurations; the numerical results are unaffected.
