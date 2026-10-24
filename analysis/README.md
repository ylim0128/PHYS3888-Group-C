# analysis/

Offline experiments behind the chaos results in the report. They do not change the app.
Results and conclusions: [RESULTS.md](RESULTS.md).

## chaos_logistic/
- `logistic_curve.py curve` - logistic map at every 1st..8th step: Rocket, MiniRocket, catch22,
  Dynamics (NN) and the oracle. Writes `logistic_curve.csv`.
- `logistic_curve.py noise` - every step and every 4th step with 1%, 5%, 10% observation noise.
  Writes `logistic_noise.csv`.
- `logistic_sampling_curve.png` - the main figure.

## chaos_dysts/
Needs `pip install dysts numba` on top of requirements.txt.
- `run_build.sh` - builds 25 real windows + 25 IAAFT fakes per dysts system
  (`./run_build.sh` natural sampling, `./run_build.sh 2.0` fixed λΔt = 2). Uses `build_one.py`.
- `evaluate.py DATA_DIR OUT.csv` - scores every method per system (same system, 10 splits) and on
  unseen systems (10 folds split by system).
- `results_natural_sampling.csv`, `results_lambda_dt_2.csv` - per-system results.
- `natural_sampling_vs_lyapunov.png`, `natural_vs_lambda_dt_2.png` - figures.

Runtimes on 2 cores: building about 5 min (natural) and 60 min (λΔt = 2); scoring about 10 min each.
