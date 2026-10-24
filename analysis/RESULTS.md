# Chaos results: when do shape-matching methods fail?

All numbers here come from the scripts in this folder. Accuracy is on held-out series; "chance"
is the accuracy of always guessing the majority class of the test set (about 0.57 with the app's
unstratified 70/30 split of 50 series).

## The task

Each dataset holds 25 real chaotic series and 25 fakes. A fake is an IAAFT surrogate: the same
values and almost the same power spectrum as a real series, with the order scrambled. The only
thing that separates the classes is determinism (whether consecutive values follow a rule).
Real and fake series in a dataset come from different orbits. Series are 100 points long and
z-normalised.

Methods compared:

| Method | Kind |
|---|---|
| Rocket (500 kernels), MiniRocket | shape: short random convolution kernels |
| catch22 | 22 summary features |
| Dynamics (NN) | nearest-neighbour one-step prediction error in delay embeddings, d = 1, 2, 3 |
| Oracle (logistic map only) | knows the rule x -> r x (1 - x), fits the unknown mean, spread and r, and measures how badly the series breaks the rule |

λΔt is the largest Lyapunov exponent times the time between recorded points: how much a tiny
error grows between two recorded values.

## Result 1: logistic map, sampling curve (`chaos_logistic/`)

10 datasets per setting, mean accuracy (standard error at most 0.07).

| Recorded | λΔt | Rocket | MiniRocket | catch22 | Dynamics (NN) | Oracle |
|---|---|---|---|---|---|---|
| every step | 0.57 | 0.99 | 0.98 | 0.99 | 1.00 | 1.00 |
| every 2nd | 1.14 | 0.77 | 0.57 | 0.89 | 1.00 | 1.00 |
| every 3rd | 1.71 | 0.51 | 0.52 | 0.55 | 1.00 | 1.00 |
| every 4th | 2.28 | 0.44 | 0.44 | 0.62 | 1.00 | 0.99 |
| every 6th | 3.43 | 0.41 | 0.45 | 0.51 | 0.99 | 0.74 |
| every 8th | 4.57 | 0.45 | 0.50 | 0.45 | 0.75 | 0.48 |

- Shape methods fall from about 0.99 to chance by λΔt ≈ 1.7; the dynamics method stays at 1.00
  up to λΔt ≈ 3.4.
- The oracle shows the task is solvable up to λΔt ≈ 2.3. Its drop at every 6th/8th step is our
  fitting procedure failing on the many-times-applied rule (the dynamics method still scores 0.99
  at every 6th step, so the information is there).

Observation noise (added before normalising, as a fraction of the series' standard deviation):

| Recorded | Noise | Rocket | MiniRocket | catch22 | Dynamics (NN) |
|---|---|---|---|---|---|
| every 4th | 0 | 0.44 | 0.44 | 0.62 | 1.00 |
| every 4th | 1% | 0.49 | 0.54 | 0.58 | 0.99 |
| every 4th | 5% | 0.54 | 0.52 | 0.51 | 0.98 |
| every 4th | 10% | 0.52 | 0.51 | 0.55 | 0.77 |

At every step, all methods stay at 0.87 to 1.00 for these noise levels. The oracle is not
designed for noisy data, so its noisy scores are not used.

## Result 2: 127 dysts systems at natural sampling (`chaos_dysts/`)

Each system recorded at 20 points per dominant period (λΔt median 0.046, range 0.002 to 1.2).
"Unseen systems": 10 folds split by system, so every system is tested by a model that never
saw it.

| Mean accuracy | Rocket | MiniRocket | catch22 | Dynamics (NN) |
|---|---|---|---|---|
| Same system | 0.95 | 0.96 | 0.88 | 0.89 |
| Unseen systems | 0.77 | 0.82 | 0.76 | 0.78 |

- No method's accuracy depends on the Lyapunov exponent (Spearman correlation with λΔt between
  −0.12 and 0, none significant).
- The dynamics method beats the best shape method by more than 0.1 on 1 of 127 systems.

## Result 3: the same systems at λΔt = 2 (`chaos_dysts/`)

113 systems (6 delay systems excluded; 14 failed to integrate or took over 120 s).

| Mean accuracy | Rocket | MiniRocket | catch22 | Dynamics (NN) |
|---|---|---|---|---|
| Same system | 0.78 | 0.74 | 0.72 | 0.74 |
| Unseen systems | 0.54 | 0.55 | 0.59 | 0.58 |

- Coarse sampling makes every method worse, not only the shape methods.
- The dynamics method beats the best shape method by more than 0.1 on 8 of 113 systems (largest:
  SprottC +0.41, SprottTorus +0.23, MultiChua +0.21). On 42 systems it loses by more than 0.1.
- Lorenz at λΔt = 2: Rocket 0.79, catch22 0.83, dynamics 0.66. Rössler: Rocket 0.98, dynamics 0.98.

## Conclusion

1. Shape-based classifiers detect finely sampled chaos easily, across 127 systems, because a
   deterministic series is locally regular and short kernels read that.
2. Their success does not depend on the Lyapunov exponent itself; what matters is λΔt, the
   scrambling between recorded points.
3. For the logistic map there is a clear window (λΔt ≈ 1.7 to 3.4) where shape methods are at
   chance while the dynamics method is near perfect and the oracle confirms the task is solvable.
4. For continuous flows, coarse sampling mostly removes information for every method: a clear
   gap appears for only a few systems (8 of 113).

## Limits

- Small datasets (50 series of 100 points); per-system accuracies are noisy, and the "gap" against
  the best of three shape methods is biased slightly negative by taking a maximum.
- The dynamics method is a simple nearest-neighbour statistic; it weakens at 10% noise.
- The oracle exists only for the logistic map.
- λ values for the dysts systems are the library's own estimates.
- One fake method (IAAFT) throughout.
