'''BUDWORM POPULATION DYNAMICS

Simulates the non-dimensional budworm model with the Euler-Maruyama scheme,

    dx = [r*x*(1 - x/k) - x^2/(1 + x^2) + h] dt + sigma dW, w/ x >= 0,

and writes data.csv in the format app.py expects: a 'Condition' column, a 'Label' column & 1 column per time step.

The classification task is "how close is the system to bifurcation?":

    Label 1 = The control parameter r is NEAR the lower fold.
    Label 0 = r is FAR from bifurcation.

By default, only the approach side (r below the fold) is sampled, the textbook early-warning setup.
Otherwise, "far" could mean "already in the outbreak state", and the population level alone would give the label away.

Every row is one independent noisy run. app.py cuts each condition into blocks
of 50 consecutive rows and needs both labels in every block, so rows are written
with the labels alternating.

Importing this file does no work. Only generate_csv simulates anything.
'''

from __future__ import annotations

import numpy as np
import pandas as pd

TITLE = "Budworm"
DESCRIPTION = (
    "Budworm population model near a saddle-node (fold) bifurcation. Conditions cover a "
    "perfect and an imperfect bifurcation, noise that grows near the fold, and a "
    "parameter that drifts during the recording. Label 1 = close to the fold."
)

# ---- model ------------------------------------------------------------------
K = 10.0            # carrying capacity (non-dimensional)
DT = 0.1            # Euler-Maruyama step
BURN = 500          # steps discarded so the start-up transient is gone
LENGTH = 200        # samples recorded per row (short on purpose)
SIGMA = 0.3         # noise amplitude far from the fold
SIGMA_MAX = 1.0     # noise amplitude at the fold, "Variable noise" condition
SIGMA_WIDTH = 0.1   # how quickly the extra noise falls off away from the fold

# ---- labelling --------------------------------------------------------------
R_MIN, R_MAX = 0.1, 0.9   # range of r that is sampled
NEAR = 0.04               # within NEAR of r_crit  -> Label 1
FAR = 0.15                # at least FAR from r_crit (and from every fold) -> Label 0
FOLD = 0                  # which fold is "the" bifurcation: 0 = lower, 1 = upper
APPROACH_ONLY = True      # True: sample r <= r_crit only. False: both sides.

# ---- dataset ----------------------------------------------------------------
ZNORMALISE = False        # True: rescale every row to mean 0, std 1 (removes level and
                          # amplitude cues, leaving only the shape of the fluctuations)
ROWS_PER_CONDITION = 200  # four blocks of 50 for app.py
SEED = 54

# name -> imperfection h, drift rate of r (per unit time), noise grows near fold
CONDITIONS = {
    "Perfect bifurcation (h=0)":      dict(h=0.0,  rate=0.0,  variable_noise=False),
    "Imperfect bifurcation (h=0.05)": dict(h=0.05, rate=0.0,  variable_noise=False),
    "Variable noise (h=0)":           dict(h=0.0,  rate=0.0,  variable_noise=True),
    "Fast sweep (h=0)":               dict(h=0.0,  rate=5e-3, variable_noise=False),
}


def drift(x, r, h):
    return r * x * (1 - x / K) - x**2 / (1 + x**2) + h


def fold_locations(h):
    '''Values of r at which fixed points are created or destroyed, lowest first.

    Fixed points satisfy r = (x^2/(1+x^2) - h) / (x (1 - x/K)); a fold is a local
    extremum of that curve.
    '''
    x = np.linspace(0.01, K - 0.01, 200_000)
    r = (x**2 / (1 + x**2) - h) / (x * (1 - x / K))
    step = np.diff(r)
    turning = np.where(np.sign(step[1:]) != np.sign(step[:-1]))[0] + 1
    return sorted(float(r[i]) for i in turning)


def noise_amplitude(r, r_crit):
    '''sigma_base far from the fold, rising smoothly to SIGMA_MAX at it.'''
    return SIGMA + (SIGMA_MAX - SIGMA) * np.exp(-(np.abs(r - r_crit) / SIGMA_WIDTH) ** 2)


def simulate(r_mid, h, rate, r_crit, variable_noise, rng):
    '''One run per entry of ``r_mid``; returns an array (len(r_mid), LENGTH).

    r is held at its starting value during the burn-in, then drifts at ``rate``
    over the recorded window, chosen so r equals ``r_mid`` halfway through it.
    '''
    n = len(r_mid)
    r = r_mid - rate * LENGTH * DT / 2
    x = np.zeros(n)                       # x(0) = 0, as in the notebook
    out = np.empty((n, LENGTH))
    for step in range(BURN + LENGTH):
        sigma = noise_amplitude(r, r_crit) if variable_noise else SIGMA
        x = x + drift(x, r, h) * DT + sigma * rng.normal(0.0, np.sqrt(DT), n)
        x = np.maximum(x, 0.0)            # a population cannot be negative
        if step >= BURN:
            out[:, step - BURN] = x
            r = r + rate * DT
    return out


def sample_parameters(n_each, r_crit, folds, rng):
    '''n_each values of r near the fold and n_each far from every fold.'''
    near = rng.uniform(r_crit - NEAR, r_crit + (0.0 if APPROACH_ONLY else NEAR), n_each)
    far = np.empty(0)
    while len(far) < n_each:               # rejection sampling
        r = rng.uniform(R_MIN, r_crit if APPROACH_ONLY else R_MAX, 4 * n_each)
        keep = np.all([np.abs(r - fold) >= FAR for fold in folds], axis=0)
        far = np.concatenate([far, r[keep]])
    return near, far[:n_each]


def build_frame():
    rng = np.random.default_rng(SEED)
    n_each = ROWS_PER_CONDITION // 2
    frames = []
    
    for name, cfg in CONDITIONS.items():
        folds = fold_locations(cfg["h"])
        r_crit = folds[FOLD]
        near, far = sample_parameters(n_each, r_crit, folds, rng)

        # Alternate far, near, far, near ... so every block of 50 rows holds both labels.
        r_mid = np.empty(2 * n_each)
        r_mid[0::2], r_mid[1::2] = far, near
        labels = np.tile([0, 1], n_each)

        series = simulate(r_mid, cfg["h"], cfg["rate"], r_crit, cfg["variable_noise"], rng)
        frame = pd.DataFrame(series, columns=[f"t{i}" for i in range(LENGTH)])
        frame.insert(0, "Label", labels)
        frame.insert(0, "Condition", name)
        frames.append(frame)
    
  return pd.concat(frames, ignore_index=True)


def generate_csv(csv_path):
    '''Write the module's data.csv. Called by app.py when the file is missing or stale.'''
    frame = build_frame()
    
    if ZNORMALISE:
        cols = [c for c in frame.columns if c not in ("Condition", "Label")]
        v = frame[cols].to_numpy()
        frame[cols] = (v - v.mean(axis=1, keepdims=True)) / v.std(axis=1, keepdims=True)
    values = frame.drop(columns = ["Condition", "Label"]).to_numpy()
    
  if not np.isfinite(values).all():
        raise ValueError("The simulation produced NaN or infinite values.")
    frame.to_csv(csv_path, index=False, float_format="%.6g")


if __name__ == "__main__":
    from pathlib import Path
    generate_csv(Path(__file__).with_name("data.csv"))
    print("wrote", Path(__file__).with_name("data.csv"))
