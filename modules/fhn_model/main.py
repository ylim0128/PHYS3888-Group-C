'''FitzHugh-Nagumo Neuron Model

Simulates the noisy FitzHugh-Nagumo model with the Euler-Maruyama scheme,

    dv = (v - v^3/3 - w + I) dt + sigma dW      (membrane potential, fast)
    dw = eps (v + a - b w) dt                   (recovery variable, slow)

and writes data.csv in the format app.py expects:
A 'Condition' column, a 'Label' column (0 or 1) & 1 column per time step. Only v is recorded.

The classification task is "how close is the system to its Hopf bifurcation?":

    Label 1 = the input current I is within NEAR of the lower Hopf point.
    Label 0 = I is at least FAR away from it.

By default, only the approach side (I below the Hopf point, where the neuron is
quiescent) is sampled, the textbook early-warning setup. Otherwise "far" could
mean "already spiking", and the signal amplitude alone would give the label away.

Every row is one independent noisy run. app.py cuts each condition into blocks
of 50 consecutive rows and needs both labels in every block, so rows are written
with the labels alternating.

Importing this file does no work. Only generate_csv simulates anything.
'''

from __future__ import annotations

import numpy as np
import pandas as pd

TITLE = "Neuron"
DESCRIPTION = (
    "FitzHugh-Nagumo neuron driven across a Hopf bifurcation from quiescence to "
    "spiking. Conditions cover a fixed input current, noise that grows near the "
    "bifurcation, and a current that ramps during the recording. Label 1 = close to "
    "the bifurcation."
)

# ---- model ------------------------------------------------------------------
A, B, EPS = 0.7, 0.8, 0.08
DT = 0.05             # Euler-Maruyama step
RECORD_EVERY = 5      # keep every 5th step, so a row spans 50 time units
BURN = 1000           # steps discarded so the start-up transient is gone
LENGTH = 200          # samples recorded per row (short on purpose)
SIGMA = 0.03          # noise amplitude far from the bifurcation
SIGMA_MAX = 0.1       # noise amplitude at the bifurcation, "Variable noise" condition
SIGMA_WIDTH = 0.05    # how quickly the extra noise falls off away from it
V0, W0 = -1.2, -0.6   # starting point, as in the notebook

# ---- labelling --------------------------------------------------------------
I_MIN, I_MAX = 0.1, 0.6   # range of I that is sampled
NEAR = 0.04               # within NEAR of I_hopf -> Label 1
FAR = 0.07                # at least FAR from I_hopf -> Label 0
APPROACH_ONLY = True      # True: sample I <= I_hopf only. False: both sides.

# ---- dataset ----------------------------------------------------------------
ZNORMALISE = False        # True: rescale every row to mean 0, std 1 (removes level and
                          # amplitude cues, leaving only the shape of the fluctuations)
ROWS_PER_CONDITION = 200  # four blocks of 50 for app.py
SEED = 54

# name -> drift rate of I (per unit time), noise grows near the bifurcation
CONDITIONS = {
    "Fixed current":  dict(rate=0.0,  variable_noise=False),
    "Variable noise": dict(rate=0.0,  variable_noise=True),
    "Fast sweep":     dict(rate=2e-3, variable_noise=False),
}


def hopf_current():
    '''Lower Hopf point: the trace of the Jacobian, 1 - v^2 - eps b, is zero at the
    equilibrium, so v* = -sqrt(1 - eps b), and I follows from the v-nullcline.'''
    v = -np.sqrt(1 - EPS * B)
    return float(-v + v**3 / 3 + (v + A) / B)


I_HOPF = hopf_current()   # 0.3313


def noise_amplitude(current):
    """sigma_base far from the bifurcation, rising smoothly to SIGMA_MAX at it."""
    return SIGMA + (SIGMA_MAX - SIGMA) * np.exp(-(np.abs(current - I_HOPF) / SIGMA_WIDTH) ** 2)


def simulate(i_mid, rate, variable_noise, rng):
    '''One run per entry of ``i_mid``; returns an array (len(i_mid), LENGTH) of v.

    I is held at its starting value during the burn-in, then ramps at ``rate``
    over the recorded window, chosen so I equals ``i_mid`` halfway through it.
    '''
    n = len(i_mid)
    window_time = LENGTH * RECORD_EVERY * DT
    current = i_mid - rate * window_time / 2
    v = np.full(n, V0)
    w = np.full(n, W0)
    out = np.empty((n, LENGTH))
    
    for step in range(BURN + LENGTH * RECORD_EVERY):
        sigma = noise_amplitude(current) if variable_noise else SIGMA
        dv = v - v**3 / 3 - w + current
        dw = EPS * (v + A - B * w)
        v = v + dv * DT + sigma * rng.normal(0.0, np.sqrt(DT), n)
        w = w + dw * DT
        
        if step >= BURN:
            k = step - BURN
            if k % RECORD_EVERY == RECORD_EVERY - 1:
                out[:, k // RECORD_EVERY] = v
            current = current + rate * DT
    
    return out


def sample_parameters(n_each, rng):
    '''n_each currents near the Hopf point and n_each far from it.'''
    near = rng.uniform(I_HOPF - NEAR, I_HOPF + (0.0 if APPROACH_ONLY else NEAR), n_each)
    far = np.empty(0)
    
    while len(far) < n_each:               # rejection sampling
        i = rng.uniform(I_MIN, I_HOPF if APPROACH_ONLY else I_MAX, 4 * n_each)
        far = np.concatenate([far, i[np.abs(i - I_HOPF) >= FAR]])
    
    return near, far[:n_each]


def build_frame():
    rng = np.random.default_rng(SEED)
    n_each = ROWS_PER_CONDITION // 2
    frames = []
    
    for name, cfg in CONDITIONS.items():
        near, far = sample_parameters(n_each, rng)

        # Alternate far, near, far, near ... so every block of 50 rows holds both labels.
        i_mid = np.empty(2 * n_each)
        i_mid[0::2], i_mid[1::2] = far, near
        labels = np.tile([0, 1], n_each)

        series = simulate(i_mid, cfg["rate"], cfg["variable_noise"], rng)
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
    values = frame.drop(columns=["Condition", "Label"]).to_numpy()
    
    if not np.isfinite(values).all():
        raise ValueError("The simulation produced NaN or infinite values.")
    frame.to_csv(csv_path, index=False, float_format="%.6g")


if __name__ == "__main__":
    from pathlib import Path
    generate_csv(Path(__file__).with_name("data.csv"))
    print("wrote", Path(__file__).with_name("data.csv"))
