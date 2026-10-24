"""Logistic map: chaos vs matched fake as the sampling stride grows, plus an observation-noise test.

For each setting: 10 datasets of 50 series (25 real, 25 IAAFT fakes, all from different orbits),
the app's protocol (one 70/30 split per dataset, random_state = dataset index).
Methods: Rocket, MiniRocket, catch22 (ridge), Dynamics (NN) (random forest on nearest-neighbour
one-step errors) and an ORACLE that knows the rule x -> r x (1 - x): it fits the unknown window
mean/std and r and measures how badly consecutive points break the rule (one number per series),
then a threshold on that number is learned on the training split.
"""
import sys, warnings, json, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.optimize import least_squares
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sktime.transformations.panel.rocket import Rocket, MiniRocket
import pycatch22

R_VALUES = (3.90, 3.95, 3.99)
N_DATASETS, SAMPLES, LENGTH, SEARCH, CANDIDATES, BURN = 10, 50, 100, 150, 10, 400

def lyap(r, n=200000):
    x, s = 0.3, 0.0
    for _ in range(1000): x = r * x * (1 - x)
    for _ in range(n):
        x = r * x * (1 - x); s += np.log(abs(r * (1 - 2 * x)))
    return s / n
LAM = float(np.mean([lyap(r) for r in R_VALUES]))   # per map step

def orbit(rng, r, stride):
    x = rng.uniform(0.05, 0.95)
    for _ in range(BURN): x = r * x * (1 - x)
    out = np.empty(SEARCH + stride * (LENGTH + 2))
    for i in range(len(out)):
        x = r * x * (1 - x); out[i] = x
    return out

def window(o, stride, noise, rng):
    best, start = np.inf, 0
    for s in range(SEARCH):
        w = o[s:s + stride * (LENGTH + 2):stride]
        mm = (w[0] - w[LENGTH]) ** 2 + (w[1] - w[LENGTH + 1]) ** 2
        if mm < best: best, start = mm, s
    w = o[start:start + stride * LENGTH:stride].copy()
    if noise: w = w + noise * w.std() * rng.standard_normal(LENGTH)
    return (w - w.mean()) / w.std()

def iaaft(x, rng):
    sx = np.sort(x); amp = np.abs(np.fft.fft(x)); s = rng.permutation(x); prev = None
    for _ in range(2000):
        f = np.real(np.fft.ifft(amp * np.exp(1j * np.angle(np.fft.fft(s)))))
        ranks = np.argsort(np.argsort(f)); s = sx[ranks]
        if prev is not None and np.array_equal(ranks, prev): break
        prev = ranks
    t, g = np.abs(np.fft.rfft(x)), np.abs(np.fft.rfft(s))
    return s, np.linalg.norm(g - t) / np.linalg.norm(t)

def blocks(stride, noise, seed):
    rng = np.random.default_rng(seed)
    for _ in range(N_DATASETS):
        w = [window(orbit(rng, float(rng.choice(R_VALUES)), stride), stride, noise, rng) for _ in range(SAMPLES)]
        fake = [min((iaaft(v, rng) for _ in range(CANDIDATES)), key=lambda c: c[1])[0] for v in w[25:]]
        yield np.array(w[:25] + fake), np.array([1] * 25 + [0] * 25)

# ---- oracle: knows the rule, fits unknown mean m, spread sd and r --------------------------
MS, SDS = np.linspace(0.30, 0.90, 31), np.linspace(0.10, 0.42, 25)
GM, GS = np.meshgrid(MS, SDS, indexing="ij"); GM, GS = GM.ravel(), GS.ravel()
def F(x, r, k):
    for _ in range(k): x = r * x * (1 - x)
    return x
def oracle_score(z, stride):
    best = np.inf
    for r in R_VALUES:
        X = GM[:, None] + GS[:, None] * z[None, :]
        err = np.mean((X[:, 1:] - F(X[:, :-1], r, stride)) ** 2, axis=1)
        for j in np.argsort(err)[:5]:
            fit = least_squares(lambda p: (p[0] + p[1] * z[1:]) - F(p[0] + p[1] * z[:-1], r, stride),
                                (GM[j], GS[j]), bounds=([0.1, 0.02], [1.0, 0.7]))
            best = min(best, np.sqrt(np.mean(fit.fun ** 2)))
    return best

def nn_err(x, d):
    n = len(x); V = np.stack([x[d-1-i:n-1-i] for i in range(d)], 1); T = x[d:]
    D = np.linalg.norm(V[:, None, :] - V[None, :, :], axis=2); np.fill_diagonal(D, np.inf)
    return np.mean(np.abs(T - T[D.argmin(1)]))

def ridge(): return make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))

def evaluate(stride, noise, seed):
    rows = []
    for i, (X, y) in enumerate(blocks(stride, noise, seed)):
        feats = {
            "Rocket": (np.asarray(Rocket(num_kernels=500, random_state=42).fit_transform(X[:, None, :])), ridge),
            "MiniRocket": (np.asarray(MiniRocket(random_state=42).fit_transform(X[:, None, :])), ridge),
            "catch22": (np.nan_to_num(np.array([pycatch22.catch22_all(list(r))["values"] for r in X])), ridge),
            "Dynamics (NN)": (np.array([[nn_err(r, 1), nn_err(r, 2), nn_err(r, 3)] for r in X]),
                              lambda: RandomForestClassifier(200, random_state=0)),
            "Oracle": (np.log10(np.array([[oracle_score(r, stride)] for r in X]) + 1e-12),
                       lambda: DecisionTreeClassifier(max_depth=1)),
        }
        tr, te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=i)
        row = dict(stride=stride, noise=noise, dataset=i)
        for name, (Fx, mk) in feats.items():
            row[name] = float((mk().fit(Fx[tr], y[tr]).predict(Fx[te]) == y[te]).mean())
            row["Chance"] = float(max(y[te].mean(), 1 - y[te].mean()))
        rows.append(row)
    return rows

if __name__ == "__main__":
    part = sys.argv[1]
    rows = []
    if part == "curve":
        for stride in (1, 2, 3, 4, 6, 8):
            rows += evaluate(stride, 0.0, 100 + stride); print("stride", stride, flush=True)
    else:
        for stride in (1, 4):
            for noise in (0.01, 0.05, 0.10):
                rows += evaluate(stride, noise, 200 + stride + int(noise * 100)); print("noise", stride, noise, flush=True)
    df = pd.DataFrame(rows); df["lam_dt"] = LAM * df.stride
    df.to_csv(f"logistic_{part}.csv", index=False)
    print("LAM per step", LAM)
    print(df.groupby(["stride", "noise"]).mean(numeric_only=True).drop(columns="dataset").round(3))
