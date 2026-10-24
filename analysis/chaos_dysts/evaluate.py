"""Score every dysts system: within-system (app protocol) and leave-one-system-out."""
import glob, os, warnings, time, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

import sys
DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results_per_system.csv"
files = sorted(glob.glob(f"{DATA}/*.npz"))
names, Xs, ys, gs, meta = [], [], [], [], []
for k, f in enumerate(files):
    d = np.load(f)
    names.append(os.path.basename(f)[:-4]); Xs.append(d["X"]); ys.append(d["y"]); gs.append(np.full(len(d["y"]), k))
    meta.append(dict(system=names[-1], lam=float(d["lam"]), period=float(d["period"]),
                     lam_dt=float(d["lam_dt"]), lam_period=float(d["lam_period"])))
X = np.vstack(Xs); y = np.concatenate(ys); g = np.concatenate(gs)
print(len(names), "systems,", X.shape)

def nn_err(x, d):
    n = len(x); V = np.stack([x[d-1-i:n-1-i] for i in range(d)], 1); T = x[d:]
    D = np.linalg.norm(V[:, None, :] - V[None, :, :], axis=2); np.fill_diagonal(D, np.inf)
    return np.mean(np.abs(T - T[D.argmin(1)]))

t = time.time()
F = {}
from sktime.transformations.panel.rocket import Rocket, MiniRocket
X3 = X[:, None, :]
F["Rocket"] = np.asarray(Rocket(num_kernels=500, random_state=42).fit_transform(X3))
F["MiniRocket"] = np.asarray(MiniRocket(random_state=42).fit_transform(X3))
import pycatch22
F["catch22"] = np.nan_to_num(np.array([pycatch22.catch22_all(list(r))["values"] for r in X]))
F["Dynamics (NN)"] = np.array([[nn_err(r, 1), nn_err(r, 2), nn_err(r, 3)] for r in X])
print("features", round(time.time() - t), "s")

def model(name):
    if name == "Dynamics (NN)":
        return RandomForestClassifier(200, random_state=0, n_jobs=2)
    return make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))

METHODS = list(F)
rows = []
for k, nm in enumerate(names):
    idx = np.where(g == k)[0]
    row = dict(meta[k])
    for m in METHODS:
        accs = []
        for i in range(10):
            tr, te = train_test_split(idx, test_size=0.3, random_state=i)
            accs.append((model(m).fit(F[m][tr], y[tr]).predict(F[m][te]) == y[te]).mean())
        row[f"within_{m}"] = np.mean(accs)
    rows.append(row)
    if k % 20 == 0: print(k, nm, round(time.time() - t), "s")
res = pd.DataFrame(rows)
# Unseen systems: 10 folds split BY SYSTEM; every system is tested once by a model that never saw it.
from sklearn.model_selection import GroupKFold
for m in METHODS:
    acc = np.zeros(len(names))
    for tr, te in GroupKFold(n_splits=10).split(X, y, g):
        pred = model(m).fit(F[m][tr], y[tr]).predict(F[m][te])
        for k in np.unique(g[te]):
            sel = g[te] == k
            acc[k] = (pred[sel] == y[te][sel]).mean()
    res[f"unseen_{m}"] = acc
    print("unseen", m, round(time.time() - t), "s")
res.to_csv(OUT, index=False)
print("done", round(time.time() - t), "s")
