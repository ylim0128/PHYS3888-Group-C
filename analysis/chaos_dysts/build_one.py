"""Build one dysts system's block: 25 real chaotic windows + 25 IAAFT fakes, fixed points per period.

Every system is sampled at PTS_PER_PERIOD points per dominant period, so how fast nearby
states separate per recorded point (lambda * dt) differs between systems: that is the axis.
"""
import sys, warnings, numpy as np
warnings.filterwarnings("ignore")
import dysts.flows as flows

PTS_PER_PERIOD = 20
LENGTH, SEARCH, GAP = 100, 40, 40
SAMPLES, CANDIDATES = 50, 10

def window(series):
    best, start = np.inf, 0
    for s in range(SEARCH):
        w = series[s:s + LENGTH + 2]
        mm = (w[0] - w[LENGTH]) ** 2 + (w[1] - w[LENGTH + 1]) ** 2
        if mm < best: best, start = mm, s
    w = series[start:start + LENGTH]
    return (w - w.mean()) / w.std()

def iaaft(x, rng):
    sx = np.sort(x); amp = np.abs(np.fft.fft(x)); s = rng.permutation(x).astype(float); prev = None
    for _ in range(2000):
        f = np.real(np.fft.ifft(amp * np.exp(1j * np.angle(np.fft.fft(s)))))
        ranks = np.argsort(np.argsort(f)); s = sx[ranks]
        if prev is not None and np.array_equal(ranks, prev): break
        prev = ranks
    t, g = np.abs(np.fft.rfft(x)), np.abs(np.fft.rfft(s))
    return s, np.linalg.norm(g - t) / np.linalg.norm(t)

def main(name, seed, lam_dt_target=None, outdir="data"):
    rng = np.random.default_rng(seed)
    model = getattr(flows, name)()
    lam, period = model.maximum_lyapunov_estimated, model.period
    gap = GAP if lam_dt_target is None else 5   # coarse samples decorrelate within a few points
    chunk = SEARCH + LENGTH + 2 + gap
    n = SAMPLES * chunk + 200
    from scipy.integrate import solve_ivp
    from dysts.base import DynSysDelay
    if isinstance(model, DynSysDelay):
        raise SystemExit("delay system skipped")
    dt = period / PTS_PER_PERIOD if lam_dt_target is None else lam_dt_target / lam
    ic = np.asarray(model.ic, float) * (1 + 1e-3 * rng.standard_normal(np.shape(model.ic)))
    sol = solve_ivp(lambda t, s: np.asarray(model.rhs(s, t), float), (0, n * dt), ic,
                    t_eval=np.arange(n) * dt, rtol=1e-6, atol=1e-6, method="DOP853")
    if sol.status != 0 or sol.y.shape[1] != n:
        raise SystemExit("integration failed")
    x = sol.y[0, 200:]
    if not np.all(np.isfinite(x)) or x.std() == 0:
        raise SystemExit("bad trajectory")
    wins = [window(x[i * chunk:(i + 1) * chunk]) for i in range(SAMPLES)]
    if any(not np.all(np.isfinite(w)) for w in wins):
        raise SystemExit("flat window")
    order = rng.permutation(SAMPLES)
    real = [wins[i] for i in order[:25]]
    fake = [min((iaaft(wins[i], rng) for _ in range(CANDIDATES)), key=lambda c: c[1])[0] for i in order[25:]]
    X = np.array(real + fake); y = np.array([1] * 25 + [0] * 25)
    np.savez(f"{outdir}/{name}.npz", X=X, y=y, lam=lam, period=period, lam_dt=lam * dt, lam_period=lam * period)
    print(name, "ok", round(lam * dt, 3), flush=True)

if __name__ == "__main__":
    t = float(sys.argv[3]) if len(sys.argv) > 3 else None
    main(sys.argv[1], int(sys.argv[2]), t, sys.argv[4] if len(sys.argv) > 4 else "data")
