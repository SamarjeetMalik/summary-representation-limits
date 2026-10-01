"""A model-free estimate of the Bayes risk of the summary representation.

Agreement between estimators is evidence of a plateau, not of a floor. This script
estimates the floor itself, E[Var(Y | X)], without fitting any regressor, using the
nearest-neighbour Gamma test: for k = 1..K,

    gamma(k) = (1/2n) sum_i (Y_i - Y_{N_k(i)})^2,   delta(k) = (1/n) sum_i |X_i - X_{N_k(i)}|^2,

and gamma(k) -> E[Var(Y|X)] as delta(k) -> 0 for continuous regression functions. The
intercept of a linear fit of gamma on delta is the estimate. The k = 1 value is also
reported: it is biased upward, so it is a conservative (too high) floor.

Features are standardised; the estimate is not invariant to that choice, which is
one reason it is reported as an estimate, not a bound.

CONTROL FIRST: the estimator is run on synthetic targets built on the real feature
matrix with a known noise level before any number on the real target is used. If it
misses a known noise level by more than 10%, the script stops.

Extends results.json. Run:  python analysis/run_bayes.py
"""
import json, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results.json')
R = json.load(open(OUT))
K = 10
N_BOOT = 200


def gamma_test(X, y, k=K):
    """Return (intercept, gamma(1)) of the Gamma test for target y on features X."""
    Z = StandardScaler().fit_transform(X)
    dist, idx = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)
    delta = (dist[:, 1:] ** 2).mean(0)
    g = 0.5 * ((y[:, None] - y[idx[:, 1:]]) ** 2).mean(0)
    slope, icpt = np.polyfit(delta, g, 1)
    return float(icpt), float(g[0])


print('loading...')
d = C.load_root()
X9 = C.build_features(d)
X4 = np.column_stack([d['log_Ne'], d['Age'], d['NKGX'], d['NKGY']])
y = d['y']

# ------------------------------------------------------------------ control
print('\n== control: known noise on the real feature matrix ==')
rng = np.random.default_rng(0)
Z9 = StandardScaler().fit_transform(X9)
m_true = (0.9 * Z9[:, 0] + 0.35 * Z9[:, 1] + 0.08 * np.tanh(2 * Z9[:, 2])
          + 0.05 * Z9[:, 0] * Z9[:, 1])
m_true = (m_true - m_true.mean()) / m_true.std() * 0.17            # target-like scale
controls = []
for s_noise in (0.05, 0.10, 0.12):
    ys = m_true + rng.normal(0, s_noise, len(y))
    res = gamma_test(X9, ys)
    est = np.sqrt(max(res[0], 0))
    controls.append({'true_sd': s_noise, 'est_sd': round(est, 4),
                     'gamma1_sd': round(float(np.sqrt(res[1])), 4)})
    print('  true %.3f  estimated %.4f  (k=1: %.4f)' % (s_noise, est, np.sqrt(res[1])))
    assert abs(est - s_noise) / s_noise < 0.10, 'Gamma test fails its control; do not use it'

# ------------------------------------------------------------------ real target
print('\n== Bayes-risk estimate on the real target ==')
out = {}
for key, X in (('nine', X9), ('four', X4)):
    Z = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=K + 1).fit(Z)
    dist, idx = nn.kneighbors(Z)
    d2 = dist[:, 1:] ** 2
    g = 0.5 * (y[:, None] - y[idx[:, 1:]]) ** 2
    def est(rows):
        slope, icpt = np.polyfit(d2[rows].mean(0), g[rows].mean(0), 1)
        return icpt
    point = est(slice(None))
    bs = [est(rng.integers(0, len(y), len(y))) for _ in range(N_BOOT)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    out[key] = {'rms_floor': round(float(np.sqrt(max(point, 0))), 4),
                'ci': [round(float(np.sqrt(max(lo, 0))), 4), round(float(np.sqrt(max(hi, 0))), 4)],
                'gamma1_rms': round(float(np.sqrt(g[:, 0].mean())), 4)}
    print('  %-5s floor RMS %.4f  95%% CI [%.4f, %.4f]   (k=1 conservative: %.4f)'
          % (key, out[key]['rms_floor'], *out[key]['ci'], out[key]['gamma1_rms']))

R['bayes'] = {'method': 'Gamma test, k=1..%d, standardised features, %d bootstrap' % (K, N_BOOT),
              'controls': controls, **out}
json.dump(R, open(OUT, 'w'), indent=2)
print('\nextended %s' % OUT)
