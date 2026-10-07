"""Checks added for the revision. No new model class, no new data.

  1. Population match. run_bayes.py estimates the Gamma-test floor on all 171,776 events,
     while every estimator is scored on the 34,356 matched test events. Here the floor is
     re-estimated with the TEST events as query points (neighbours from the whole sample,
     same-shower copies excluded) and, separately, inside the test set alone.
  2. Interval. The published interval resamples events; events come 3.6 per shower. Here the
     interval resamples showers, and the excess of gradient boosting over the floor is
     bootstrapped jointly (same resampled showers for the model's errors and for the floor).
  3. Redundant coordinates. The nine features contain cos(theta), R_core and the muon ratio,
     which are functions of the others and change the neighbour metric. The floor is repeated
     on the six non-redundant coordinates.
  4. sigma versus RMSE. Proposition 1 is about mean squared error; the paper reports the
     standard deviation of the residual. Both are recorded for gradient boosting.
  5. Units of the KASCADE "gap closed" figures: in sigma (as published) and in variance.

CONTROL FIRST, as in run_bayes.py: each floor variant must recover a known noise level on the
real feature matrix before its value on the real target is recorded.

Extends results.json.  Run:  python analysis/run_tmlr_checks.py
"""
import json, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as GBR
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results.json')
R = json.load(open(OUT))
K = 10
KQ = 40                     # neighbours fetched, so that K remain after dropping same-shower copies
N_BOOT = 1000
rng = np.random.default_rng(0)

d = C.load_root(); c = C.load_csv()
y = d['y']; X9 = C.build_features(d)
X6 = np.column_stack([d['log_Ne'], d['Age'], d['log_Nmu'], d['Theta1'], d['NKGX'], d['NKGY']])
r_idx, _ = C.match(d, c)
test = np.zeros(len(y), bool); test[r_idx] = True
_, grp = np.unique(np.column_stack([d['PrimaryEnergy'], d['PrimaryID'].astype(float)]),
                   axis=0, return_inverse=True)
grp = grp.ravel()


def neighbours(X, query, pool):
    """Squared distances and pool indices of the K nearest other-shower neighbours of each query."""
    sc = StandardScaler().fit(X)
    Zp, Zq = sc.transform(X[pool]), sc.transform(X[query])
    dist, idx = NearestNeighbors(n_neighbors=KQ).fit(Zp).kneighbors(Zq)
    idx = pool[idx]
    other = grp[idx] != grp[query][:, None]
    rank = np.where(other, np.arange(KQ)[None, :], 99)
    order = np.sort(rank, axis=1)[:, :K]
    assert (order < 99).all(), 'fewer than K other-shower neighbours for some events'
    return np.take_along_axis(dist, order, 1) ** 2, np.take_along_axis(idx, order, 1)


def intercept(d2, g2, rows=slice(None)):
    return np.polyfit(d2[rows].mean(0), g2[rows].mean(0), 1)[1]


def rms(v):
    return float(np.sqrt(max(v, 0)))


all_idx = np.arange(len(y))
variants = {'test_queries_all_neighbours': (X9, r_idx, all_idx),
            'test_only': (X9, r_idx, r_idx),
            'all_events': (X9, all_idx, all_idx),
            'six_coordinates_test_queries': (X6, r_idx, all_idx)}
NB = {k: neighbours(*v) for k, v in variants.items()}

# ------------------------------------------------------------------ control
print('== control: known noise on the real feature matrix ==')
Z9 = StandardScaler().fit_transform(X9)
m_true = (0.9 * Z9[:, 0] + 0.35 * Z9[:, 1] + 0.08 * np.tanh(2 * Z9[:, 2]) + 0.05 * Z9[:, 0] * Z9[:, 1])
m_true = (m_true - m_true.mean()) / m_true.std() * 0.17
controls = {}
for name, (Xv, q, pool) in variants.items():
    d2, I = NB[name]
    row = []
    for s_noise in (0.05, 0.10, 0.12):
        ys = m_true + rng.normal(0, s_noise, len(y))
        est = rms(intercept(d2, 0.5 * (ys[q][:, None] - ys[I]) ** 2))
        row.append({'true_sd': s_noise, 'est_sd': round(est, 4)})
        assert abs(est - s_noise) / s_noise < 0.10, 'control failed for %s; do not use it' % name
    controls[name] = row
    print('  %-30s %s' % (name, ['%.4f' % r['est_sd'] for r in row]))

# ------------------------------------------------------------------ gradient boosting, sigma and RMSE
p = GBR(max_iter=600, learning_rate=.06, random_state=42).fit(X9[C.train_mask(d, test)], y[C.train_mask(d, test)]).predict(X9[r_idx])
e = p - y[r_idx]
out = {'controls': controls,
       'gb': {'sigma': round(float(e.std()), 4), 'rmse': round(float(np.sqrt((e ** 2).mean())), 4),
              'bias': round(float(e.mean()), 5)}}
print('GB on test: sigma %.4f  rmse %.4f  bias %+.5f' % (e.std(), np.sqrt((e ** 2).mean()), e.mean()))

# ------------------------------------------------------------------ floors, shower-level bootstrap
tg = grp[r_idx]
ug, inv = np.unique(tg, return_inverse=True)
members = [np.where(inv == i)[0] for i in range(len(ug))]


def shower_rows():
    return np.concatenate([members[i] for i in rng.integers(0, len(ug), len(ug))])


floors = {}
for name, (Xv, q, pool) in variants.items():
    d2, I = NB[name]
    g2 = 0.5 * (y[q][:, None] - y[I]) ** 2
    point = rms(intercept(d2, g2))
    rec = {'floor': round(point, 4), 'gamma1': round(rms(g2[:, 0].mean()), 4)}
    if q is r_idx:                          # joint bootstrap with the model's errors, by shower
        bf, bx = [], []
        for _ in range(N_BOOT):
            rows = shower_rows()
            f = rms(intercept(d2, g2, rows))
            bf.append(f); bx.append(100 * (e[rows].std() - f) / f)
        rec['ci_shower'] = [round(float(v), 4) for v in np.percentile(bf, [2.5, 97.5])]
        be = [rms(intercept(d2, g2, rng.integers(0, len(q), len(q)))) for _ in range(N_BOOT)]
        rec['ci_event'] = [round(float(v), 4) for v in np.percentile(be, [2.5, 97.5])]
        rec['gb_excess_pct'] = round(100 * (e.std() - point) / point, 2)
        rec['gb_excess_ci_shower'] = [round(float(v), 2) for v in np.percentile(bx, [2.5, 97.5])]
    floors[name] = rec
    print('  %-30s %s' % (name, rec))
out['floors'] = floors

# ------------------------------------------------------------------ KASCADE gap fractions, both units
kf = os.path.join(HERE, 'kcdc', 'kcdc_floor_results.json')
kc = os.path.join(HERE, 'kcdc', 'kcdc_cnn_results.json')
if os.path.exists(kf) and os.path.exists(kc):
    KF = json.load(open(kf)); KC = json.load(open(kc))
    s9, so = KF['oracle']['sigma_nine'], KF['oracle']['sigma_true_sizes']
    rows = {'engineered (all station features)': KF['intermediate']['profiles + counts, totals, timing']['sigma'],
            'CNN, measured image only': float(np.mean(KC['cnn']['image only']['sigma_seeds'])),
            'CNN residual on GB(nine)': float(np.mean(KC['cnn']['residual on GB(nine)']['sigma_seeds']))}
    out['kascade_gap'] = {'oracle_share_of_variance_pct': round(100 * (1 - so ** 2 / s9 ** 2), 1),
                          'oracle_share_of_sigma_pct': round(100 * (1 - so / s9), 1),
                          'closed': {k: {'sigma_pct': round(100 * (s9 - v) / (s9 - so), 1),
                                         'variance_pct': round(100 * (s9 ** 2 - v ** 2) / (s9 ** 2 - so ** 2), 1)}
                                     for k, v in rows.items()}}
    print('KASCADE gap:', out['kascade_gap'])

R['tmlr_checks'] = out
json.dump(R, open(OUT, 'w'), indent=2)
print('extended %s' % OUT)
