"""Diagnostics behind the headline numbers.

1. Learning curves: test sigma of gradient boosting and the MLP on the nine features against
   training-set size (1e3 .. full complement), with the Gamma-test floor as reference.
2. Error against true energy: sigma and mean bias of each estimator in bins of true log10 E,
   with bootstrap 95% bands, so the global sigma cannot hide energy-dependent behaviour.
3. Per-species matrix: bias, sigma and conformal 68% / 90% coverage for each primary.
4. Coverage calibration curve: raw quantile regression and CQR at nominal 50-95%.
5. Two more estimators of different kinds: k-nearest-neighbour regression and XGBoost.
6. Leave-one-primary-out: train on four species, test on the fifth (supplementary).

Extends results.json.  Run:  python analysis/run_diagnostics.py
"""
import json, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
from joblib import Parallel, delayed, parallel_config
from sklearn.ensemble import HistGradientBoostingRegressor as GBR, RandomForestRegressor as RF
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results.json')
R = json.load(open(OUT))
NJOBS = int(os.environ.get('NJOBS', '8'))

d = C.load_root(); c = C.load_csv(); y = d['y']; X = C.build_features(d)
r_idx, c_idx = C.match(d, c)
test = np.zeros(len(y), bool); test[r_idx] = True
tr_all = np.where(~test)[0]; yt = y[r_idx]; pid = d['PrimaryID'][r_idx]
GB = lambda s=42, **kw: GBR(max_iter=600, learning_rate=.06, random_state=s, **kw)
MLP = lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=120,
                                                           random_state=42, early_stopping=True))

# ------------------------------------------------------------------ 1. learning curves
SIZES = [1000, 3000, 10000, 30000, 100000, len(tr_all)]


def lc(name, n, s):
    tr = np.random.default_rng(s).choice(tr_all, n, replace=False) if n < len(tr_all) else tr_all
    m = GB(s) if name == 'GradBoost' else MLP()
    return name, n, C.sigma(m.fit(X[tr], y[tr]).predict(X[r_idx]), yt)


with parallel_config(backend='loky', inner_max_num_threads=4):
    res = Parallel(n_jobs=NJOBS)(delayed(lc)(m, n, s) for m in ('GradBoost', 'MLP') for n in SIZES
                                 for s in (0, 1, 2))
curves = {m: {'n': SIZES, 'sigma': [round(float(np.mean([v for mm, nn, v in res if mm == m and nn == n])), 4)
                                    for n in SIZES],
              'sd': [round(float(np.std([v for mm, nn, v in res if mm == m and nn == n])), 4) for n in SIZES]}
          for m in ('GradBoost', 'MLP')}
print('learning curves:', {m: v['sigma'] for m, v in curves.items()})

# ------------------------------------------------------------------ 2. error vs energy
MODELS = {'Ridge': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
          'RandomForest': RF(n_estimators=60, min_samples_leaf=25, max_depth=18, n_jobs=NJOBS, random_state=42),
          'GradBoost': GB(), 'MLP': MLP()}
preds = {k: m.fit(X[tr_all], y[tr_all]).predict(X[r_idx]) for k, m in MODELS.items()}
preds['size-age calibration'] = c['nkg'][c_idx]; preds['archived CNN/ViT'] = c['cnnvit'][c_idx]
edges = np.quantile(yt, np.linspace(0, 1, 9))
centres = [round(float(v), 3) for v in 0.5 * (edges[1:] + edges[:-1])]
rng = np.random.default_rng(0)
vs_energy = {}
for k, p in preds.items():
    e = p - yt; rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (yt >= a) & (yt <= b); ee = e[m]
        bs = [ee[rng.integers(0, len(ee), len(ee))] for _ in range(300)]
        rows.append({'sigma': round(float(ee.std()), 4), 'bias': round(float(ee.mean()), 4),
                     'sigma_ci': [round(float(v), 4) for v in np.percentile([b_.std() for b_ in bs], [2.5, 97.5])],
                     'bias_ci': [round(float(v), 4) for v in np.percentile([b_.mean() for b_ in bs], [2.5, 97.5])]})
    vs_energy[k] = rows
print('vs energy computed for', list(vs_energy))

# ------------------------------------------------------------------ 3 + 4. CQR, per species
tr, cal = train_test_split(tr_all, test_size=.25, random_state=42)
NOM = [0.50, 0.60, 0.68, 0.80, 0.90, 0.95]


def cqr(nominal):
    lo_q, hi_q = (1 - nominal) / 2, 1 - (1 - nominal) / 2
    mlo = GB(loss='quantile', quantile=lo_q).fit(X[tr], y[tr])
    mhi = GB(loss='quantile', quantile=hi_q).fit(X[tr], y[tr])
    e = np.maximum(mlo.predict(X[cal]) - y[cal], y[cal] - mhi.predict(X[cal]))
    Q = float(np.quantile(e, min(1.0, nominal * (1 + 1 / len(cal)))))
    lo, hi = mlo.predict(X[r_idx]), mhi.predict(X[r_idx])
    raw = (yt >= lo) & (yt <= hi); conf = (yt >= lo - Q) & (yt <= hi + Q)
    return nominal, raw, conf


with parallel_config(backend='loky', inner_max_num_threads=4):
    cq = Parallel(n_jobs=min(NJOBS, len(NOM)))(delayed(cqr)(n) for n in NOM)
coverage = {'nominal': NOM, 'raw': [round(100 * float(r.mean()), 1) for _, r, _ in cq],
            'cqr': [round(100 * float(c_.mean()), 1) for _, _, c_ in cq]}
print('coverage curve:', coverage)
covmap = {n: (r, c_) for n, r, c_ in cq}
eg = preds['GradBoost'] - yt
species = {}
for code, nm in C.SPECIES.items():
    m = pid == code
    species[nm] = {'n': int(m.sum()), 'bias': round(float(eg[m].mean()), 4), 'sigma': round(float(eg[m].std()), 4),
                   'cov68': round(100 * float(covmap[0.68][1][m].mean()), 1),
                   'cov90': round(100 * float(covmap[0.90][1][m].mean()), 1)}
print('per species:', species)

# ------------------------------------------------------------------ 5. kNN and XGBoost
extra = {}
knn = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=50, weights='distance', n_jobs=NJOBS))
extra['kNN (k=50)'] = round(C.sigma(knn.fit(X[tr_all], y[tr_all]).predict(X[r_idx]), yt), 4)
try:
    import xgboost as xgb
    xm = xgb.XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=7, subsample=0.8,
                          colsample_bytree=0.9, n_jobs=NJOBS, random_state=42)
    extra['XGBoost'] = round(C.sigma(xm.fit(X[tr_all], y[tr_all]).predict(X[r_idx]), yt), 4)
except ImportError:
    print('  xgboost not installed: skipped')
print('extra estimators:', extra)

# ------------------------------------------------------------------ 6. leave-one-primary-out
lopo = {}
pid_all = d['PrimaryID']
for code, nm in C.SPECIES.items():
    trm = tr_all[pid_all[tr_all] != code]; tem = r_idx[pid_all[r_idx] == code]
    p_out = GB().fit(X[trm], y[trm]).predict(X[tem])
    p_in = GB().fit(X[tr_all], y[tr_all]).predict(X[tem])
    lopo[nm] = {'sigma_held_out': round(C.sigma(p_out, y[tem]), 4), 'bias_held_out': round(C.bias(p_out, y[tem]), 4),
                'sigma_in': round(C.sigma(p_in, y[tem]), 4), 'bias_in': round(C.bias(p_in, y[tem]), 4)}
print('leave-one-primary-out:', lopo)

R['diagnostics'] = {'learning_curves': curves, 'energy_centres': centres, 'vs_energy': vs_energy,
                    'coverage_curve': coverage, 'species': species, 'extra_estimators': extra, 'lopo': lopo}
json.dump(R, open(OUT, 'w'), indent=2)
print('extended %s' % OUT)
