"""Does the plateau-at-the-floor result hold across sub-regimes of the sample?

The sample is one shower-size slice, but it splits into two size halves (the two simulation
files, log10 Ne in [4.8, 5.0) and [5.0, 5.2]) and spans zenith 0-24.6 deg. For each regime
(two size halves; three zenith bins of equal event count) this script trains ridge, random
forest, gradient boosting and an MLP on the training-complement events in that regime, scores
them on the matched test events in that regime, and estimates the regime's own nine-feature
floor with the Gamma test. It reports sigma / floor for every estimator.

Extends results.json.  Run:  python analysis/run_regimes.py
"""
import json, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
from joblib import Parallel, delayed, parallel_config
from sklearn.ensemble import HistGradientBoostingRegressor as GBR, RandomForestRegressor as RF
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results.json')
R = json.load(open(OUT))
NJOBS = int(os.environ.get('NJOBS', '8'))

d = C.load_root(); c = C.load_csv(); y = d['y']; X = C.build_features(d)
r_idx, _ = C.match(d, c)
test = np.zeros(len(y), bool); test[r_idx] = True
lne, th = d['log_Ne'], d['Theta1']
zq = np.quantile(th, [1 / 3, 2 / 3])
REG = {'logNe 4.8-5.0': lne < 5.0, 'logNe 5.0-5.2': lne >= 5.0,
       'zenith low': th < zq[0], 'zenith mid': (th >= zq[0]) & (th < zq[1]), 'zenith high': th >= zq[1]}
MODELS = {'Ridge': lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
          'RandomForest': lambda: RF(n_estimators=60, min_samples_leaf=25, max_depth=18, n_jobs=1, random_state=42),
          'GradBoost': lambda: GBR(max_iter=600, learning_rate=.06, random_state=42),
          'MLP': lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=120,
                                                                      random_state=42, early_stopping=True))}


def gamma_floor(Xr, yr, K=10):
    Z = StandardScaler().fit_transform(Xr)
    dist, idx = NearestNeighbors(n_neighbors=K + 1).fit(Z).kneighbors(Z)
    g = 0.5 * ((yr[:, None] - yr[idx[:, 1:]]) ** 2).mean(0); dl = (dist[:, 1:] ** 2).mean(0)
    return float(np.sqrt(max(np.polyfit(dl, g, 1)[1], 0)))


def run(name, mask, mname):
    tr = np.where(mask & ~test)[0]; te = np.where(mask & test)[0]
    p = MODELS[mname]().fit(X[tr], y[tr]).predict(X[te])
    return name, mname, C.sigma(p, y[te]), len(tr), len(te)


with parallel_config(backend='loky', inner_max_num_threads=4):
    res = Parallel(n_jobs=NJOBS)(delayed(run)(n, m, k) for n, m in REG.items() for k in MODELS)
out = {}
for name, mask in REG.items():
    fl = gamma_floor(X[mask], y[mask])
    rows = {k: round(s, 4) for n, k, s, _, _ in res if n == name}
    ntr, nte = [(a, b) for n, _, _, a, b in res if n == name][0]
    nonlin = [rows[k] for k in ('RandomForest', 'GradBoost', 'MLP')]
    out[name] = {'n_train': ntr, 'n_test': nte, 'floor': round(fl, 4), 'sigma': rows,
                 'best_over_floor_pct': round(100 * (min(nonlin) - fl) / fl, 1),
                 'nonlin_spread': round(max(nonlin) - min(nonlin), 4)}
    print('  %-14s n_test=%5d floor %.4f | %s | best/floor %+.1f%%, nonlinear spread %.4f'
          % (name, nte, fl, ' '.join('%s %.4f' % (k[:5], v) for k, v in rows.items()),
             out[name]['best_over_floor_pct'], out[name]['nonlin_spread']))
R['regimes'] = {'zenith_edges': [round(float(v), 2) for v in zq], 'rows': out}
json.dump(R, open(OUT, 'w'), indent=2)
print('extended %s' % OUT)
