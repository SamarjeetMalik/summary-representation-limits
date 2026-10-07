"""Two uncertainty checks requested in pre-submission review.

1. Groupwise conformal coverage: CQR intervals (as in run_diagnostics.py) at nominal 50-95%,
   coverage reported per primary species. The CQR guarantee is marginal; this shows how far each
   subgroup departs from nominal.
2. Regime excess with intervals: for each of the five sub-regimes of run_regimes.py, the excess of
   gradient boosting over that regime's Gamma-test floor estimate, (sigma / floor - 1), with a 95%
   bootstrap interval that resamples both the test events (sigma) and the regime's events (floor,
   neighbour graph held fixed as in run_bayes.py).

Extends results.json.  Run:  python analysis/run_group_checks.py
"""
import json, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
from joblib import Parallel, delayed, parallel_config
from sklearn.ensemble import HistGradientBoostingRegressor as GBR
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results.json')
R = json.load(open(OUT))
NJOBS = int(os.environ.get('NJOBS', '8')); NB = 400

d = C.load_root(); c = C.load_csv(); y = d['y']; X = C.build_features(d)
r_idx, _ = C.match(d, c)
test = np.zeros(len(y), bool); test[r_idx] = True
tr_all = np.where(C.train_mask(d, test))[0]; yt = y[r_idx]; pid = d['PrimaryID'][r_idx]
GB = lambda **kw: GBR(max_iter=600, learning_rate=.06, random_state=42, **kw)

# ---------------------------------------------------------------- 1. groupwise coverage
tr, cal = train_test_split(tr_all, test_size=.25, random_state=42)     # as run_diagnostics.py
NOM = [0.50, 0.68, 0.80, 0.90, 0.95]


def cqr(nominal):
    lo_q, hi_q = (1 - nominal) / 2, 1 - (1 - nominal) / 2
    mlo = GB(loss='quantile', quantile=lo_q).fit(X[tr], y[tr])
    mhi = GB(loss='quantile', quantile=hi_q).fit(X[tr], y[tr])
    e = np.maximum(mlo.predict(X[cal]) - y[cal], y[cal] - mhi.predict(X[cal]))
    Q = float(np.quantile(e, min(1.0, nominal * (1 + 1 / len(cal)))))
    lo, hi = mlo.predict(X[r_idx]) - Q, mhi.predict(X[r_idx]) + Q
    return nominal, (yt >= lo) & (yt <= hi)


with parallel_config(backend='loky', inner_max_num_threads=4):
    cv = dict(Parallel(n_jobs=min(NJOBS, len(NOM)))(delayed(cqr)(n) for n in NOM))
group = {'nominal': NOM, 'species': {}}
for code, nm in C.SPECIES.items():
    m = pid == code
    group['species'][nm] = [round(100 * float(cv[n][m].mean()), 1) for n in NOM]
dev = [abs(v - 100 * n) for vals in group['species'].values() for v, n in zip(vals, NOM)]
group['max_abs_dev'] = round(max(dev), 1)
print('groupwise CQR coverage (nominal %s):' % NOM)
for nm, v in group['species'].items():
    print('  %-10s %s' % (nm, v))

# ---------------------------------------------------------------- 2. regime excess CIs
lne, th = d['log_Ne'], d['Theta1']
zq = np.quantile(th, [1 / 3, 2 / 3])
REG = {'logNe 4.8-5.0': lne < 5.0, 'logNe 5.0-5.2': lne >= 5.0,
       'zenith low': th < zq[0], 'zenith mid': (th >= zq[0]) & (th < zq[1]), 'zenith high': th >= zq[1]}
rng = np.random.default_rng(0)
excess = {}
for name, mask in REG.items():
    trm = np.where(mask & C.train_mask(d, test))[0]; tem = np.where(mask & test)[0]
    e = GB().fit(X[trm], y[trm]).predict(X[tem]) - y[tem]
    ids = np.where(mask)[0]; yr = y[ids]
    Z = StandardScaler().fit_transform(X[ids])
    dist, idx = NearestNeighbors(n_neighbors=11).fit(Z).kneighbors(Z)
    g = 0.5 * (yr[:, None] - yr[idx[:, 1:]]) ** 2; dl = dist[:, 1:] ** 2

    def floor(rows):
        return float(np.sqrt(max(np.polyfit(dl[rows].mean(0), g[rows].mean(0), 1)[1], 0)))

    f0 = floor(slice(None)); s0 = float(e.std())
    bs = [100 * (e[rng.integers(0, len(e), len(e))].std() / floor(rng.integers(0, len(ids), len(ids))) - 1)
          for _ in range(NB)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    excess[name] = {'sigma': round(s0, 4), 'floor': round(f0, 4), 'excess_pct': round(100 * (s0 / f0 - 1), 1),
                    'ci': [round(float(lo), 1), round(float(hi), 1)]}
    print('  %-14s GB %.4f floor %.4f excess %+.1f%% [%+.1f, %+.1f]'
          % (name, s0, f0, excess[name]['excess_pct'], lo, hi))

R['group_checks'] = {'coverage_by_species': group, 'regime_excess': excess, 'n_boot': NB}
json.dump(R, open(OUT, 'w'), indent=2)
print('extended %s' % OUT)
