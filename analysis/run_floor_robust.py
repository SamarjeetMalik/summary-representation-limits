"""Is the Bayes-floor estimate an artefact of one estimator setting?

Recomputes the nine-feature floor
  * with the Gamma test over different neighbour ranges, k = 1..K for K in {5, 10, 20, 50};
  * under three feature scalings: z-score, robust (median / IQR), and rank (uniform marginals);
  * with a second, independent estimator: the nearest-neighbour residual-variance estimator of
    Devroye, Gyorfi, Lugosi and Walk (2018),  L_n = (1/n) sum_i (Y_i - Y_N(i)) Y_i  (centred Y).
    It does not extrapolate to zero neighbour distance, so at finite n it absorbs some local
    variation of the regression function and is biased UPWARD. Its first control run showed this
    (+13.6% at noise 0.05, +3.7% at 0.10), so it is used only as an UPPER estimate, and its
    control checks exactly that role: never below the truth, and within 10% for noise >= 0.08,
    the regime of the real target (~0.10);
  * with neighbours restricted to other CORSIKA showers.
The Devroye estimator is checked first on the same synthetic known-noise targets as the Gamma
test (run_bayes.py), against the upper-estimate role stated above; the script stops otherwise.

Extends results.json.  Run:  python analysis/run_floor_robust.py
"""
import json, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer

import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results.json')
R = json.load(open(OUT))
KMAX = 50

d = C.load_root(); y = d['y'] - d['y'].mean(); X = C.build_features(d)
_, grp = np.unique(np.column_stack([d['PrimaryEnergy'], d['PrimaryID'].astype(float)]), axis=0,
                   return_inverse=True)
grp = grp.ravel()

SCALERS = {'zscore': StandardScaler(), 'robust': RobustScaler(),
           'rank': QuantileTransformer(n_quantiles=2000, output_distribution='uniform', subsample=10**9)}


def neighbours(Z, k=KMAX + 1):
    return NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(Z)


def gamma(dist, idx, yy, K):
    g = 0.5 * ((yy[:, None] - yy[idx[:, 1:K + 1]]) ** 2).mean(0)
    dl = (dist[:, 1:K + 1] ** 2).mean(0)
    return float(np.sqrt(max(np.polyfit(dl, g, 1)[1], 0)))


def devroye(idx, yy):
    return float(np.sqrt(max(np.mean((yy - yy[idx[:, 1]]) * yy), 0)))


# control for the Devroye estimator (same construction as run_bayes.py)
Z = StandardScaler().fit_transform(X)
dist, idx = neighbours(Z)
m = 0.9 * Z[:, 0] + 0.35 * Z[:, 1] + 0.08 * np.tanh(2 * Z[:, 2]) + 0.05 * Z[:, 0] * Z[:, 1]
m = (m - m.mean()) / m.std() * 0.17
ctrl = []
for s_ in (0.05, 0.08, 0.10, 0.12):
    yy = m + np.random.default_rng(0).normal(0, s_, len(m)); yy = yy - yy.mean()
    est = devroye(idx, yy); ctrl.append({'true_sd': s_, 'devroye': round(est, 4),
                                         'bias_pct': round(100 * (est - s_) / s_, 1)})
    print('  control: true %.3f  Devroye %.4f (%+.1f%%)' % (s_, est, 100 * (est - s_) / s_))
    assert est >= 0.99 * s_, 'Devroye estimate below the truth: not usable as an upper estimate'
    if s_ >= 0.08:
        assert (est - s_) / s_ < 0.10, 'Devroye upper estimate too loose in the target regime'

out = {'gamma': {}, 'devroye': {}, 'controls_devroye': ctrl}
for sname, sc in SCALERS.items():
    Zs = sc.fit_transform(X)
    dist_s, idx_s = (dist, idx) if sname == 'zscore' else neighbours(Zs)
    out['gamma'][sname] = {str(K): round(gamma(dist_s, idx_s, y, K), 4) for K in (5, 10, 20, 50)}
    out['devroye'][sname] = round(devroye(idx_s, y), 4)
    print('  %-7s Gamma K=5/10/20/50: %s   Devroye: %.4f'
          % (sname, ' '.join('%.4f' % v for v in out['gamma'][sname].values()), out['devroye'][sname]))

# other-shower neighbours, z-score, Devroye
same = grp[idx] == grp[:, None]
first_other = np.argmax(~same[:, 1:], axis=1) + 1
out['devroye_other_showers'] = round(float(np.sqrt(max(np.mean((y - y[idx[np.arange(len(y)), first_other]]) * y), 0))), 4)
print('  Devroye, other-shower neighbour: %.4f' % out['devroye_other_showers'])

allv = [v for s_ in out['gamma'].values() for v in s_.values()] + list(out['devroye'].values()) \
    + [out['devroye_other_showers'], R['shower_groups']['bayes_nine_other_showers']]
out['range'] = [round(min(allv), 4), round(max(allv), 4)]
print('  all estimates span [%.4f, %.4f]; best estimator sigma %.4f'
      % (*out['range'], R['estimators']['GradBoost']['sigma']))
R['floor_robust'] = out
json.dump(R, open(OUT, 'w'), indent=2)
print('extended %s' % OUT)
