"""How accurate is the Gamma-test floor estimate in the KASCADE geometry itself?

The known-truth benchmark of the paper (run_synthetic.py) uses a Gaussian measurement, so it cannot
show how the estimator behaves on the real feature geometry, sample size and shower clustering of the
KASCADE samples. Here the target is replaced by a semi-synthetic one with a KNOWN floor, built on the
real nine-feature matrix of each sample:

    y_syn = g(X) + s(X) * xi_shower

  g     gradient boosting fitted to the real target on the training showers (a realistic regression
        surface, including its curvature and its interaction with the feature geometry);
  s(X)  square root of gradient boosting fitted to out-of-fold squared residuals (a realistic,
        heteroscedastic noise level);
  xi    standard normal, drawn once per CORSIKA shower, so copies of a shower share their noise as
        they share their true energy.

The floor of the nine features for y_syn over the test events is sqrt(mean s(X)^2), known exactly.
For R noise replicates every setting of run_kcdc_floor_sens.py is evaluated on the same neighbours
(test events as queries, neighbours from other showers): K = 5..50, three scalings, the Manhattan
metric, the six non-redundant coordinates, and the direct 1-NN value sqrt(gamma(1)); for the
reference setting (standardised, Euclidean, K = 10) also the 95% shower-bootstrap interval, to
measure its actual coverage.

CONTROL FIRST: with s(X) constant at 0.10 the reference setting must recover 0.10 within 10%.

Writes analysis/kcdc/kcdc_floor_benchmark[_gen][_dDEC].json.
Run:  [KCDC_GEN=epos] [KCDC_DECADE=15-16] python analysis/kcdc/run_kcdc_floor_benchmark.py
"""
import json, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as GBR
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
GEN = os.environ.get('KCDC_GEN', '')
DEC = os.environ.get('KCDC_DECADE', '')
SUF = ('_' + GEN if GEN else '') + ('_d' + DEC if DEC else '')
NPZ = (os.path.join(ROOT, 'data', 'kcdc', 'kcdc_kascade_%s.npz' % (GEN or 'qgs4')) if DEC else
       os.path.join(ROOT, 'data', 'kcdc', 'kcdc_station_features%s.npz' % ('_' + GEN if GEN else '')))
OUT = os.path.join(HERE, 'kcdc_floor_benchmark%s.json' % SUF)
KMAX, KQ, NB = 50, 80, 200
REPS = int(os.environ.get('REPS', '20'))

# ---- the sample, shower identity and split of run_kcdc_floor_sens.py (identical selection)
D = dict(np.load(NPZ))
ze = D['Ze'] if np.nanmax(D['Ze']) > 3.2 else np.degrees(D['Ze'])
y_all = D['TrEP'] - 9.0
ok = np.ones(len(y_all), bool)
for k in ('Ne', 'Nmu', 'Age', 'Xc', 'Yc', 'TrNe', 'TrNm'):
    ok &= np.isfinite(D[k])
lo, hi = ((float(DEC.split('-')[0]) - 9, float(DEC.split('-')[1]) - 9) if DEC else (5, 6))
ok &= (D['Ne'] > 1) & (D['Nmu'] > 0) & (D['TrNe'] > 1) & (D['TrNm'] > 0) & (y_all >= lo) & (y_all <= hi)
if DEC:
    ok &= y_all < hi
ids = np.where(ok)[0]
if DEC and len(ids) > 300000:
    ids = np.sort(np.random.default_rng(1).choice(ids, 300000, replace=False))
y = y_all[ids]
_, shower = np.unique(np.column_stack([D['TrEP'][ids], D['TrPP'][ids].astype(np.float32)]), axis=0, return_inverse=True)
shower = shower.ravel()
ush = np.random.default_rng(0).permutation(np.unique(shower))
is_tr = np.isin(shower, ush[:int(.75 * len(ush))])
tr, te = np.where(is_tr)[0], np.where(~is_tr)[0]
col = lambda k: D[k][ids].astype(float)
Ne, Nmu, age, xc, yc = col('Ne'), col('Nmu'), col('Age'), col('Xc'), col('Yc')
zz = ze[ids].astype(float)
NINE = np.column_stack([Ne, Nmu, Nmu - Ne, age, zz, np.cos(np.radians(zz)), xc, yc, np.hypot(xc, yc)])
SIX = np.column_stack([Ne, Nmu, age, zz, xc, yc])
print('KASCADE %s %s: %d events, %d test queries' % (GEN or 'qgs4', DEC or 'stations', len(y), len(te)), flush=True)

# ---- realistic regression surface and noise level, both learned on training showers only
GB = lambda: GBR(max_iter=600, learning_rate=.06, random_state=42)
g = GB().fit(NINE[tr], y[tr]).predict(NINE)
oof = np.empty(len(tr))
for a, b in GroupKFold(3).split(NINE[tr], groups=shower[tr]):
    oof[b] = GB().fit(NINE[tr][a], y[tr][a]).predict(NINE[tr][b])
s2 = np.clip(GBR(max_iter=300, learning_rate=.06, random_state=43).fit(NINE[tr], (y[tr] - oof) ** 2).predict(NINE), 1e-6, None)
s = np.sqrt(s2)
TRUE = float(np.sqrt(s2[te].mean()))
print('semi-synthetic target: true floor on the test events %.4f (s ranges %.3f to %.3f)'
      % (TRUE, np.percentile(s[te], 5), np.percentile(s[te], 95)), flush=True)

# ---- neighbours for every setting (they do not depend on the target)
SCALERS = {'zscore': StandardScaler, 'robust': RobustScaler,
           'rank': lambda: QuantileTransformer(output_distribution='uniform', random_state=0)}


def neighbours(X, scaler, p=2):
    Z = scaler().fit_transform(X)
    dist, idx = NearestNeighbors(n_neighbors=KQ, p=p).fit(Z).kneighbors(Z[te])
    other = shower[idx] != shower[te][:, None]
    order = np.sort(np.where(other, np.arange(KQ)[None, :], 999), 1)[:, :KMAX]
    assert (order < 999).all(), 'fewer than KMAX other-shower neighbours for some events'
    return np.take_along_axis(dist, order, 1) ** 2, np.take_along_axis(idx, order, 1)


def floor(d2, g2, K, rows=slice(None)):
    return float(np.sqrt(max(np.polyfit(d2[rows, :K].mean(0), g2[rows, :K].mean(0), 1)[1], 0)))


NB_ = {name: neighbours(NINE, sc) for name, sc in SCALERS.items()}
NB_['zscore-L1'] = neighbours(NINE, StandardScaler, 1)
NB_['six'] = neighbours(SIX, StandardScaler)
SETTINGS = ([('zscore', K) for K in (5, 10, 20, 50)] + [('robust', 10), ('rank', 10), ('zscore-L1', 10), ('six', 10)])
REF = ('zscore', 10)

ug, inv = np.unique(shower[te], return_inverse=True)
order_ = np.argsort(inv, kind='stable'); starts = np.r_[0, np.cumsum(np.bincount(inv))]
boot_rows = [np.concatenate([order_[starts[i]:starts[i + 1]] for i in np.random.default_rng(1000 + b).integers(0, len(ug), len(ug))])
             for b in range(NB)]

# ---- control: constant noise must be recovered
d2r, Ir = NB_['zscore']
xi0 = np.random.default_rng(7).standard_normal(shower.max() + 1)
yc_ = g + 0.10 * xi0[shower]
ctrl = floor(d2r, 0.5 * (yc_[te][:, None] - yc_[Ir]) ** 2, 10)
assert abs(ctrl - 0.10) / 0.10 < 0.10, 'constant-noise control failed: %.4f' % ctrl
print('control (constant 0.10): %.4f' % ctrl, flush=True)

est = {('%s_K%d' % st): [] for st in SETTINGS}
est['direct_1nn'] = []
cover, widths = [], []
for r in range(REPS):
    xi = np.random.default_rng(100 + r).standard_normal(shower.max() + 1)
    ys = g + s * xi[shower]
    for name, K in SETTINGS:
        d2, I = NB_[name]
        est['%s_K%d' % (name, K)].append(floor(d2, 0.5 * (ys[te][:, None] - ys[I]) ** 2, K))
    est['direct_1nn'].append(float(np.sqrt(np.mean(0.5 * (ys[te] - ys[Ir[:, 0]]) ** 2))))
    g2 = 0.5 * (ys[te][:, None] - ys[Ir]) ** 2
    bt = [floor(d2r, g2, 10, rw) for rw in boot_rows]
    lo_, hi_ = np.percentile(bt, [2.5, 97.5])
    cover.append(bool(lo_ <= TRUE <= hi_)); widths.append(float(hi_ - lo_))
    print('  replicate %2d: reference %.4f  interval [%.4f, %.4f]  covers %s'
          % (r, est['zscore_K10'][-1], lo_, hi_, cover[-1]), flush=True)

out = {'sample': DEC or '5-6 (lgE/GeV), station-level', 'generator': GEN or 'qgs4', 'n_events': int(len(y)),
       'n_test': int(len(te)), 'replicates': REPS, 'true_floor': round(TRUE, 5), 'control_constant_0.10': round(ctrl, 4),
       'settings': {k: {'mean': round(float(np.mean(v)), 5), 'sd': round(float(np.std(v)), 5),
                        'bias_pct': round(100 * (float(np.mean(v)) / TRUE - 1), 2)} for k, v in est.items()},
       'reference_interval_coverage': round(float(np.mean(cover)), 3),
       'reference_interval_mean_width': round(float(np.mean(widths)), 5)}
b_ = [v['bias_pct'] for k, v in out['settings'].items() if k != 'direct_1nn']
out['settings_bias_range_pct'] = [min(b_), max(b_)]
print({k: out[k] for k in ('true_floor', 'reference_interval_coverage', 'settings_bias_range_pct')})
print({k: v['bias_pct'] for k, v in out['settings'].items()})
json.dump(out, open(OUT, 'w'), indent=2)
print('wrote %s' % OUT)
