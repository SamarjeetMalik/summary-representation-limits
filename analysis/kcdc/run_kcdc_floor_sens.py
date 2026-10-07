"""Sensitivity of the KASCADE floor estimate, mirroring the GRAPES-3 study.

run_kcdc_floor.py reports one Gamma-test setting (K = 10, standardised features, all events).
Here the estimate is repeated with the TEST events as query points (neighbours from all other
showers), over neighbour ranges K = 5..50 and three feature scalings, on the six non-redundant
coordinates, and with a shower-level bootstrap taken jointly with gradient boosting's errors.

CONTROL FIRST: each scaling must recover a known noise level on the real feature matrix.

Writes analysis/kcdc/kcdc_floor_sens.json.  Run:  python analysis/kcdc/run_kcdc_floor_sens.py
"""
import json, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as GBR
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
NPZ = os.path.join(ROOT, 'data', 'kcdc', 'kcdc_station_features.npz')
OUT = os.path.join(HERE, 'kcdc_floor_sens.json')
KMAX, KQ, NB = 50, 80, 500
rng = np.random.default_rng(0)

# same events, shower identity and split as run_kcdc_floor.py
D = dict(np.load(NPZ))
ze = D['Ze'] if np.nanmax(D['Ze']) > 3.2 else np.degrees(D['Ze'])
y_all = D['TrEP'] - 9.0
ok = np.ones(len(y_all), bool)
for k in ('Ne', 'Nmu', 'Age', 'Xc', 'Yc', 'TrNe', 'TrNm'):
    ok &= np.isfinite(D[k])
ok &= (D['Ne'] > 1) & (D['Nmu'] > 0) & (D['TrNe'] > 1) & (D['TrNm'] > 0) & (y_all >= 5) & (y_all <= 6)
ids = np.where(ok)[0]
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
print('KASCADE: %d events, %d test queries' % (len(y), len(te)))

SCALERS = {'zscore': StandardScaler, 'robust': RobustScaler,
           'rank': lambda: QuantileTransformer(output_distribution='uniform', random_state=0)}


def neighbours(X, scaler):
    Z = scaler().fit_transform(X)
    dist, idx = NearestNeighbors(n_neighbors=KQ).fit(Z).kneighbors(Z[te])
    other = shower[idx] != shower[te][:, None]
    order = np.sort(np.where(other, np.arange(KQ)[None, :], 999), 1)[:, :KMAX]
    assert (order < 999).all(), 'fewer than KMAX other-shower neighbours for some events'
    return np.take_along_axis(dist, order, 1) ** 2, np.take_along_axis(idx, order, 1)


def floor(d2, g2, K, rows=slice(None)):
    return float(np.sqrt(max(np.polyfit(d2[rows, :K].mean(0), g2[rows, :K].mean(0), 1)[1], 0)))


Zc = StandardScaler().fit_transform(NINE)
mtrue = 0.9 * Zc[:, 0] + 0.35 * Zc[:, 1] + 0.08 * np.tanh(2 * Zc[:, 3]) + 0.05 * Zc[:, 0] * Zc[:, 3]
mtrue = (mtrue - mtrue.mean()) / mtrue.std() * y.std() * 0.85
out = {'n_test': int(len(te)), 'controls': {}, 'gamma': {}}
NBR = {}
for sname, sc in SCALERS.items():
    d2, I = NBR[sname] = neighbours(NINE, sc)
    ctrl = []
    for s_noise in (0.10, 0.15):
        ys = mtrue + rng.normal(0, s_noise, len(y))
        est = floor(d2, 0.5 * (ys[te][:, None] - ys[I]) ** 2, 10)
        ctrl.append({'true_sd': s_noise, 'est': round(est, 4)})
        assert abs(est - s_noise) / s_noise < 0.10, 'control failed for %s' % sname
    out['controls'][sname] = ctrl
    g2 = 0.5 * (y[te][:, None] - y[I]) ** 2
    out['gamma'][sname] = {str(K): round(floor(d2, g2, K), 4) for K in (5, 10, 20, 50)}
    print('%-7s controls %s  floor %s' % (sname, [c['est'] for c in ctrl], out['gamma'][sname]))

d6, I6 = neighbours(SIX, StandardScaler)
out['six_coordinates'] = round(floor(d6, 0.5 * (y[te][:, None] - y[I6]) ** 2, 10), 4)

# gradient boosting and the joint shower-level bootstrap (K = 10, standardised)
e = GBR(max_iter=600, learning_rate=.06, random_state=42).fit(NINE[tr], y[tr]).predict(NINE[te]) - y[te]
d2, I = NBR['zscore']; g2 = 0.5 * (y[te][:, None] - y[I]) ** 2
ug, inv = np.unique(shower[te], return_inverse=True)
order = np.argsort(inv, kind='stable'); starts = np.r_[0, np.cumsum(np.bincount(inv))]
bf, bx = [], []
for _ in range(NB):
    pick = rng.integers(0, len(ug), len(ug))
    rows = np.concatenate([order[starts[i]:starts[i + 1]] for i in pick])
    f = floor(d2, g2, 10, rows); bf.append(f); bx.append(100 * (e[rows].std() - f) / f)
point = floor(d2, g2, 10)
allv = [v for r in out['gamma'].values() for v in r.values()] + [out['six_coordinates']]
out.update({'gb_sigma': round(float(e.std()), 4), 'floor_test': round(point, 4),
            'floor_test_ci_shower': [round(float(v), 4) for v in np.percentile(bf, [2.5, 97.5])],
            'gb_excess_pct': round(100 * (float(e.std()) - point) / point, 2),
            'gb_excess_ci_shower': [round(float(v), 2) for v in np.percentile(bx, [2.5, 97.5])],
            'range': [min(allv), max(allv)]})
print({k: out[k] for k in ('gb_sigma', 'floor_test', 'floor_test_ci_shower', 'gb_excess_pct',
                           'gb_excess_ci_shower', 'six_coordinates', 'range')})
json.dump(out, open(OUT, 'w'), indent=2)
print('wrote %s' % OUT)
