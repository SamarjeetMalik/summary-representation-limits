"""The per-decade audit without knowing the true-energy decade.

run_kcdc_energy.py trains and evaluates inside one decade of TRUE energy at a time, so every model
there (and the floor estimate) is conditioned on knowing y in [a, a+1], which shrinks errors near the
decade edges. Here each estimator is trained once on the whole release (true 10^14-10^18 eV), and
its test errors are then stratified by true decade, as an experiment would do:

  boosting on the nine features; the oracle (true sizes); KCDC's estimate E_rec calibrated by
  boosting on E_rec alone; and the raw E_rec.

Both the standard deviation of the residual and the root-mean-square error are reported, overall
and per true decade, with the mean residual. The Gamma-test floor of the nine features is estimated
once for the whole range (test events as queries, neighbours from other showers). Split 75/25 by
shower; at most NMAX events drawn uniformly at random from the release.

Writes analysis/kcdc/kcdc_fullrange_results[_gen].json.  Run: [KCDC_GEN=epos] python run_kcdc_fullrange.py
"""
import json, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as GBR
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.environ.get('KCDC_GEN', '')
NPZ = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'data', 'kcdc', 'kcdc_kascade_%s.npz' % (GEN or 'qgs4'))
FLAT = os.environ.get('FLAT', '') == '1'        # weight training to a flat distribution in log10 E
OUT = os.path.join(HERE, 'kcdc_fullrange_results%s%s.json' % ('_' + GEN if GEN else '', '_flat' if FLAT else ''))
NMAX, K, KQ = int(os.environ.get('NMAX', '2000000')), 10, 40
rng = np.random.default_rng(0)

D = dict(np.load(NPZ))
ze_all = D['Ze'] if np.nanmax(D['Ze']) > 3.2 else np.degrees(D['Ze'])
ok = np.ones(len(D['Ne']), bool)
for k in ('Ne', 'Nmu', 'Age', 'Xc', 'Yc', 'TrNe', 'TrNm', 'TrEP', 'E'):
    ok &= np.isfinite(D[k])
ok &= (D['Ne'] > 1) & (D['Nmu'] > 0) & (D['TrNe'] > 1) & (D['TrNm'] > 0) & (D['TrEP'] >= 14) & (D['TrEP'] < 18)
pool = np.where(ok)[0]
ids = np.sort(rng.choice(pool, min(NMAX, len(pool)), replace=False))
c = lambda k: D[k][ids].astype(float)
y = c('TrEP') - 9.0; Ne, Nmu, age, xc, yc = c('Ne'), c('Nmu'), c('Age'), c('Xc'), c('Yc')
zz = ze_all[ids].astype(float); cz = np.cos(np.radians(zz)); rc = np.hypot(xc, yc)
F = lambda ne, nm: np.column_stack([ne, nm, nm - ne, age, zz, cz, xc, yc, rc])
X9, XO = F(Ne, Nmu), F(c('TrNe'), c('TrNm'))
ek = c('E') - 9.0
_, sh = np.unique(np.column_stack([D['TrEP'][ids], D['TrPP'][ids].astype(float)]), axis=0, return_inverse=True)
sh = sh.ravel(); us = np.random.default_rng(0).permutation(np.unique(sh))
m = np.isin(sh, us[:int(.75 * len(us))]); tr, te = np.where(m)[0], np.where(~m)[0]; yt = y[te]
GB = lambda: GBR(max_iter=600, learning_rate=.06, random_state=42)
# generated spectrum is E^-2, i.e. dN/dlgE ~ 10^-lgE; weights 10^(lgE - 5) make it flat in lgE
SW = (10.0 ** (y[tr] - 5.0)) if FLAT else None
res = {'GradBoost': GB().fit(X9[tr], y[tr], sample_weight=SW).predict(X9[te]) - yt,
       'oracle': GB().fit(XO[tr], y[tr], sample_weight=SW).predict(XO[te]) - yt,
       'KCDC calibrated': GB().fit(ek[tr, None], y[tr], sample_weight=SW).predict(ek[te, None]) - yt,
       'KCDC raw': ek[te] - yt}
print('full range: %d events, train %d / test %d' % (len(ids), len(tr), len(te)), flush=True)


def stats(e):
    return {'n': int(len(e)), 'sd': round(float(e.std()), 4), 'rmse': round(float(np.sqrt(np.mean(e ** 2))), 4),
            'mean': round(float(e.mean()), 4)}


out = {'flat_training_weights': FLAT, 'n_events': int(len(ids)), 'n_test': int(len(te)), 'overall': {k: stats(e) for k, e in res.items()}, 'by_true_decade': {}}
for lo in (14, 15, 16, 17):
    w = (yt + 9 >= lo) & (yt + 9 < lo + 1)
    row = {k: stats(e[w]) for k, e in res.items()}
    row['oracle_share_of_variance_pct_rmse'] = round(100 * (1 - row['oracle']['rmse'] ** 2 / row['GradBoost']['rmse'] ** 2), 1)
    out['by_true_decade']['%d-%d' % (lo, lo + 1)] = row
    print(lo, {k: (v['sd'], v['rmse']) for k, v in row.items() if isinstance(v, dict)}, row['oracle_share_of_variance_pct_rmse'], flush=True)

# Gamma-test floor of the nine features over the whole range (control first)
Z = StandardScaler().fit_transform(X9)
dist, idx = NearestNeighbors(n_neighbors=KQ).fit(Z).kneighbors(Z[te])
order = np.sort(np.where(sh[idx] != sh[te][:, None], np.arange(KQ)[None, :], 99), 1)[:, :K]
good = (order < 99).all(1); order = np.minimum(order, KQ - 1)
d2 = (np.take_along_axis(dist, order, 1) ** 2)[good]; I = np.take_along_axis(idx, order, 1)[good]
fl = lambda g2: float(np.sqrt(max(np.polyfit(d2.mean(0), g2.mean(0), 1)[1], 0)))
mt = 0.9 * Z[:, 0] + 0.35 * Z[:, 1] + 0.08 * np.tanh(2 * Z[:, 3]); mt = (mt - mt.mean()) / mt.std() * y.std() * 0.85
ys = mt + rng.normal(0, 0.10, len(y)); ctrl = fl(0.5 * (ys[te][good][:, None] - ys[I]) ** 2)
assert abs(ctrl - 0.10) / 0.10 < 0.10, 'floor control failed'
out['floor_overall'] = round(fl(0.5 * (yt[good][:, None] - y[I]) ** 2), 4); out['floor_control_0.10'] = round(ctrl, 4)
out['gb_excess_overall_pct'] = round(100 * (out['overall']['GradBoost']['rmse'] / out['floor_overall'] - 1), 2)
print('floor overall', out['floor_overall'], 'GB rmse', out['overall']['GradBoost']['rmse'], 'excess %', out['gb_excess_overall_pct'])
json.dump(out, open(OUT, 'w'), indent=2)
print('wrote', OUT)
