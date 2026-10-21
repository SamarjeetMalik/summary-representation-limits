"""Steps 1 and 2 of the audit per energy decade on KASCADE, for one hadronic-interaction model.

The main KASCADE analysis uses one decade, 5 <= log10(E/GeV) <= 6. Here each decade of the release
(14-15, 15-16, 16-17, 17-18 in log10(E/eV)) is treated separately: estimators on the nine
reconstructed features, the Gamma-test floor estimate (test events as queries, neighbours from
other showers, shower-level bootstrap, known-noise control first), the MC-truth oracle, and the two
main ladder rungs (zenith angle, muon size). Events are drawn uniformly at random from the whole
file (the file is ordered by primary and energy, so blocks would bias the sample), up to 300,000
per decade. Split 75/25 by shower.

Writes analysis/kcdc/kcdc_energy_results.json.  Run:  python analysis/kcdc/run_kcdc_energy.py
"""
import json, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as GBR
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
# KCDC_GEN: unset = QGSJET-II-04; 'epos' = EPOS-LHC; 'sibyll' = SIBYLL-2.3d (same release, same cuts).
GEN = os.environ.get('KCDC_GEN', '')
NPZ = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'data', 'kcdc', 'kcdc_kascade_%s.npz' % (GEN or 'qgs4'))
OUT = os.path.join(HERE, 'kcdc_energy_results%s.json' % ('_' + GEN if GEN else ''))
NMAX, K, KQ, NB = 300000, 10, 40, 300
rng = np.random.default_rng(0)

D = dict(np.load(NPZ))
ze_all = D['Ze'] if np.nanmax(D['Ze']) > 3.2 else np.degrees(D['Ze'])
ok = np.ones(len(D['Ne']), bool)
for k in ('Ne', 'Nmu', 'Age', 'Xc', 'Yc', 'TrNe', 'TrNm', 'TrEP'):
    ok &= np.isfinite(D[k])
ok &= (D['Ne'] > 1) & (D['Nmu'] > 0) & (D['TrNe'] > 1) & (D['TrNm'] > 0)
GB = lambda: GBR(max_iter=600, learning_rate=.06, random_state=42)
MODELS = {'Ridge': lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)), 'GradBoost': GB,
          'MLP': lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=120,
                                                                      random_state=42, early_stopping=True))}
out = {}
for lo in (14, 15, 16, 17):
    pool = np.where(ok & (D['TrEP'] >= lo) & (D['TrEP'] < lo + 1))[0]
    ids = np.sort(rng.choice(pool, min(NMAX, len(pool)), replace=False))
    c = lambda k: D[k][ids].astype(float)
    y = c('TrEP') - 9.0; Ne, Nmu, age, xc, yc = c('Ne'), c('Nmu'), c('Age'), c('Xc'), c('Yc')
    zz = ze_all[ids].astype(float); cz = np.cos(np.radians(zz)); rc = np.hypot(xc, yc)
    F = lambda ne, nm: np.column_stack([ne, nm, nm - ne, age, zz, cz, xc, yc, rc])
    X9, XO = F(Ne, Nmu), F(c('TrNe'), c('TrNm'))
    _, sh = np.unique(np.column_stack([D['TrEP'][ids], D['TrPP'][ids].astype(float)]), axis=0, return_inverse=True)
    sh = sh.ravel(); us = np.random.default_rng(0).permutation(np.unique(sh))
    m = np.isin(sh, us[:int(.75 * len(us))]); tr, te = np.where(m)[0], np.where(~m)[0]; yt = y[te]

    sig = {k: float((mk().fit(X9[tr], y[tr]).predict(X9[te]) - yt).std()) for k, mk in MODELS.items()}
    e_gb = GB().fit(X9[tr], y[tr]).predict(X9[te]) - yt
    e_or = GB().fit(XO[tr], y[tr]).predict(XO[te]) - yt
    oracle = float(e_or.std())
    rung = lambda cols: float((GB().fit(X9[tr][:, cols], y[tr]).predict(X9[te][:, cols]) - yt).std())
    fit4, plus_zen = rung([0, 3, 6, 7]), rung([0, 3, 6, 7, 4, 5])
    plus_mu = rung([0, 3, 6, 7, 4, 5, 8, 1])

    Z = StandardScaler().fit_transform(X9)
    dist, idx = NearestNeighbors(n_neighbors=KQ).fit(Z).kneighbors(Z[te])
    order = np.sort(np.where(sh[idx] != sh[te][:, None], np.arange(KQ)[None, :], 99), 1)[:, :K]
    good = (order < 99).all(1)
    order = np.minimum(order, KQ - 1)
    d2 = (np.take_along_axis(dist, order, 1) ** 2)[good]; I = np.take_along_axis(idx, order, 1)[good]
    fl = lambda g2, r=slice(None): float(np.sqrt(max(np.polyfit(d2[r].mean(0), g2[r].mean(0), 1)[1], 0)))
    mt = 0.9 * Z[:, 0] + 0.35 * Z[:, 1] + 0.08 * np.tanh(2 * Z[:, 3])
    mt = (mt - mt.mean()) / mt.std() * y.std() * 0.85
    ys = mt + rng.normal(0, 0.10, len(y))
    ctrl = fl(0.5 * (ys[te][good][:, None] - ys[I]) ** 2)
    assert abs(ctrl - 0.10) / 0.10 < 0.10, 'floor control failed for decade %d' % lo
    g2 = 0.5 * (yt[good][:, None] - y[I]) ** 2
    floor = fl(g2)
    eg = e_gb[good]
    ug, inv = np.unique(sh[te][good], return_inverse=True)
    o_ = np.argsort(inv, kind='stable'); st = np.r_[0, np.cumsum(np.bincount(inv))]
    bx = []
    for _ in range(NB):
        r = np.concatenate([o_[st[i]:st[i + 1]] for i in rng.integers(0, len(ug), len(ug))])
        bx.append(100 * (eg[r].std() / fl(g2, r) - 1))
    # KCDC's own energy estimate E (a fixed function of Ne, Nmu and zenith, log10 eV) on the same test
    # events: raw, and calibrated by boosting on E alone. Bias by primary for boosting, KCDC and oracle.
    # Nothing here draws from rng, so every other number of this script is unchanged by it.
    ek = c('E') - 9.0
    r_k = ek[te] - yt
    e_kcal = GB().fit(ek[tr, None], y[tr]).predict(ek[te, None]) - yt
    pp = D['TrPP'][ids][te]

    def by_primary(e):
        mb = {int(p): round(float(e[pp == p].mean()), 4) for p in np.unique(pp)}
        return {'mean_bias': mb, 'spread': round(max(mb.values()) - min(mb.values()), 4),
                'sigma': {int(p): round(float(e[pp == p].std()), 4) for p in np.unique(pp)}}
    kcdc = {'raw_rms': round(float(np.sqrt(np.mean(r_k ** 2))), 4), 'raw_sd': round(float(r_k.std()), 4),
            'raw_mean_bias': round(float(r_k.mean()), 4), 'calibrated_sigma': round(float(e_kcal.std()), 4),
            'by_primary_raw': by_primary(r_k - r_k.mean()), 'by_primary_calibrated': by_primary(e_kcal)}
    prim = {'GradBoost': by_primary(e_gb), 'oracle': by_primary(e_or)}
    out['%d-%d' % (lo, lo + 1)] = {
        'kcdc_estimate': kcdc, 'by_primary': prim,
        'n_events': int(len(ids)), 'n_test': int(len(te)), 'target_sd': round(float(y.std()), 3),
        'sigma': {k: round(v, 4) for k, v in sig.items()}, 'floor': round(floor, 4), 'floor_control_0.10': round(ctrl, 4),
        'gb_excess_pct': round(100 * (float(eg.std()) / floor - 1), 2),
        'gb_excess_ci': [round(float(v), 2) for v in np.percentile(bx, [2.5, 97.5])],
        'oracle': round(oracle, 4), 'oracle_share_of_variance_pct': round(100 * (1 - oracle ** 2 / sig['GradBoost'] ** 2), 1),
        'ladder': {'fit_only': round(fit4, 4), 'plus_zenith': round(plus_zen, 4), 'plus_muon_size': round(plus_mu, 4)},
        'frac_queries_with_other_shower_neighbours': round(float(good.mean()), 4)}
    print(lo, out['%d-%d' % (lo, lo + 1)], flush=True)
json.dump(out, open(OUT, 'w'), indent=2)
print('wrote', OUT)
