"""Steps 1 and 2 of the audit on two further hadronic-interaction models.

The KASCADE release also provides simulations made with EPOS-LHC and SIBYLL-2.3d. For each, the
question asked here is the within-sample one: do estimators of unrelated inductive bias plateau
at that sample's own floor estimate, and does the information ladder have the same shape as with
QGSJET-II-04?

Caveat that travels with every number from this script. The two input files hold 240,000 events
each, read as 30 contiguous blocks of 8,000 from files ordered by primary. That is not a random
sample of the release (see extract_stations.py for what block sampling does on QGSJET), and the
raw files were not available to re-extract. Plateau and floor are both measured on the same
events, so their comparison is meaningful; absolute values, and differences between generators,
are not.

Inputs (not shipped): data/kcdc/generators/perstation_epos.npz and perstation_sibyll.npz (or
GENERATOR_DIR), with the nine
reconstructed features F9 = (Ne, Nmu, Nmu - Ne, age, zenith, cos zenith, Xc, Yc, Rc), the true
log10(E/eV) and the primary. Window, split and models are those of run_kcdc_floor.py.

Writes analysis/kcdc/kcdc_generators.json.  Run:  python analysis/kcdc/run_kcdc_generators.py
"""
import json, os, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as GBR, RandomForestRegressor as RF
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get('GENERATOR_DIR', os.path.join(os.path.dirname(os.path.dirname(HERE)), 'data', 'kcdc', 'generators'))
OUT = os.path.join(HERE, 'kcdc_generators.json')
FILES = {'EPOS-LHC': 'perstation_epos.npz', 'SIBYLL-2.3d': 'perstation_sibyll.npz'}
K, KQ, NB = 10, 40, 500
NJOBS = int(os.environ.get('NJOBS', max(1, (os.cpu_count() or 4) - 1)))
rng = np.random.default_rng(0)
if not all(os.path.exists(os.path.join(SRC, f)) for f in FILES.values()):
    sys.exit('generator files not found in %s: nothing done' % SRC)

GB = lambda: GBR(max_iter=600, learning_rate=.06, random_state=42)
MODELS = {'Ridge': lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
          'RandomForest': lambda: RF(n_estimators=60, min_samples_leaf=25, max_depth=18, n_jobs=NJOBS, random_state=42),
          'GradBoost': GB,
          'MLP': lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=120,
                                                                      random_state=42, early_stopping=True))}
# ladder in the column order of F9: fit (Ne, age, Xc, Yc) -> + zenith -> + core distance -> + muon size -> + ratio
LADDER = [('A', [0, 3, 6, 7]), ('B', [0, 3, 6, 7, 4, 5]), ('C', [0, 3, 6, 7, 4, 5, 8]),
          ('D', [0, 3, 6, 7, 4, 5, 8, 1]), ('E', [0, 3, 6, 7, 4, 5, 8, 1, 2])]

out = {}
for gen, fn in FILES.items():
    D = np.load(os.path.join(SRC, fn))
    y_all = D['y'].astype(np.float64) - 9.0                    # log10(E/GeV)
    m = (y_all >= 5) & (y_all <= 6)
    X = D['F9'][m].astype(np.float64); y = y_all[m]; pp = D['pp'][m]
    _, sh = np.unique(np.column_stack([y, pp.astype(float)]), axis=0, return_inverse=True); sh = sh.ravel()
    us = np.random.default_rng(0).permutation(np.unique(sh))
    is_tr = np.isin(sh, us[:int(.75 * len(us))]); tr, te = np.where(is_tr)[0], np.where(~is_tr)[0]
    yt = y[te]
    print('%s: %d events (%d showers), train %d / test %d, target sd %.3f, protons %.1f%%'
          % (gen, len(y), len(us), len(tr), len(te), y.std(), 100 * (pp == 14).mean()), flush=True)

    # floor: test events as queries, neighbours from other showers
    Z = StandardScaler().fit_transform(X)
    dist, idx = NearestNeighbors(n_neighbors=KQ).fit(Z).kneighbors(Z[te])
    order = np.sort(np.where(sh[idx] != sh[te][:, None], np.arange(KQ)[None, :], 99), 1)[:, :K]
    assert (order < 99).all()
    d2 = np.take_along_axis(dist, order, 1) ** 2; I = np.take_along_axis(idx, order, 1)
    fl = lambda g2, r=slice(None): float(np.sqrt(max(np.polyfit(d2[r].mean(0), g2[r].mean(0), 1)[1], 0)))
    mt = 0.9 * Z[:, 0] + 0.35 * Z[:, 1] + 0.08 * np.tanh(2 * Z[:, 3]) + 0.05 * Z[:, 0] * Z[:, 3]
    mt = (mt - mt.mean()) / mt.std() * y.std() * 0.85
    ctrl = []
    for s_noise in (0.10, 0.15):
        ys = mt + rng.normal(0, s_noise, len(y))
        est = fl(0.5 * (ys[te][:, None] - ys[I]) ** 2)
        ctrl.append({'true_sd': s_noise, 'est': round(est, 4)})
        assert abs(est - s_noise) / s_noise < 0.10, 'floor control failed on %s' % gen
    g2 = 0.5 * (yt[:, None] - y[I]) ** 2
    floor = fl(g2)

    E_ = {k: mk().fit(X[tr], y[tr]).predict(X[te]) - yt for k, mk in MODELS.items()}
    ug, inv = np.unique(sh[te], return_inverse=True)
    o_ = np.argsort(inv, kind='stable'); st = np.r_[0, np.cumsum(np.bincount(inv))]
    BOOT = [np.concatenate([o_[st[i]:st[i + 1]] for i in rng.integers(0, len(ug), len(ug))]) for _ in range(NB)]
    e = E_['GradBoost']
    bf = [fl(g2, r) for r in BOOT]; bx = [100 * (e[r].std() / f - 1) for r, f in zip(BOOT, bf)]
    ladder, prev = [], None
    for key, colsel in LADDER:
        el = GB().fit(X[tr][:, colsel], y[tr]).predict(X[te][:, colsel]) - yt
        row = {'rung': key, 'sigma': round(float(el.std()), 4)}
        if prev is not None:
            dl = [el[r].std() - prev[r].std() for r in BOOT]
            row['delta'] = round(float(el.std() - prev.std()), 5)
            row['delta_ci'] = [round(float(v), 5) for v in np.percentile(dl, [2.5, 97.5])]
        ladder.append(row); prev = el
    sig = {k: round(float(v.std()), 4) for k, v in E_.items()}
    nl = [sig[k] for k in ('RandomForest', 'GradBoost', 'MLP')]
    out[gen] = {'n_events': int(len(y)), 'n_test': int(len(te)), 'target_sd': round(float(y.std()), 3),
                'proton_pct': round(100 * float((pp == 14).mean()), 1), 'controls': ctrl, 'sigma': sig,
                'nonlin_spread': round(max(nl) - min(nl), 4), 'floor': round(floor, 4),
                'floor_ci': [round(float(v), 4) for v in np.percentile(bf, [2.5, 97.5])],
                'gb_excess_pct': round(100 * (float(e.std()) / floor - 1), 2),
                'gb_excess_ci': [round(float(v), 2) for v in np.percentile(bx, [2.5, 97.5])], 'ladder': ladder}
    print('  ', {k: out[gen][k] for k in ('sigma', 'floor', 'floor_ci', 'gb_excess_pct', 'gb_excess_ci')}, flush=True)
    print('   ladder', [(r['rung'], r['sigma'], r.get('delta_ci')) for r in ladder], flush=True)
json.dump(out, open(OUT, 'w'), indent=2)
print('wrote %s' % OUT)
