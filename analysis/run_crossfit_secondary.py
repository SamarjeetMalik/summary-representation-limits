"""The secondary GRAPES-3 analyses under the shower-disjoint protocol.

run_shower_crossfit.py recomputes the headline results by a five-fold cross-fit by shower. This
script does the same for the analyses that were left on the random split: the five sub-regimes,
the calibration of the degeneracy probe, and the per-primary tables. The blocks it writes have the
same layout as the random-split blocks (regimes, probe_calibration, diagnostics.species/lopo), so
the same figure code draws both.

  sub-regimes   models trained on the regime's events from the other shower folds; the regime's
                floor estimated with its evaluation events as query points and neighbours from
                other showers of the regime; the excess of gradient boosting over that estimate
                with a joint shower-level bootstrap.
  probe         outputs from cross-fitted models where they exist (gradient boosting, the linear
                calibration, the CNNs of the fold runs); probe folds made by shower, so that no
                copy of a shower is in both the probe's training and scoring folds.
  per primary   bias and resolution of cross-fitted gradient boosting, and leave-one-primary-out
                within the cross-fit.

Extends results.json.  Run:  python analysis/run_crossfit_secondary.py   (after run_shower_crossfit.py)
"""
import glob, json, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as GBR, RandomForestRegressor as RF
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results.json')
R = json.load(open(OUT))
KF, K, KQ, NB = 5, 10, 40, 400
NJOBS = int(os.environ.get('NJOBS', max(1, (os.cpu_count() or 4) - 1)))
rng = np.random.default_rng(0)

d = C.load_root(); c = C.load_csv(); y = d['y']; X9 = C.build_features(d)
r_idx, c_idx = C.match(d, c)
test = np.zeros(len(y), bool); test[r_idx] = True
yt = y[r_idx]; pid_t = d['PrimaryID'][r_idx]
fold = C.shower_folds(d, KF); grp = C.shower_ids(d); ftest = fold[r_idx]
GB = lambda: GBR(max_iter=600, learning_rate=.06, random_state=42)


def crossfit(make, X, keep=None, rows=None):
    """Predictions for the evaluation events in `rows` (default all); `keep` restricts training."""
    rows = np.arange(len(r_idx)) if rows is None else rows
    p = np.empty(len(rows))
    for k in range(KF):
        tr = fold != k if keep is None else (fold != k) & keep
        m = ftest[rows] == k
        if m.any():
            p[m] = make().fit(X[tr], y[tr]).predict(X[r_idx[rows[m]]])
    return p


def shower_boot(g, n):
    ug, inv = np.unique(g, return_inverse=True)
    order = np.argsort(inv, kind='stable'); starts = np.r_[0, np.cumsum(np.bincount(inv))]
    return [np.concatenate([order[starts[i]:starts[i + 1]] for i in rng.integers(0, len(ug), len(ug))])
            for _ in range(n)]


out = {}

# ------------------------------------------------------------------ sub-regimes
lne, th = d['log_Ne'], d['Theta1']
zq = np.quantile(th, [1 / 3, 2 / 3])
REG = {'logNe 4.8-5.0': lne < 5.0, 'logNe 5.0-5.2': lne >= 5.0,
       'zenith low': th < zq[0], 'zenith mid': (th >= zq[0]) & (th < zq[1]), 'zenith high': th >= zq[1]}
MODELS = {'Ridge': lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
          'RandomForest': lambda: RF(n_estimators=60, min_samples_leaf=25, max_depth=18, n_jobs=NJOBS, random_state=42),
          'GradBoost': GB,
          'MLP': lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=120,
                                                                      random_state=42, early_stopping=True))}
reg_rows, excess = {}, {}
for name, mask in REG.items():
    rows = np.where(mask[r_idx])[0]
    sig, E_ = {}, {}
    for k, mk in MODELS.items():
        E_[k] = crossfit(mk, X9, keep=mask, rows=rows) - yt[rows]
        sig[k] = round(float(E_[k].std()), 4)
    # floor: the regime's evaluation events as queries, neighbours from other showers of the regime
    pool = np.where(mask)[0]; q = r_idx[rows]
    sc = StandardScaler().fit(X9[pool])
    dist, idx = NearestNeighbors(n_neighbors=KQ).fit(sc.transform(X9[pool])).kneighbors(sc.transform(X9[q]))
    idx = pool[idx]
    order = np.sort(np.where(grp[idx] != grp[q][:, None], np.arange(KQ)[None, :], 99), 1)[:, :K]
    assert (order < 99).all()
    d2 = np.take_along_axis(dist, order, 1) ** 2
    g2 = 0.5 * (y[q][:, None] - y[np.take_along_axis(idx, order, 1)]) ** 2
    fl_of = lambda r=slice(None): float(np.sqrt(max(np.polyfit(d2[r].mean(0), g2[r].mean(0), 1)[1], 0)))
    fl = fl_of()
    e = E_['GradBoost']
    bx = [100 * (e[r].std() / fl_of(r) - 1) for r in shower_boot(grp[q], NB)]
    nonlin = [sig[k] for k in ('RandomForest', 'GradBoost', 'MLP')]
    reg_rows[name] = {'n_test': int(len(rows)), 'floor': round(fl, 4), 'sigma': sig,
                      'best_over_floor_pct': round(100 * (min(nonlin) - fl) / fl, 1),
                      'nonlin_spread': round(max(nonlin) - min(nonlin), 4)}
    excess[name] = {'excess_pct': round(100 * (e.std() / fl - 1), 2),
                    'ci': [round(float(v), 2) for v in np.percentile(bx, [2.5, 97.5])]}
    print('  %-14s floor %.4f | %s | GB excess %+.1f%% %s' % (name, fl, sig, excess[name]['excess_pct'], excess[name]['ci']))
out['regimes'] = {'zenith_edges': [round(float(v), 2) for v in zq], 'rows': reg_rows, 'regime_excess': excess}

# ------------------------------------------------------------------ probe calibration, folds by shower
PHI = np.column_stack([d['log_Ne'], d['Age'], d['NKGX'], d['NKGY']])
MU = np.log10(1.0 + d['MuonsLarge'].astype(float))
SETS = {'phi': PHI[r_idx], 'phi_M': np.column_stack([PHI, MU, d['Theta1'][:, None]])[r_idx]}
pfold = np.random.default_rng(5).integers(0, 5, grp.max() + 1)[grp[r_idx]]      # probe folds, by shower


def probe(m, feats):
    oof = np.empty_like(m, dtype=float)
    for k in range(5):
        a, b = pfold != k, pfold == k
        oof[b] = GBR(max_iter=400, random_state=0).fit(feats[a], m[a]).predict(feats[b])
    return C.r2(oof, m)


def row(label, m, kind):
    o = {'kind': kind, **{k: round(probe(m, F), 4) for k, F in SETS.items()}}
    o['increment'] = round(o['phi_M'] - o['phi'], 4)
    print('  %-34s R2(phi)=%.4f  R2(phi+M)=%.4f' % (label, o['phi'], o['phi_M']))
    return o


files = sorted(glob.glob(os.path.join(HERE, 'folds', 'render_fold*.npz')))
assert len(files) == KF, 'the fold runs of run_rendering.py are needed'
pos = {int(e): i for i, e in enumerate(r_idx)}
NETP = {k: np.empty(len(r_idx)) for k in ('B_rendered', 'C_rendered_muon')}
for fn in files:
    z = np.load(fn); at = np.array([pos[int(i)] for i in z['idx']])
    for k in NETP:
        NETP[k][at] = z[k]
S = np.load(os.path.join(HERE, 'preds_sweep.npz')); P = np.load(os.path.join(HERE, 'preds_rendering.npz'))
assert np.array_equal(S['r_idx'], r_idx)

gb4 = crossfit(GB, PHI)
lin = crossfit(LinearRegression, PHI[:, :2])
gb9 = crossfit(GB, X9)
rows = {'gb_phi': row('GB on phi', gb4, 'function of phi'),
        'linear_Ne_s': row('linear (log Ne, s) calibration', lin, 'function of phi, exact within a fold'),
        'cnn_rendered': row('CNN, rendered image (B)', NETP['B_rendered'], 'function of phi'),
        'vit_rendered': row('ViT, rendered image (G), random-split model', S['G_vit_patches'], 'function of phi'),
        'cnn_rendered_muon': row('CNN, rendered + muons (C)', NETP['C_rendered_muon'], 'phi + muons'),
        'gb_nine': row('GB on nine features', gb9, 'phi + muon total + angle'),
        'true_energy': row('true energy (not a model)', yt, 'reference')}
noise = {'%.2f' % t: row('GB on phi + N(0, %.2f)' % t, gb4 + rng.normal(0, t, len(gb4)), 'phi + pure noise')
         for t in (0.02, 0.05, 0.10)}
mu_only = crossfit(lambda: GBR(max_iter=400, random_state=0), MU)
mu_only = (mu_only - mu_only.mean()) / mu_only.std() * gb4.std() + gb4.mean()
mix = {'%.1f' % l: row('(1-%.1f) GB(phi) + %.1f u(M)' % (l, l), (1 - l) * gb4 + l * mu_only, 'phi + muon information')
       for l in (0.1, 0.3, 0.5)}
arch = c['cnnvit'][c_idx]
rows['archived'] = row('archived CNN/ViT', arch, 'unknown')
rows['archived']['phi_M_trueE'] = round(probe(arch, np.column_stack([SETS['phi_M'], yt])), 4)
b1, b2 = P['B_rendered'], P['B_rendered_seed2']
out['probe_calibration'] = {'rows': rows, 'noise': noise, 'mix': mix,
                            'retrain_ceiling': round(C.r2(LinearRegression().fit(b2[:, None], b1).predict(b2[:, None]), b1), 4)}

# ------------------------------------------------------------------ per primary, and leave-one-primary-out
e9 = gb9 - yt
pid_all = d['PrimaryID']
species, lopo = {}, {}
for code, nm in C.SPECIES.items():
    m = pid_t == code; rws = np.where(m)[0]
    species[nm] = {'n': int(m.sum()), 'bias': round(float(e9[m].mean()), 4), 'sigma': round(float(e9[m].std()), 4)}
    eo = crossfit(GB, X9, keep=pid_all != code, rows=rws) - yt[rws]
    lopo[nm] = {'sigma_in': species[nm]['sigma'], 'bias_in': species[nm]['bias'],
                'sigma_held_out': round(float(eo.std()), 4), 'bias_held_out': round(float(eo.mean()), 4)}
print('per primary:', species); print('leave-one-primary-out:', lopo)
out['species'] = species; out['lopo'] = lopo

R['crossfit_secondary'] = out
json.dump(R, open(OUT, 'w'), indent=2)
print('extended %s' % OUT)
