"""Shower-disjoint evaluation by cross-fitting.

The evaluation events are a random holdout of events, and each CORSIKA shower is thrown several
times, so most evaluation showers have copies in the training complement. Dropping those copies
from training removes the leak but leaves a biased training sample (a shower with k copies
survives with probability 0.8^k; see common.train_mask). The unbiased protocol is a cross-fit:
showers are assigned to five folds, a model for fold k is trained on every event whose shower
is outside fold k, and it predicts the evaluation events inside fold k. Every evaluation event
is then predicted by a model that never saw a copy of its shower, the training sets are random
samples of showers of the original size, and the evaluation population is unchanged.

Recomputed this way: the estimator table, the information ladder with its permutation control,
the selection sweep with retraining, and (from the fold runs of run_rendering.py, if present)
the networks on rendered input. Intervals resample showers.

Extends results.json.  Run:  python analysis/run_shower_crossfit.py
"""
import glob, json, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as GBR, RandomForestRegressor as RF
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results.json')
R = json.load(open(OUT))
KF, NB = 5, 1000
NJOBS = int(os.environ.get('NJOBS', max(1, (os.cpu_count() or 4) - 1)))
rng = np.random.default_rng(0)

d = C.load_root(); c = C.load_csv(); y = d['y']; X9 = C.build_features(d)
r_idx, _ = C.match(d, c)
yt = y[r_idx]; age_t = d['Age'][r_idx]; pid_t = d['PrimaryID'][r_idx]
fold = C.shower_folds(d, KF); grp = C.shower_ids(d)
ftest = fold[r_idx]

# shower-level bootstrap rows over the evaluation events, shared by every comparison (paired)
ug, inv = np.unique(grp[r_idx], return_inverse=True)
order = np.argsort(inv, kind='stable'); starts = np.r_[0, np.cumsum(np.bincount(inv))]
BOOT = [np.concatenate([order[starts[i]:starts[i + 1]] for i in rng.integers(0, len(ug), len(ug))])
        for _ in range(NB)]


def crossfit(make, X, keep=None):
    """Residuals on the evaluation events; `keep` restricts the TRAINING events (a selection)."""
    p = np.empty(len(r_idx))
    for k in range(KF):
        tr = fold != k if keep is None else (fold != k) & keep
        p[ftest == k] = make().fit(X[tr], y[tr]).predict(X[r_idx[ftest == k]])
    return p - yt


def sd_ci(e, rows=None):
    e = e if rows is None else e
    b = [e[r].std() for r in BOOT]
    return round(float(e.std()), 4), [round(float(v), 4) for v in np.percentile(b, [2.5, 97.5])]


def paired(e_new, e_old, mask=None):
    if mask is None:
        dl = [e_new[r].std() - e_old[r].std() for r in BOOT]
        pt = e_new.std() - e_old.std()
    else:                                   # e_new scored on a subset, e_old on everything
        dl = [e_new[r][mask[r]].std() - e_old[r].std() for r in BOOT]
        pt = e_new[mask].std() - e_old.std()
    return round(float(pt), 5), [round(float(v), 5) for v in np.percentile(dl, [2.5, 97.5])]


out = {'folds': KF, 'n_showers_eval': int(len(ug)),
       'train_per_fold': [int((fold != k).sum()) for k in range(KF)]}

# why not simply drop the copies: the surviving training events are a biased sample
test = np.zeros(len(y), bool); test[r_idx] = True
copies = np.bincount(grp)[grp]
excl = ~np.isin(grp, np.unique(grp[r_idx]))
out['exclusion_bias'] = {
    'n_train': int(excl.sum()),
    'mean_copies': {'evaluation': round(float(copies[test].mean()), 2), 'excluded_train': round(float(copies[excl].mean()), 2)},
    'proton_frac': {'evaluation': round(float((d['PrimaryID'][test] == 14).mean()), 3),
                    'excluded_train': round(float((d['PrimaryID'][excl] == 14).mean()), 3)},
    'gb_sigma_excluded_train': R['shower_groups']['gb_sigma']['shower_disjoint']}
print('exclusion bias:', out['exclusion_bias'])

# ------------------------------------------------------------------ estimators
GB = lambda s=42: GBR(max_iter=600, learning_rate=.06, random_state=s)
MODELS = {'Ridge': lambda s=42: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
          'RandomForest': lambda s=42: RF(n_estimators=60, min_samples_leaf=25, max_depth=18, n_jobs=NJOBS,
                                          random_state=s),
          'GradBoost': GB,
          'MLP': lambda s=42: make_pipeline(StandardScaler(), MLPRegressor(
              hidden_layer_sizes=(64, 64), max_iter=120, random_state=s, early_stopping=True))}
SEEDED = {'GradBoost': C.SEEDS[:5], 'MLP': C.SEEDS[:5]}        # as in the random-split column: mean of five seeds
E_ = {}
out['estimators'] = {}
for k, mk in MODELS.items():
    E_[k] = crossfit(mk, X9)                                    # seed 42: used for intervals and paired tests
    per = [float(E_[k].std())] + [float(crossfit(lambda s=s: mk(s), X9).std()) for s in SEEDED.get(k, [])[1:]]
    _, ci = sd_ci(E_[k])
    s = round(float(np.mean(per)), 4)
    out['estimators'][k] = {'sigma': s, 'seed_sd': round(float(np.std(per)), 4), 'n_seeds': len(per), 'ci': ci,
                            'random_split': R['estimators'][k]['sigma'],
                            'leak': round(s - R['estimators'][k]['sigma'], 4)}   # cross-fit minus random split
    print('%-13s cross-fit %.4f (sd %.4f over %d seeds) %s   random split %.4f'
          % (k, s, np.std(per), len(per), ci, R['estimators'][k]['sigma']))
e_gb = E_['GradBoost']

# excess over the evaluation-population floor estimate, jointly bootstrapped by run_tmlr_checks
if 'tmlr_checks' in R:
    out['gb_excess_over_floor_pct'] = {
        k: round(100 * (e_gb.std() - v['floor']) / v['floor'], 2)
        for k, v in R['tmlr_checks']['floors'].items()}

# ------------------------------------------------------------------ ladder
th = d['Theta1']; mu16 = d['MuonsLarge'].astype(float)
cols = {'phi': [d['log_Ne'], d['Age'], d['NKGX'], d['NKGY']], 'theta': [th, np.cos(np.radians(th))],
        'rc': [np.hypot(d['NKGX'], d['NKGY'])], 'mu': [d['log_Nmu']], 'ratio': [d['log_Nmu'] - d['log_Ne']]}
mat = lambda ks: np.column_stack([v for k in ks for v in cols[k]])
rungs = [('A', ['phi']), ('B', ['phi', 'theta']), ('C', ['phi', 'theta', 'rc']),
         ('D', ['phi', 'theta', 'rc', 'mu']), ('E', ['phi', 'theta', 'rc', 'mu', 'ratio'])]
ladder, prev = [], None
for key, ks in rungs:
    e = crossfit(GB, mat(ks))
    row = {'rung': key, 'sigma': round(float(e.std()), 4)}
    if prev is not None:
        row['delta'], row['delta_ci'] = paired(e, prev)
    ladder.append(row); prev = e
e_E = prev
lmu = np.log10(1.0 + mu16)
e_F = crossfit(GB, np.column_stack([mat(rungs[-1][1]), lmu]))
dl, ci = paired(e_F, e_E)
ladder.append({'rung': 'F', 'sigma': round(float(e_F.std()), 4), 'delta': dl, 'delta_ci': ci})
perm = np.array([np.random.default_rng(i).permutation(16) for i in range(len(y))])
e_P = crossfit(GB, np.column_stack([mat(rungs[-1][1]), np.take_along_axis(lmu, perm, 1)]))
dl, ci = paired(e_F, e_P)
ladder.append({'rung': "F'", 'sigma': round(float(e_P.std()), 4), 'delta': dl, 'delta_ci': ci})
out['ladder'] = ladder
for r in ladder:
    print('ladder %-2s %.4f  %s' % (r['rung'], r['sigma'], (r.get('delta'), r.get('delta_ci'))))

# ------------------------------------------------------------------ selection
std = (age_t >= .90) & (age_t <= 1.15)
sel_all = (d['Age'] >= .90) & (d['Age'] <= 1.15)
pt, ci = paired(e_gb, e_gb, std)
e_within = crossfit(GB, X9, keep=sel_all)
spread = lambda e, m: (lambda b: max(b) - min(b))([float(e[m & (pid_t == k)].mean()) for k in C.SPECIES])
allm = np.ones(len(yt), bool)
out['selection'] = {'kept_pct': round(100 * std.mean(), 1), 'sigma_none': round(float(e_gb.std()), 4),
                    'sigma_standard': round(float(e_gb[std].std()), 4), 'penalty': pt, 'penalty_ci': ci,
                    'sigma_standard_trained_within': round(float(e_within[std].std()), 4),
                    'bias_spread_none': round(spread(e_gb, allm), 4),
                    'bias_spread_standard_within': round(spread(e_within, std), 4)}
print('selection:', out['selection'])

# ------------------------------------------------------------------ networks on rendered input
files = sorted(glob.glob(os.path.join(HERE, 'folds', 'render_fold*.npz')))
if len(files) == KF:
    pos = {int(e): i for i, e in enumerate(r_idx)}
    P = {k: np.full(len(r_idx), np.nan) for k in ('A_scalars4', 'B_rendered', 'C_rendered_muon', 'D_nine')}
    for fn in files:
        z = np.load(fn)
        at = np.array([pos[int(i)] for i in z['idx']])
        assert np.allclose(z['y'], yt[at])
        for k in P:
            P[k][at] = z[k]
    assert not any(np.isnan(v).any() for v in P.values()), 'fold files do not cover the evaluation events'
    Er = {k: v - yt for k, v in P.items()}
    rd = {k: round(float(e.std()), 4) for k, e in Er.items()}
    rd['B_minus_A'], rd['B_minus_A_ci'] = paired(Er['B_rendered'], Er['A_scalars4'])
    rd['random_split'] = {k: R['rendering'][k] for k in P}
    out['rendering'] = rd
    print('rendering (cross-fit, 60k training events per fold):', rd)
else:
    print('folds/render_fold*.npz: %d of %d present; rendering cross-fit skipped' % (len(files), KF))

R['shower_crossfit'] = out
json.dump(R, open(OUT, 'w'), indent=2)
print('extended %s' % OUT)
