"""Compute every number cited in the manuscript. Writes results.json.

All estimators are evaluated on the identical matched population; ML models are
trained on its complement. Run:  python analysis/run_all.py
"""
import json, os, warnings
import numpy as np
warnings.filterwarnings('ignore')

from sklearn.ensemble import HistGradientBoostingRegressor as GBR, RandomForestRegressor as RF
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import common as C

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')
R = {}

print('loading...')
d = C.load_root()
c = C.load_csv()
X = C.build_features(d)
y = d['y']
r_idx, c_idx = C.match(d, c)

test = np.zeros(len(y), bool); test[r_idx] = True
train = ~test
R['n_root'] = int(len(y))
R['n_csv'] = int(len(c['true']))
R['n_matched'] = int(len(r_idx))
R['pct_matched_of_csv'] = round(100 * len(r_idx) / len(c['true']), 2)
R['pct_of_parent'] = round(100 * len(r_idx) / len(y), 3)
R['n_train'] = int(train.sum())
R['target_spread'] = round(float(y.std()), 4)
R['logE_min'] = round(float(y.min()), 4); R['logE_max'] = round(float(y.max()), 4)
R['theta_max'] = round(float(d['Theta1'].max()), 3)
R['logNe_min'] = round(float(d['log_Ne'].min()), 3)
R['logNe_max'] = round(float(d['log_Ne'].max()), 3)
print('  %d ROOT, %d CSV, %d matched (%.2f%% of CSV, %.3f%% of parent), train on %d'
      % (R['n_root'], R['n_csv'], R['n_matched'], R['pct_matched_of_csv'],
         R['pct_of_parent'], R['n_train']))

y_test = y[r_idx]
age_test = d['Age'][r_idx]
pid_test = d['PrimaryID'][r_idx]

MODELS = {
    'Ridge':        lambda s: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    'RandomForest': lambda s: RF(n_estimators=60, min_samples_leaf=25, max_depth=18,
                                 n_jobs=1, random_state=s),
    'GradBoost':    lambda s: GBR(max_iter=600, learning_rate=.06, random_state=s),
    'MLP':          lambda s: make_pipeline(StandardScaler(),
                        MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=120,
                                     random_state=s, early_stopping=True)),
}

# ---------------------------------------------------------------- Table 1
print('\n== estimator families, identical population ==')
R['estimators'] = {}
preds = {}
for name, mk in MODELS.items():
    sig = []
    for s in C.SEEDS[:5]:
        p = mk(s).fit(X[train], y[train]).predict(X[r_idx])  # r_idx order, aligned with y_test
        sig.append(C.sigma(p, y_test))
        if s == 42:
            preds[name] = p
    R['estimators'][name] = {'sigma': round(float(np.mean(sig)), 4),
                             'sd': round(float(np.std(sig)), 4), 'seeds': 5}
    print('  %-14s %.4f +/- %.4f' % (name, np.mean(sig), np.std(sig)))
    # Guard: any estimator worse than predicting the sample mean indicates a
    # train/test alignment fault, not a modelling result. Fail loudly.
    assert np.mean(sig) < 0.9 * y.std(), (
        '%s sigma=%.4f exceeds 0.9x target spread %.4f -- check index alignment'
        % (name, np.mean(sig), y.std()))

preds['NKG'] = c['nkg'][c_idx]
preds['CNN/ViT'] = c['cnnvit'][c_idx]
for name in ('NKG', 'CNN/ViT'):
    R['estimators'][name] = {'sigma': round(C.sigma(preds[name], y_test), 4),
                             'bias': round(C.bias(preds[name], y_test), 5),
                             'seeds': 0}
    print('  %-14s %.4f  (archived, same events)' % (name, R['estimators'][name]['sigma']))

# ---------------------------------------------------------------- Table 2
print('\n== muon spatial pattern ablation ==')
mu16 = d['MuonsLarge'].astype(float)
shape16 = mu16 / np.maximum(d['mu_total'], 1)[:, None]
R['muon'] = {}
for lbl, XX in (('nine_feature', X),
                ('plus_raw16', np.column_stack([X, mu16])),
                ('plus_shape16', np.column_stack([X, shape16]))):
    sig = [C.sigma(GBR(max_iter=600, learning_rate=.06, random_state=s)
                   .fit(XX[train], y[train]).predict(XX[r_idx]), y_test)
           for s in C.SEEDS[:5]]
    R['muon'][lbl] = {'sigma': round(float(np.mean(sig)), 4), 'sd': round(float(np.std(sig)), 4)}
    print('  %-14s %.4f +/- %.4f' % (lbl, np.mean(sig), np.std(sig)))
b = R['muon']['nine_feature']['sigma']
for k in ('plus_raw16', 'plus_shape16'):
    R['muon'][k]['delta_pct'] = round(100 * (R['muon'][k]['sigma'] - b) / b, 2)

# ---------------------------------------------------------------- Table 3
print('\n== degeneracy: recoverable from four summary scalars ==')
S4 = np.column_stack([d['log_Ne'][r_idx], d['Age'][r_idx],
                      d['NKGX'][r_idx], d['NKGY'][r_idx]])
from sklearn.model_selection import train_test_split
R['degeneracy'] = {}
for lbl, target in (('NKG', preds['NKG']), ('CNN/ViT', preds['CNN/ViT']), ('true_energy', y_test)):
    a, bq, ta, tb = train_test_split(S4, target, test_size=.25, random_state=0)
    m = GBR(max_iter=400, random_state=0).fit(a, ta)
    p = m.predict(bq)
    R['degeneracy'][lbl] = {'r2': round(C.r2(p, tb), 4), 'resid_sd': round(float((tb - p).std()), 4)}
    print('  %-12s R2=%.4f  resid sd=%.4f' % (lbl, R['degeneracy'][lbl]['r2'],
                                              R['degeneracy'][lbl]['resid_sd']))

# ---------------------------------------------------------------- Table 4
print('\n== mechanism: corr(age, logE) vs shower-size window ==')
R['mechanism'] = []
for a1, a2 in ((4.80, 4.90), (4.95, 5.05), (5.10, 5.20)):
    m = (d['log_Ne'] >= a1) & (d['log_Ne'] < a2)
    R['mechanism'].append({'window': '%.2f-%.2f' % (a1, a2), 'n': int(m.sum()),
                           'corr': round(float(np.corrcoef(d['Age'][m], y[m])[0, 1]), 3)})
    print('  logNe %.2f-%.2f  n=%6d  corr=%+.3f' % (a1, a2, m.sum(), R['mechanism'][-1]['corr']))
R['corr_whole'] = round(float(np.corrcoef(d['Age'], y)[0, 1]), 3)
print('  whole sample              corr=%+.3f' % R['corr_whole'])

# ------------------------------------------------- age curves (Fig 1) + selection
print('\n== sigma vs age, per estimator ==')
BINS = [(.8, .9), (.9, 1.0), (1.0, 1.1), (1.1, 1.15), (1.15, 1.25), (1.25, 1.4)]
R['age_curves'] = {}
for name, p in preds.items():
    row = []
    for a1, a2 in BINS:
        m = (age_test >= a1) & (age_test < a2)
        row.append({'bin': '%.2f-%.2f' % (a1, a2), 'n': int(m.sum()),
                    'sigma': round(C.sigma(p[m], y_test[m]), 4) if m.sum() > 150 else None})
    R['age_curves'][name] = row
    first = row[0]['sigma']; last = row[-1]['sigma']
    R['estimators'][name]['age_trend_pct'] = round(100 * (last - first) / first, 1)
    print('  %-12s %+.1f%% across age' % (name, R['estimators'][name]['age_trend_pct']))

acc = (age_test >= .9) & (age_test <= 1.15)
R['selection'] = {
    'no_cut':   {'kept': 100.0, 'sigma': round(C.sigma(preds['GradBoost'], y_test), 4)},
    'sdrn':     {'kept': round(100 * acc.mean(), 1),
                 'sigma': round(C.sigma(preds['GradBoost'][acc], y_test[acc]), 4)},
}
print('\n== selection ==')
print('  no cut  %.4f   |   SDRN 0.90-1.15 (%.1f%% kept)  %.4f'
      % (R['selection']['no_cut']['sigma'], R['selection']['sdrn']['kept'],
         R['selection']['sdrn']['sigma']))

json.dump(R, open(OUT, 'w'), indent=2)
print('\nwrote %s' % OUT)
