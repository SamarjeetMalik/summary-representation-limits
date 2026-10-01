"""Robustness of the age-cut result to how an analyst would actually apply the cut.

run_extra.py scores ONE model, trained on all ages, on each selected subset. A real
analysis that applies the standard 0.90 <= s <= 1.15 cut would also train (or
calibrate) on the accepted events only, and would care about composition bias as
much as resolution, since the point of a quality cut is a mass-independent energy
scale. This script adds both views:

  1. train-within-selection: retrain gradient boosting on the accepted training events
     and score on the accepted test events;
  2. per-species mean bias with and without the cut, and its spread across species
     (the quantity a composition-independent energy estimator is meant to minimise).

Extends results.json.  Run:  python analysis/run_selection_robustness.py
"""
import json, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as GBR

import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results.json')
R = json.load(open(OUT))

print('loading...')
d = C.load_root(); c = C.load_csv()
X = C.build_features(d); y = d['y']
r_idx, _ = C.match(d, c)
test = np.zeros(len(y), bool); test[r_idx] = True
age, pid = d['Age'], d['PrimaryID']

CUTS = {'none': lambda a: np.ones(len(a), bool),
        'standard': lambda a: (a >= .90) & (a <= 1.15),
        'wide': lambda a: (a >= .90) & (a <= 1.40)}
SEEDS = C.SEEDS[:4]

out = {}
print('\n  %-9s %-22s %8s %8s %12s' % ('cut', 'training', 'kept', 'sigma', 'bias spread'))
for name, fn in CUTS.items():
    sel = fn(age)
    te = np.where(test & sel)[0]
    for mode, tr in (('all ages', np.where(~test)[0]),
                     ('within selection', np.where(~test & sel)[0])):
        sig, spread, per = [], [], []
        for s in SEEDS:
            p = GBR(max_iter=600, learning_rate=.06, random_state=s).fit(X[tr], y[tr]).predict(X[te])
            sig.append(C.sigma(p, y[te]))
            b = {C.SPECIES[k]: C.bias(p[pid[te] == k], y[te][pid[te] == k]) for k in C.SPECIES}
            per.append(b); spread.append(max(b.values()) - min(b.values()))
        key = '%s|%s' % (name, mode)
        out[key] = {'kept': round(100 * sel[test].mean(), 1), 'n_test': int(len(te)),
                    'n_train': int(len(tr)), 'sigma': round(float(np.mean(sig)), 4),
                    'bias_spread': round(float(np.mean(spread)), 4),
                    'bias_by_species': {k: round(float(np.mean([q[k] for q in per])), 4)
                                        for k in per[0]}}
        print('  %-9s %-22s %7.1f%% %8.4f %12.4f'
              % (name, mode, out[key]['kept'], out[key]['sigma'], out[key]['bias_spread']))

R['selection_robustness'] = {'seeds': len(SEEDS), 'rows': out}
json.dump(R, open(OUT, 'w'), indent=2)
print('\nextended %s' % OUT)
