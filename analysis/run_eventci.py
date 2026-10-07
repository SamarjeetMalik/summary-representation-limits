"""Event-level bootstrap confidence intervals for the cut sweep.

The CIs in run_extra.py resample across model SEEDS on a fixed evaluation set, so
they measure seed stability, not sampling uncertainty -- which is why they come out
absurdly tight. What a reader needs is the uncertainty from having a finite test
sample. This script resamples EVENTS with replacement and reports both, so the
manuscript can state plainly which is which.

Run after run_extra.py:  python analysis/run_eventci.py
"""
import json, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as GBR

import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results.json')
R = json.load(open(OUT))

d = C.load_root(); c = C.load_csv()
X = C.build_features(d); y = d['y']
r_idx, _ = C.match(d, c)
test = np.zeros(len(y), bool); test[r_idx] = True
train_all = np.where(C.train_mask(d, test))[0]
y_te, age_te = y[r_idx], d['Age'][r_idx]

print('fitting 3 seeds for the event-bootstrap...')
P = [GBR(max_iter=600, learning_rate=.06, random_state=s).fit(X[train_all], y[train_all])
     .predict(X[r_idx]) for s in (42, 7, 13)]

CUTS = [('no cut', lambda a: np.ones(len(a), bool)),
        ('0.90-1.15 (standard)', lambda a: (a >= .90) & (a <= 1.15)),
        ('0.90-1.25', lambda a: (a >= .90) & (a <= 1.25)),
        ('0.90-1.40', lambda a: (a >= .90) & (a <= 1.40)),
        ('s >= 1.15 only', lambda a: a >= 1.15)]

rng = np.random.default_rng(0)
NB = 2000
print('\n%-22s %9s %24s %12s' % ('selection', 'sigma', '95% CI (events)', 'seed sd'))
for name, fn in CUTS:
    m = fn(age_te)
    idx = np.where(m)[0]
    sig_seeds = [float((p[m] - y_te[m]).std()) for p in P]
    resid = P[0][m] - y_te[m]
    boot = np.empty(NB)
    for b in range(NB):
        take = rng.integers(0, len(resid), len(resid))
        boot[b] = resid[take].std()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    R['cut_sweep'][name]['ci_events'] = [round(float(lo), 4), round(float(hi), 4)]
    R['cut_sweep'][name]['seed_sd'] = round(float(np.std(sig_seeds)), 5)
    print('%-22s %9.4f    [%.4f, %.4f]   %10.5f'
          % (name, np.mean(sig_seeds), lo, hi, np.std(sig_seeds)))

# does the standard cut differ from no cut by more than sampling noise?
m_std = CUTS[1][1](age_te)
r_all, r_std = P[0] - y_te, (P[0] - y_te)[m_std]
diff = np.empty(NB)
for b in range(NB):
    a = r_all[rng.integers(0, len(r_all), len(r_all))].std()
    s_ = r_std[rng.integers(0, len(r_std), len(r_std))].std()
    diff[b] = s_ - a
lo, hi = np.percentile(diff, [2.5, 97.5])
R['cut_penalty'] = {'delta': round(float(diff.mean()), 4),
                    'ci': [round(float(lo), 4), round(float(hi), 4)]}
print('\nstandard cut minus no cut:  %+.4f  95%% CI [%+.4f, %+.4f]'
      % (diff.mean(), lo, hi))
print('  -> the penalty %s zero' % ('EXCLUDES' if lo > 0 else 'includes'))

json.dump(R, open(OUT, 'w'), indent=2)
print('\nupdated %s' % OUT)
