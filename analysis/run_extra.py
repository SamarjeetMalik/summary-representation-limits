"""Journal-version analyses: cut sweep, per-species breakdown, conformal calibration.

Extends results.json rather than replacing it, so the slow estimator-family sweep
in run_all.py does not need re-running.

Run after run_all.py:  python analysis/run_extra.py
"""
import json, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as GBR
from sklearn.model_selection import train_test_split

import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results.json')
R = json.load(open(OUT))

print('loading...')
d = C.load_root(); c = C.load_csv()
X = C.build_features(d); y = d['y']
r_idx, c_idx = C.match(d, c)
test = np.zeros(len(y), bool); test[r_idx] = True
train_all = np.where(C.train_mask(d, test))[0]

y_te, age_te, pid_te = y[r_idx], d['Age'][r_idx], d['PrimaryID'][r_idx]


def gbm(seed, tr, **kw):
    return GBR(max_iter=600, learning_rate=.06, random_state=seed, **kw).fit(X[tr], y[tr])


# ------------------------------------------------------------ 1. cut sweep
print('\n== cut-boundary sweep, 8 seeds ==')
CUTS = [('no cut', lambda a: np.ones(len(a), bool)),
        ('0.90-1.15 (standard)', lambda a: (a >= .90) & (a <= 1.15)),
        ('0.90-1.25', lambda a: (a >= .90) & (a <= 1.25)),
        ('0.90-1.40', lambda a: (a >= .90) & (a <= 1.40)),
        ('s >= 1.15 only', lambda a: a >= 1.15)]
acc = {k: {'sigma': [], 'kept': []} for k, _ in CUTS}
for s in C.SEEDS:
    p = gbm(s, train_all).predict(X[r_idx])
    for name, fn in CUTS:
        m = fn(age_te)
        acc[name]['sigma'].append(C.sigma(p[m], y_te[m]))
        acc[name]['kept'].append(100 * m.mean())
R['cut_sweep'] = {}
print('  %-22s %7s %9s %22s' % ('selection', 'kept', 'sigma', '95% CI'))
for name, _ in CUTS:
    v = acc[name]['sigma']; lo, hi = C.bootstrap_ci(v)
    R['cut_sweep'][name] = {'kept': round(float(np.mean(acc[name]['kept'])), 1),
                            'sigma': round(float(np.mean(v)), 4),
                            'ci': [round(lo, 4), round(hi, 4)], 'seeds': len(v)}
    print('  %-22s %6.1f%% %9.4f    [%.4f, %.4f]'
          % (name, np.mean(acc[name]['kept']), np.mean(v), lo, hi))

# --------------------------------------------------- 2. per-species breakdown
print('\n== per-species, split at s = 1.15 ==')
p42 = gbm(42, train_all).predict(X[r_idx])
R['species'] = []
print('  %-11s %8s %10s %10s %8s %10s' % ('species', 'n', 's<1.15', 's>=1.15', 'delta', 'frac old'))
for code, nm in C.SPECIES.items():
    m = pid_te == code
    lo_m, hi_m = m & (age_te < 1.15), m & (age_te >= 1.15)
    if lo_m.sum() < 60 or hi_m.sum() < 60:
        print('  %-11s SKIPPED (n_low=%d, n_high=%d)' % (nm, lo_m.sum(), hi_m.sum()))
        R['species'].append({'name': nm, 'n': int(m.sum()), 'skipped': True,
                             'n_low': int(lo_m.sum()), 'n_high': int(hi_m.sum())})
        continue
    a, b = C.sigma(p42[lo_m], y_te[lo_m]), C.sigma(p42[hi_m], y_te[hi_m])
    R['species'].append({'name': nm, 'n': int(m.sum()), 'skipped': False,
                         'n_low': int(lo_m.sum()), 'n_high': int(hi_m.sum()),
                         'sigma_young': round(a, 4), 'sigma_old': round(b, 4),
                         'delta_pct': round(100 * (b - a) / a, 1),
                         'frac_old': round(100 * hi_m.sum() / m.sum(), 1)})
    print('  %-11s %8d %10.4f %10.4f %+7.1f%% %9.1f%%'
          % (nm, m.sum(), a, b, 100 * (b - a) / a, 100 * hi_m.sum() / m.sum()))

pooled_y = C.sigma(p42[age_te < 1.15], y_te[age_te < 1.15])
pooled_o = C.sigma(p42[age_te >= 1.15], y_te[age_te >= 1.15])
R['pooled_split'] = {'sigma_young': round(pooled_y, 4), 'sigma_old': round(pooled_o, 4),
                     'delta_pct': round(100 * (pooled_o - pooled_y) / pooled_y, 1)}
within = [s['delta_pct'] for s in R['species'] if not s['skipped']]
R['within_species_range'] = [round(max(within), 1), round(min(within), 1)]
print('  %-11s %8s %10.4f %10.4f %+7.1f%%  <-- POOLED (mixture-inflated)'
      % ('all', len(p42), pooled_y, pooled_o, R['pooled_split']['delta_pct']))
print('  within-species deltas span %.1f%% to %.1f%%' % tuple(R['within_species_range']))

# ------------------------------------- 3. conformalised quantile regression
print('\n== conformalised quantile regression (CQR) ==')
tr, cal = train_test_split(train_all, test_size=.25, random_state=42)
BANDS = [(0.68, .16, .84), (0.90, .05, .95)]
R['calibration'] = {}
BINS = [(.8, .9), (.9, 1.0), (1.0, 1.1), (1.1, 1.15), (1.15, 1.25), (1.25, 1.4)]
for nominal, qlo, qhi in BANDS:
    mlo = gbm(42, tr, loss='quantile', quantile=qlo)
    mhi = gbm(42, tr, loss='quantile', quantile=qhi)
    # conformity scores on the calibration split
    e = np.maximum(mlo.predict(X[cal]) - y[cal], y[cal] - mhi.predict(X[cal]))
    Q = float(np.quantile(e, min(1.0, nominal * (1 + 1 / len(cal)))))
    lo_p, hi_p = mlo.predict(X[r_idx]) - Q, mhi.predict(X[r_idx]) + Q
    raw_lo, raw_hi = mlo.predict(X[r_idx]), mhi.predict(X[r_idx])
    rows = []
    print('  nominal %.0f%%  (conformal offset Q = %.4f)' % (100 * nominal, Q))
    print('    %-13s %8s %10s %10s %9s' % ('age bin', 'n', 'raw cov', 'CQR cov', 'width'))
    for a1, a2 in BINS:
        m = (age_te >= a1) & (age_te < a2)
        if m.sum() < 150:
            continue
        raw = float(((y_te[m] >= raw_lo[m]) & (y_te[m] <= raw_hi[m])).mean())
        cqr = float(((y_te[m] >= lo_p[m]) & (y_te[m] <= hi_p[m])).mean())
        rows.append({'bin': '%.2f-%.2f' % (a1, a2), 'n': int(m.sum()),
                     'raw': round(100 * raw, 1), 'cqr': round(100 * cqr, 1),
                     'width': round(float((hi_p[m] - lo_p[m]).mean()), 4)})
        flag = '' if abs(cqr - nominal) <= .02 else '  *'
        print('    %-13s %8d %9.1f%% %9.1f%% %9.4f%s'
              % (rows[-1]['bin'], m.sum(), 100 * raw, 100 * cqr, rows[-1]['width'], flag))
    overall_raw = float(((y_te >= raw_lo) & (y_te <= raw_hi)).mean())
    overall_cqr = float(((y_te >= lo_p) & (y_te <= hi_p)).mean())
    R['calibration']['%d' % (100 * nominal)] = {
        'nominal': 100 * nominal, 'Q': round(Q, 4), 'bins': rows,
        'overall_raw': round(100 * overall_raw, 1), 'overall_cqr': round(100 * overall_cqr, 1)}
    print('    overall: raw %.1f%%  ->  CQR %.1f%%' % (100 * overall_raw, 100 * overall_cqr))

json.dump(R, open(OUT, 'w'), indent=2)
print('\nextended %s' % OUT)
