"""Provenance of the archived NKG baseline, and the 0.168 figure.

The archived table labels its `nkg_pred_logE` column an analytic NKG reconstruction.
This script establishes what it actually is: an exact linear function of
(log10 Ne, s), i.e. a two-parameter size-age energy calibration. It then refits that
calibration on the training complement and scores it on the matched population, which
must reproduce the archived sigma. Finally it scores the simpler calibrations that an
earlier draft's 0.168 baseline could have come from.

Extends results.json. Run after run_all.py:  python analysis/run_nkg_baseline.py
"""
import json, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression

import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results.json')
R = json.load(open(OUT))

print('loading...')
d = C.load_root(); c = C.load_csv()
y = d['y']
r_idx, c_idx = C.match(d, c)
test = np.zeros(len(y), bool); test[r_idx] = True
train = C.train_mask(d, test)
y_te = y[r_idx]
sec = 1.0 / np.cos(np.radians(d['Theta1']))

# 1. What the archived column is: regress it on the CSV's own (log Ne, s).
Xc = np.column_stack([c['log_Ne'], c['age']])
arch = LinearRegression().fit(Xc, c['nkg'])
arch_res = float(np.abs(c['nkg'] - arch.predict(Xc)).max())
print('\n== archived NKG column as a function of (log Ne, s) ==')
print('  coef %s  intercept %.4f  max |residual| %.1e'
      % (np.round(arch.coef_, 4), arch.intercept_, arch_res))
assert arch_res < 1e-9, 'archived NKG column is not an exact linear function of (log Ne, s)'

# 2. Refit that calibration out of sample; it must reproduce the archived sigma.
cal = {}
for key, cols in (('logNe', [d['log_Ne']]),
                  ('logNe_sec', [d['log_Ne'], sec]),
                  ('logNe_age', [d['log_Ne'], d['Age']]),
                  ('logNe_age_sec', [d['log_Ne'], d['Age'], sec])):
    X = np.column_stack(cols)
    m = LinearRegression().fit(X[train], y[train])
    p = m.predict(X[r_idx])
    cal[key] = {'sigma': round(C.sigma(p, y_te), 4), 'bias': round(C.bias(p, y_te), 5),
                'coef': [round(float(v), 4) for v in m.coef_],
                'intercept': round(float(m.intercept_), 4)}
    print('  refit %-14s sigma=%.4f  coef=%s' % (key, cal[key]['sigma'], cal[key]['coef']))

archived_sigma = R['estimators']['NKG']['sigma']
assert abs(cal['logNe_age']['sigma'] - archived_sigma) < 5e-4, (
    'refit (log Ne, s) calibration %.4f does not reproduce archived %.4f'
    % (cal['logNe_age']['sigma'], archived_sigma))

R['nkg_baseline'] = {
    'archived_coef': [round(float(v), 4) for v in arch.coef_],
    'archived_intercept': round(float(arch.intercept_), 4),
    'archived_max_residual': arch_res,
    'refit': cal,
    'reproduces_archived': True,
    'note': ('Archived nkg_pred_logE is exactly linear in (log10 Ne, s): a learned '
             'two-parameter calibration, not a non-learned analytic reconstruction. '
             'No calibration on this sample gives 0.168; see KCDC arm.'),
}
json.dump(R, open(OUT, 'w'), indent=2)
print('\nextended %s' % OUT)
