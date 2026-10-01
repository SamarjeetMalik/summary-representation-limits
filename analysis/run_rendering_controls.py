"""Tabular controls for rows C and D of the rendering table.

Row C (CNN on the rendered image plus the 16 muon counts) has as its information
content (phi, M16). Proposition 1 applied to (phi, M16) says it cannot beat a tabular
model on those same inputs, so the matching control is gradient boosting on
(phi, M16), with the same training subsample. Row D (nine features) additionally
contains the zenith angle; adding theta to (phi, M16) tests whether the C-D gap is the
angle. (An earlier draft attributed that gap to the telescope reporting a track count
rather than a muon size; the nine features use the same summed track count, so that
explanation was wrong.)

Extends results.json['rendering'].  Run after run_rendering.py.
"""
import json, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as GBR

import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results.json')
R = json.load(open(OUT))

d = C.load_root(); c = C.load_csv(); y = d['y']
r_idx, _ = C.match(d, c)
test = np.zeros(len(y), bool); test[r_idx] = True
pool = np.where(~test)[0]
tr = np.random.default_rng(42).choice(pool, R['rendering']['n_train'], replace=False)  # as run_rendering

PHI = np.column_stack([d['log_Ne'], d['Age'], d['NKGX'], d['NKGY']])
MU = np.log10(1.0 + d['MuonsLarge'].astype(float))
TH = d['Theta1'][:, None]
out = {}
for key, X in (('tab_phi_mu', np.column_stack([PHI, MU])),
               ('tab_phi_mu_theta', np.column_stack([PHI, MU, TH])),
               ('tab_phi_theta', np.column_stack([PHI, TH]))):
    p = GBR(max_iter=600, learning_rate=.06, random_state=42).fit(X[tr], y[tr]).predict(X[r_idx])
    out[key] = round(C.sigma(p, y[r_idx]), 4)
    out[key + '_bias'] = round(C.bias(p, y[r_idx]), 5)
    print('  %-18s sigma %.4f  bias %+.5f' % (key, out[key], out[key + '_bias']))
R['rendering'].update(out)
json.dump(R, open(OUT, 'w'), indent=2)
print('extended results.json[rendering]')
