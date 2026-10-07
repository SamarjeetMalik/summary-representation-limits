"""Calibrate the degeneracy probe on models whose inputs are known.

The probe asks how much of a model's output m is a function of the fitted summary
parameters phi = (log10 Ne, s, Xc, Yc):

    R2_deg(phi) = 1 - E[(m - E[m | phi])^2] / Var(m),

with E[m | phi] replaced by a gradient-boosting regression evaluated out of fold,
which makes every value a lower bound on the population quantity.

On its own R2_deg < 1 is ambiguous: it can mean the model used a channel outside
phi, or only that its output is noisy. The calibration separates the two with the
INCREMENTAL probe, R2(phi + M) - R2(phi), where M is the only other measured channel
(the 16 muon-module counts) plus arrival direction:

    information outside phi   ->  R2(phi) < 1,  R2(phi + M) ~ 1
    noise                     ->  R2(phi) < 1,  R2(phi + M) < 1 by the same amount

The rows are models whose inputs we control, synthetic outputs with a known
construction, a retrain noise ceiling, and finally the archived model, whose inputs
are what the probe is meant to diagnose.

Needs analysis/preds_rendering.npz and preds_sweep.npz (written by run_rendering.py
and run_sweep.py). Extends results.json.  Run:  python analysis/run_probe_calibration.py
"""
import json, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as GBR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results.json')
R = json.load(open(OUT))

print('loading...')
d = C.load_root(); c = C.load_csv()
X9 = C.build_features(d); y = d['y']
r_idx, c_idx = C.match(d, c)
test = np.zeros(len(y), bool); test[r_idx] = True
train = C.train_mask(d, test)
y_te = y[r_idx]

PHI = np.column_stack([d['log_Ne'], d['Age'], d['NKGX'], d['NKGY']])
MU = np.log10(1.0 + d['MuonsLarge'].astype(float))
TH = d['Theta1'][:, None]
PROBE_SETS = {
    'phi':          PHI[r_idx],
    'phi_M':        np.column_stack([PHI, MU, TH])[r_idx],
}


def probe(m, feats, seed=0):
    """Out-of-fold R^2 of m on feats (5-fold gradient boosting)."""
    oof = np.empty_like(m)
    for a, b in KFold(5, shuffle=True, random_state=seed).split(feats):
        oof[b] = GBR(max_iter=400, random_state=seed).fit(feats[a], m[a]).predict(feats[b])
    return C.r2(oof, m)


def row(label, m, kind, extra=None):
    out = {'kind': kind}
    for k, F in PROBE_SETS.items():
        out[k] = round(probe(m, F), 4)
    out['increment'] = round(out['phi_M'] - out['phi'], 4)
    if extra is not None:
        out.update(extra)
    print('  %-34s R2(phi)=%.4f  R2(phi+M)=%.4f  increment=%+.4f'
          % (label, out['phi'], out['phi_M'], out['increment']))
    return out


P = np.load(os.path.join(HERE, 'preds_rendering.npz'))
S = np.load(os.path.join(HERE, 'preds_sweep.npz'))
assert np.array_equal(P['r_idx'], r_idx) and np.array_equal(S['r_idx'], r_idx), \
    'saved predictions are not aligned with the matched population'

rows = {}
print('\n== known inputs ==')
gb4 = GBR(max_iter=600, learning_rate=.06, random_state=42).fit(PHI[train], y[train]).predict(PHI[r_idx])
rows['gb_phi'] = row('GB on phi (positive control)', gb4, 'function of phi')
lin = LinearRegression().fit(PHI[train][:, :2], y[train]).predict(PHI[r_idx][:, :2])
rows['linear_Ne_s'] = row('linear (log Ne, s) calibration', lin, 'function of phi, exact')
rows['cnn_rendered'] = row('CNN, rendered image (B)', P['B_rendered'], 'function of phi')
rows['vit_rendered'] = row('ViT, rendered image (G)', S['G_vit_patches'], 'function of phi')
rows['cnn_rendered_muon'] = row('CNN, rendered + muons (C)', P['C_rendered_muon'], 'phi + muons')
gb9 = GBR(max_iter=600, learning_rate=.06, random_state=42).fit(X9[train], y[train]).predict(X9[r_idx])
rows['gb_nine'] = row('GB on nine features', gb9, 'phi + muon total + angle')
rows['true_energy'] = row('true energy (not a model)', y_te, 'reference')

print('\n== synthetic outputs with a known construction ==')
rng = np.random.default_rng(0)
noise = {}
for tau in (0.02, 0.05, 0.10):
    noise['%.2f' % tau] = row('GB on phi + N(0, %.2f)' % tau, gb4 + rng.normal(0, tau, len(gb4)),
                              'phi + pure noise')
mix = {}
mu_only = GBR(max_iter=400, random_state=0).fit(MU[train], y[train]).predict(MU[r_idx])
mu_only = (mu_only - mu_only.mean()) / mu_only.std() * gb4.std() + gb4.mean()
for lam in (0.1, 0.3, 0.5):
    mix['%.1f' % lam] = row('(1-%.1f) GB(phi) + %.1f u(M)' % (lam, lam),
                            (1 - lam) * gb4 + lam * mu_only, 'phi + muon information')

print('\n== retrain noise ceiling ==')
b1, b2 = P['B_rendered'], P['B_rendered_seed2']
ceiling = round(C.r2(LinearRegression().fit(b2[:, None], b1).predict(b2[:, None]), b1), 4)
print('  CNN (B) seed 42 predicted from seed 43: R2 = %.4f' % ceiling)

print('\n== the archived model ==')
arch = c['cnnvit'][c_idx]
rows['archived'] = row('archived CNN/ViT', arch, 'unknown')
# Leakage check: if the archived output carried label information beyond its inputs,
# adding the true energy to the probe features would raise R2 further.
te_ = round(probe(arch, np.column_stack([PROBE_SETS['phi_M'], y_te])), 4)
print('  archived, phi + M + true energy:   R2 = %.4f' % te_)
rows['archived']['phi_M_trueE'] = te_

R['probe_calibration'] = {
    'method': '5-fold out-of-fold gradient boosting on the matched population; '
              'phi = (log10 Ne, s, Xc, Yc); M = 16 log muon counts + zenith',
    'rows': rows, 'noise': noise, 'mix': mix, 'retrain_ceiling': ceiling,
}
json.dump(R, open(OUT, 'w'), indent=2)
print('\nextended %s' % OUT)
