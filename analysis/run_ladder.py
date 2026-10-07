"""The information ladder, paired-bootstrap differences, and the muon permutation test.

1. Ladder: gradient boosting on nested feature sets, each step adding one summary quantity,
       A  phi = (log10 Ne, s, Xc, Yc)
       B  A + zenith angle (theta, cos theta)
       C  B + core distance R_c
       D  C + log10 total muons
       E  D + muon ratio            (= the nine-feature summary)
       F  E + 16 module counts
   with sigma, bias and the paired-bootstrap difference to the previous rung on the identical
   test events (95% CI). Seeds: 3 per rung (seed spread is ~1e-5 for gradient boosting).
2. Permutation test: F with the 16 counts permuted within each event (total preserved, spatial
   arrangement destroyed). If the arrangement carried information, F would beat F_perm.
3. Paired CIs for the rendering comparisons, from the saved predictions: B - A (CNN on the
   rendered image vs four scalars), G - B (ViT vs CNN), C - C' (CNN vs tabular on phi + muons).

Extends results.json.  Run:  python analysis/run_ladder.py
"""
import json, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as GBR

import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results.json')
R = json.load(open(OUT))
NB = int(os.environ.get('NBOOT', '2000'))

d = C.load_root(); c = C.load_csv(); y = d['y']
r_idx, _ = C.match(d, c)
test = np.zeros(len(y), bool); test[r_idx] = True
tr = np.where(C.train_mask(d, test))[0]; yt = y[r_idx]

th = d['Theta1']; mu16 = d['MuonsLarge'].astype(float)
cols = {'phi': [d['log_Ne'], d['Age'], d['NKGX'], d['NKGY']],
        'theta': [th, np.cos(np.radians(th))],
        'rc': [np.hypot(d['NKGX'], d['NKGY'])],
        'mu': [d['log_Nmu']],
        'ratio': [d['log_Nmu'] - d['log_Ne']]}
rungs = [('A', 'four fit parameters', ['phi']),
         ('B', '+ zenith angle', ['phi', 'theta']),
         ('C', '+ core distance', ['phi', 'theta', 'rc']),
         ('D', '+ total muons', ['phi', 'theta', 'rc', 'mu']),
         ('E', '+ muon ratio (nine features)', ['phi', 'theta', 'rc', 'mu', 'ratio'])]

rng = np.random.default_rng(0)
BOOT = rng.integers(0, len(yt), size=(NB, len(yt)), dtype=np.int32)   # shared resamples: paired


def resid(X, seeds=(42, 7, 13)):
    """Mean-over-seeds residual vector on the test events."""
    return np.mean([GBR(max_iter=600, learning_rate=.06, random_state=s).fit(X[tr], y[tr])
                    .predict(X[r_idx]) for s in seeds], axis=0) - yt


def sd_boot(e):
    return e[BOOT].std(axis=1)


def paired(e_new, e_old):
    dlt = sd_boot(e_new) - sd_boot(e_old)
    lo, hi = np.percentile(dlt, [2.5, 97.5])
    return round(float(e_new.std() - e_old.std()), 5), [round(float(lo), 5), round(float(hi), 5)]


def mat(keys):
    return np.column_stack([v for k in keys for v in cols[k]])


ladder, prev = [], None
for key, label, ks in rungs:
    e = resid(mat(ks))
    row = {'rung': key, 'label': label, 'sigma': round(float(e.std()), 4),
           'bias': round(float(e.mean()), 5)}
    lo, hi = np.percentile(sd_boot(e), [2.5, 97.5]); row['ci'] = [round(float(lo), 4), round(float(hi), 4)]
    if prev is not None:
        row['delta'], row['delta_ci'] = paired(e, prev)
    ladder.append(row); prev = e
    print('  %s %-30s sigma %.4f  %s' % (key, label, row['sigma'],
          'delta %+.5f [%+.5f, %+.5f]' % (row['delta'], *row['delta_ci']) if 'delta' in row else ''))
eE = prev

XE = mat(['phi', 'theta', 'rc', 'mu', 'ratio'])
eF = resid(np.column_stack([XE, mu16]))
dF, ciF = paired(eF, eE)
ladder.append({'rung': 'F', 'label': '+ 16 module counts', 'sigma': round(float(eF.std()), 4),
               'bias': round(float(eF.mean()), 5), 'delta': dF, 'delta_ci': ciF})
print('  F + 16 module counts              sigma %.4f  delta %+.5f [%+.5f, %+.5f]' % (eF.std(), dF, *ciF))

# permutation: shuffle the 16 counts within each event (total preserved)
perm = np.argsort(np.random.default_rng(1).random(mu16.shape), axis=1)
mu_perm = np.take_along_axis(mu16, perm, 1)
eP = resid(np.column_stack([XE, mu_perm]))
dP, ciP = paired(eF, eP)
perm_res = {'sigma_perm': round(float(eP.std()), 4), 'sigma_real': round(float(eF.std()), 4),
            'delta_real_minus_perm': dP, 'ci': ciP}
print('  permutation: real %.4f vs permuted %.4f, real - perm %+.5f [%+.5f, %+.5f]'
      % (eF.std(), eP.std(), dP, *ciP))

# rendering comparisons from saved predictions (same test events, same order)
P = np.load(os.path.join(HERE, 'preds_rendering.npz')); S = np.load(os.path.join(HERE, 'preds_sweep.npz'))
assert np.array_equal(P['r_idx'], r_idx)
sub = np.random.default_rng(42).choice(tr, R['rendering']['n_train'], replace=False)   # as run_rendering
PHI = mat(['phi'])
eA = GBR(max_iter=600, learning_rate=.06, random_state=42).fit(PHI[sub], y[sub]).predict(PHI[r_idx]) - yt
eCp = GBR(max_iter=600, learning_rate=.06, random_state=42).fit(
    np.column_stack([PHI, np.log10(1 + mu16)])[sub], y[sub]).predict(
    np.column_stack([PHI, np.log10(1 + mu16)])[r_idx]) - yt
eB = P['B_rendered'] - yt; eC = P['C_rendered_muon'] - yt; eG = S['G_vit_patches'] - yt
pairs = {}
for k, (a_, b_) in {'B_minus_A': (eB, eA), 'G_minus_B': (eG, eB), 'C_minus_Cprime': (eC, eCp)}.items():
    pairs[k] = dict(zip(('delta', 'ci'), paired(a_, b_)))
    print('  %-16s %+.5f [%+.5f, %+.5f]' % (k, pairs[k]['delta'], *pairs[k]['ci']))

R['ladder'] = {'rows': ladder, 'permutation': perm_res, 'render_pairs': pairs, 'n_boot': NB}
json.dump(R, open(OUT, 'w'), indent=2)
print('extended %s' % OUT)
