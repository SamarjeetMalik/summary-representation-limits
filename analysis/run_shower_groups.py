"""Shower reuse: does throwing one CORSIKA shower at several core positions leak into the results?

Air-shower simulations routinely reuse each CORSIKA shower several times at random core
positions (the KCDC simulation manual documents 3 reuses per shower for KASCADE). Copies of
one shower share the primary energy, species and arrival direction, so a random train/test
split can place copies of the same shower on both sides. This script

  1. identifies showers in the GRAPES-3 sample as groups of identical (true energy, primary) and
     checks that groups are physical copies (shared direction, different core position);
  2. measures the leakage: gradient boosting on the nine features trained on a shower-disjoint
     training set versus a same-size random subset of the usual (leaky) complement;
  3. recomputes the Gamma-test Bayes floor using only neighbours from other showers.

Extends results.json.  Run:  python analysis/run_shower_groups.py
"""
import json, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as GBR
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import uproot

import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results.json')
R = json.load(open(OUT))
K = 10

d = C.load_root(); c = C.load_csv()
y = d['y']; X = C.build_features(d)
r_idx, _ = C.match(d, c)
test = np.zeros(len(y), bool); test[r_idx] = True

_, grp, cnt = np.unique(np.column_stack([d['PrimaryEnergy'], d['PrimaryID'].astype(float)]),
                        axis=0, return_inverse=True, return_counts=True)
grp = grp.ravel()
phi_az = uproot.concatenate([f + ':param' for f in C.ROOT_FILES], ['Phi1'], library='np')['Phi1']


def within_sd(v):
    s = np.zeros(cnt.size); s2 = np.zeros(cnt.size)
    np.add.at(s, grp, v); np.add.at(s2, grp, v * v)
    n = cnt.astype(float)
    return float(np.median(np.sqrt(np.maximum(s2 / n - (s / n) ** 2, 0))[cnt > 1]))


out = {'n_events': int(len(y)), 'n_showers': int(cnt.size),
       'mean_reuse': round(len(y) / cnt.size, 2),
       'within_sd': {k: round(within_sd(v.astype(float)), 4) for k, v in
                     (('theta', d['Theta1']), ('phi', phi_az), ('Xc', d['NKGX']),
                      ('logNe', d['log_Ne']), ('age', d['Age']))},
       'sample_sd': {k: round(float(np.std(v)), 4) for k, v in
                     (('theta', d['Theta1']), ('phi', phi_az), ('Xc', d['NKGX']),
                      ('logNe', d['log_Ne']), ('age', d['Age']))}}
tg = set(grp[r_idx])
out['test_showers'] = len(tg)
out['test_showers_in_train'] = int(len(tg & set(grp[~test])))
print('%d events, %d showers (%.2f per shower); %d of %d test showers also in training'
      % (out['n_events'], out['n_showers'], out['mean_reuse'], out['test_showers_in_train'], len(tg)))
print('within-shower median sd vs sample sd: %s' % {k: (out['within_sd'][k], out['sample_sd'][k])
                                                     for k in out['within_sd']})

tr_all = np.where(~test)[0]
tr_dis = tr_all[~np.isin(grp[tr_all], list(tg))]
sig = {}
for lab, seeds in (('leaky_same_size', range(5)), ('shower_disjoint', [0])):
    v = []
    for s in seeds:
        tr = (np.random.default_rng(s).choice(tr_all, len(tr_dis), replace=False)
              if lab == 'leaky_same_size' else tr_dis)
        p = GBR(max_iter=600, learning_rate=.06, random_state=42).fit(X[tr], y[tr]).predict(X[r_idx])
        v.append(C.sigma(p, y[r_idx]))
    sig[lab] = round(float(np.mean(v)), 4)
out['n_train_disjoint'] = int(len(tr_dis))
out['gb_sigma'] = sig
print('GB nine features: leaky (same size, 5 draws) %.4f | shower-disjoint %.4f (n=%d)'
      % (sig['leaky_same_size'], sig['shower_disjoint'], len(tr_dis)))

Z = StandardScaler().fit_transform(X)
dist, idx = NearestNeighbors(n_neighbors=40).fit(Z).kneighbors(Z)
same = grp[idx] == grp[:, None]
out['nn1_same_shower_frac'] = round(float(same[:, 1].mean()), 4)
keep = np.where(~same, np.arange(40)[None, :], 99)
keep[:, 0] = 99
order = np.sort(keep, axis=1)[:, :K]
assert (order < 99).all(), 'fewer than K other-shower neighbours for some events'
D2 = np.take_along_axis(dist, order, 1); I2 = np.take_along_axis(idx, order, 1)
g = 0.5 * ((y[:, None] - y[I2]) ** 2).mean(0); dl = (D2 ** 2).mean(0)
out['bayes_nine_other_showers'] = round(float(np.sqrt(max(np.polyfit(dl, g, 1)[1], 0))), 4)
print('Gamma floor, other-shower neighbours: %.4f (all neighbours: %.4f); 1st NN same shower %.1f%%'
      % (out['bayes_nine_other_showers'], R['bayes']['nine']['rms_floor'], 100 * out['nn1_same_shower_frac']))

R['shower_groups'] = out
json.dump(R, open(OUT, 'w'), indent=2)
print('extended %s' % OUT)
