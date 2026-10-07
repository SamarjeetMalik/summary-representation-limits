"""Conformalised quantile regression with shower-level exchangeability.

run_diagnostics.py calibrates on a random 25% of the training complement, so copies of one
shower can sit in the training, calibration and evaluation sets at once, and the conformal
guarantee, which needs calibration and evaluation scores to be exchangeable, is not the one that
applies. Here every split is by shower: for each of the five shower folds, the showers of the
other four are divided 75/25 into a training and a calibration set, and the evaluation events of
the held-out fold are scored. Calibration and evaluation events then come from disjoint,
randomly assigned showers.

Coverage is reported overall, per age stratum and per primary, at nominal 68% and 90%.

Extends results.json.  Run:  python analysis/run_cqr_shower.py
"""
import json, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as GBR

import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results.json')
R = json.load(open(OUT))
KF = 5
NOM = [0.50, 0.68, 0.80, 0.90, 0.95]

d = C.load_root(); c = C.load_csv(); y = d['y']; X = C.build_features(d)
r_idx, _ = C.match(d, c)
yt = y[r_idx]; age = d['Age'][r_idx]; pid = d['PrimaryID'][r_idx]
fold = C.shower_folds(d, KF); grp = C.shower_ids(d); ftest = fold[r_idx]
# a second, independent shower-level draw decides training against calibration
is_cal = (np.random.default_rng(1).random(grp.max() + 1) < 0.25)[grp]

raw = {n: np.zeros(len(r_idx), bool) for n in NOM}
cqr = {n: np.zeros(len(r_idx), bool) for n in NOM}
for k in range(KF):
    tr = (fold != k) & ~is_cal; cal = (fold != k) & is_cal
    te = r_idx[ftest == k]; m = ftest == k
    for n in NOM:
        lo_q, hi_q = (1 - n) / 2, 1 - (1 - n) / 2
        mlo = GBR(loss='quantile', quantile=lo_q, max_iter=600, learning_rate=.06, random_state=42).fit(X[tr], y[tr])
        mhi = GBR(loss='quantile', quantile=hi_q, max_iter=600, learning_rate=.06, random_state=42).fit(X[tr], y[tr])
        e = np.maximum(mlo.predict(X[cal]) - y[cal], y[cal] - mhi.predict(X[cal]))
        Q = float(np.quantile(e, min(1.0, n * (1 + 1 / cal.sum()))))
        lo, hi = mlo.predict(X[te]), mhi.predict(X[te])
        raw[n][m] = (y[te] >= lo) & (y[te] <= hi)
        cqr[n][m] = (y[te] >= lo - Q) & (y[te] <= hi + Q)

pct = lambda v: round(100 * float(v.mean()), 1)
out = {'nominal': NOM, 'raw': [pct(raw[n]) for n in NOM], 'cqr': [pct(cqr[n]) for n in NOM]}
BINS = [(0.80, 0.90), (0.90, 1.00), (1.00, 1.10), (1.10, 1.15), (1.15, 1.25), (1.25, 1.40)]
out['by_age'] = [{'bin': list(b), 'n': int(((age >= b[0]) & (age < b[1])).sum()),
                  **{'%s%d' % (t, round(100 * n)): pct(v[n][(age >= b[0]) & (age < b[1])])
                     for n in (0.68, 0.90) for t, v in (('raw', raw), ('cqr', cqr))}} for b in BINS]
out['by_primary'] = {nm: {'n': int((pid == code).sum()),
                          **{'cqr%d' % round(100 * n): pct(cqr[n][pid == code]) for n in (0.68, 0.90)}}
                     for code, nm in C.SPECIES.items()}
dev = lambda rows, key, nom: max(abs(r[key] - nom) for r in rows)
out['max_dev_cqr_overall'] = round(max(abs(a - 100 * n) for a, n in zip(out['cqr'], NOM)), 1)
out['max_dev_raw_overall'] = round(max(abs(a - 100 * n) for a, n in zip(out['raw'], NOM)), 1)
out['max_dev_age_cqr90'] = round(dev(out['by_age'], 'cqr90', 90), 1)
out['max_dev_primary'] = round(max(max(abs(v['cqr68'] - 68), abs(v['cqr90'] - 90)) for v in out['by_primary'].values()), 1)
print(json.dumps(out, indent=1)[:1800])
R['cqr_shower'] = out
json.dump(R, open(OUT, 'w'), indent=2)
print('extended %s' % OUT)
