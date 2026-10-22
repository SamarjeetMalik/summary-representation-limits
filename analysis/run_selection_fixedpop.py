"""The inherited GRAPES-3 age selection, with its estimands kept apart.

run_shower_crossfit.py reports sigma on ALL evaluation events (0.1021) and sigma of the same model on
the events the cut ACCEPTS (0.1115). Those are different populations, so their difference mixes the
effect of the cut on reconstruction with the difference in difficulty between the two populations.
Here the comparisons are separated (shower cross-fit throughout, intervals resample showers):

  1. operational: one model, scored on all events and on the accepted events (population changes);
  2. fixed population = accepted events: trained on all events vs trained within the cut, paired;
  3. the rejected events, scored by the model trained on all events;
  4. the Gamma-test floor estimate of the nine features within the accepted and within the rejected
     events (queries and neighbours from that set, other showers only), i.e. how hard each population is;
  5. the spread of the mean bias across primaries on the accepted events, for both training choices.

Because the cut is a function of the summary (age is one of the features), restricting training to it
cannot lower the Bayes risk on the accepted events: the Bayes predictor there is the same E[y | T].

Extends results.json with 'selection_fixedpop'.  Run:  python analysis/run_selection_fixedpop.py
"""
import json, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as GBR
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results.json')
R = json.load(open(OUT))
KF, NB, K, KQ = 5, 1000, 10, 40
rng = np.random.default_rng(0)

d = C.load_root(); c = C.load_csv(); y = d['y']; X9 = C.build_features(d)
r_idx, _ = C.match(d, c)
yt = y[r_idx]; age_t = d['Age'][r_idx]; pid_t = d['PrimaryID'][r_idx]
fold = C.shower_folds(d, KF); grp = C.shower_ids(d)
ftest = fold[r_idx]
ug, inv = np.unique(grp[r_idx], return_inverse=True)
order = np.argsort(inv, kind='stable'); starts = np.r_[0, np.cumsum(np.bincount(inv))]
BOOT = [np.concatenate([order[starts[i]:starts[i + 1]] for i in rng.integers(0, len(ug), len(ug))]) for _ in range(NB)]
GB = lambda s=42: GBR(max_iter=600, learning_rate=.06, random_state=s)


def crossfit(X, keep=None):
    p = np.empty(len(r_idx))
    for k in range(KF):
        tr = fold != k if keep is None else (fold != k) & keep
        p[ftest == k] = GB().fit(X[tr], y[tr]).predict(X[r_idx[ftest == k]])
    return p - yt


acc_t = (age_t >= .90) & (age_t <= 1.15)
acc_all = (d['Age'] >= .90) & (d['Age'] <= 1.15)
e_all = crossfit(X9)
e_within = crossfit(X9, keep=acc_all)

# reproduce the published numbers before adding new ones
ref = R['shower_crossfit']['selection']
chk = (round(float(e_all.std()), 4), round(float(e_all[acc_t].std()), 4), round(float(e_within[acc_t].std()), 4))
assert chk == (ref['sigma_none'], ref['sigma_standard'], ref['sigma_standard_trained_within']), \
    'does not reproduce run_shower_crossfit.py: %s vs %s' % (chk, ref)
print('reproduces sigma all / accepted / accepted-trained-within:', chk)


def sd_ci(e, m):
    b = [e[r][m[r]].std() for r in BOOT]
    return round(float(e[m].std()), 4), [round(float(v), 4) for v in np.percentile(b, [2.5, 97.5])]


def paired_same(e_new, e_old, m):
    b = [e_new[r][m[r]].std() - e_old[r][m[r]].std() for r in BOOT]
    return round(float(e_new[m].std() - e_old[m].std()), 5), [round(float(v), 5) for v in np.percentile(b, [2.5, 97.5])]


def gamma_floor(mask_all, mask_t):
    """Floor estimate of the nine features for queries = evaluation events in the set, neighbours =
    all events in the set from other showers; shower-bootstrap interval over the queries."""
    pool = np.where(mask_all)[0]; query = r_idx[mask_t]
    sc = StandardScaler().fit(X9[pool])
    dist, idx = NearestNeighbors(n_neighbors=KQ).fit(sc.transform(X9[pool])).kneighbors(sc.transform(X9[query]))
    idx = pool[idx]
    rank_ = np.where(grp[idx] != grp[query][:, None], np.arange(KQ)[None, :], 99)
    o = np.sort(rank_, 1)[:, :K]
    assert (o < 99).all()
    d2 = np.take_along_axis(dist, o, 1) ** 2; I = np.take_along_axis(idx, o, 1)
    g2 = 0.5 * (y[query][:, None] - y[I]) ** 2
    f = lambda rows=slice(None): float(np.sqrt(max(np.polyfit(d2[rows].mean(0), g2[rows].mean(0), 1)[1], 0)))
    qg = grp[query]; ugq, iq = np.unique(qg, return_inverse=True)
    oq = np.argsort(iq, kind='stable'); sq = np.r_[0, np.cumsum(np.bincount(iq))]
    bt = [f(np.concatenate([oq[sq[i]:sq[i + 1]] for i in np.random.default_rng(500 + b).integers(0, len(ugq), len(ugq))]))
          for b in range(300)]
    return round(f(), 4), [round(float(v), 4) for v in np.percentile(bt, [2.5, 97.5])]


spread = lambda e, m: round((lambda b: max(b) - min(b))([float(e[m & (pid_t == k)].mean()) for k in C.SPECIES]), 4)
rej_t = ~acc_t
out = {'accepted_pct': round(100 * acc_t.mean(), 1),
       'operational': {'all_events': sd_ci(e_all, np.ones(len(yt), bool)), 'accepted_events': sd_ci(e_all, acc_t)},
       'fixed_accepted': {'trained_on_all': sd_ci(e_all, acc_t), 'trained_within_cut': sd_ci(e_within, acc_t),
                          'within_minus_all': paired_same(e_within, e_all, acc_t)},
       'rejected_events': sd_ci(e_all, rej_t),
       'floor_accepted': gamma_floor(acc_all, acc_t), 'floor_rejected': gamma_floor(~acc_all, rej_t),
       'bias_spread_accepted': {'trained_on_all': spread(e_all, acc_t), 'trained_within_cut': spread(e_within, acc_t)},
       'bias_spread_all_events_trained_on_all': spread(e_all, np.ones(len(yt), bool))}
R['selection_fixedpop'] = out
for k, v in out.items():
    print(k, v)
json.dump(R, open(OUT, 'w'), indent=1)
print('extended results.json with selection_fixedpop')
