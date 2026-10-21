"""The representation audit on a second experiment: KASCADE simulation from KCDC.

Repeats on KASCADE the analyses that the GRAPES-3 part of the paper rests on, and adds the
one GRAPES-3 cannot support, because KCDC also releases per-station signals and Monte Carlo
truth:

  1. plateau: ridge, random forest, gradient boosting, MLP and kNN on the nine reconstructed
     summary features;
  2. estimated floor: Gamma test on the nine features (control first, as in run_bayes.py);
  3. information ladder: phi = (Ne, s, Xc, Yc) -> + zenith -> + core distance -> + muon size ->
     + muon ratio, with paired bootstrap differences;
  4. MC-truth oracle: the nine features with the reconstructed Ne, Nmu replaced by their true
     values. The difference to (1) is what the observation-and-reconstruction chain loses;
  5. intermediate representations between "the fit" and "all stations", built from the real
     station positions (extract_stations.py): radial e/gamma profile, radial muon profile,
     counts, totals and timing, each added to the nine features; reported as the fraction of
     the oracle gap each recovers.

Sample: events with log10(E/GeV) in [5, 6] after the KCDC data-selection cuts, from the
spread-chunk extraction. Every split is by CORSIKA shower (true energy and primary), since KCDC
throws each shower at three core positions.

Output: analysis/kcdc/kcdc_floor_results.json.  Run:  python analysis/kcdc/run_kcdc_floor.py
"""
import json, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
from joblib import Parallel, delayed, parallel_config
from sklearn.ensemble import HistGradientBoostingRegressor as GBR, RandomForestRegressor as RF
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.environ.get('KCDC_GEN', '')                  # '' = QGSJET-II-04; 'epos' or 'sibyll'
LGE = os.environ.get('KCDC_LGE', '')                  # window of true log10(E/GeV); unset = '5-6'
SUF = ('_' + GEN if GEN else '') + ('_e' + LGE.replace('-', '') if LGE else '')
LO_, HI_ = (float(v) for v in (LGE or '5-6').split('-'))
NPZ = os.environ.get('KCDC_STATIONS', os.path.join(os.path.dirname(os.path.dirname(HERE)), 'data', 'kcdc',
                                                    'kcdc_station_features%s.npz' % SUF))
OUT = os.path.join(HERE, 'kcdc_floor_results%s.json' % SUF)
NJOBS = int(os.environ.get('NJOBS', '8')); NB = 1000
R = {}

D = dict(np.load(NPZ))
ze = D['Ze'] if np.nanmax(D['Ze']) > 3.2 else np.degrees(D['Ze'])
y_all = D['TrEP'] - 9.0
ok = np.ones(len(y_all), bool)
for k in ('Ne', 'Nmu', 'Age', 'Xc', 'Yc', 'TrNe', 'TrNm'):
    ok &= np.isfinite(D[k])
ok &= (D['Ne'] > 1) & (D['Nmu'] > 0) & (D['TrNe'] > 1) & (D['TrNm'] > 0) & (y_all >= LO_) & (y_all <= HI_)
ids = np.where(ok)[0]
y = y_all[ids]; n = len(ids)
_, shower = np.unique(np.column_stack([D['TrEP'][ids], D['TrPP'][ids].astype(np.float32)]), axis=0, return_inverse=True)
shower = shower.ravel()
ush = np.random.default_rng(0).permutation(np.unique(shower))
is_tr = np.isin(shower, ush[:int(.75 * len(ush))])
tr, te = np.where(is_tr)[0], np.where(~is_tr)[0]; yt = y[te]
R.update({'n_events': int(n), 'n_showers': int(len(ush)), 'n_train': int(len(tr)), 'n_test': int(len(te)),
          'target_sd': round(float(y.std()), 4)})
print('KASCADE: %d events (%d showers), train %d / test %d, target sd %.3f' % (n, len(ush), len(tr), len(te), y.std()))


def col(k):
    return D[k][ids].astype(float)


Ne, Nmu, age, xc, yc = col('Ne'), col('Nmu'), col('Age'), col('Xc'), col('Yc')
zz = ze[ids].astype(float); cz = np.cos(np.radians(zz)); rc = np.hypot(xc, yc)
NINE = np.column_stack([Ne, Nmu, Nmu - Ne, age, zz, cz, xc, yc, rc])
ORACLE = np.column_stack([col('TrNe'), col('TrNm'), col('TrNm') - col('TrNe'), age, zz, cz, xc, yc, rc])
GB = lambda s=42: GBR(max_iter=600, learning_rate=.06, random_state=s)
BOOT = np.random.default_rng(1).integers(0, len(te), size=(NB, len(te)), dtype=np.int32)


def resid(X, model=None):
    m = model if model is not None else GB()
    return m.fit(X[tr], y[tr]).predict(X[te]) - yt


def paired(e_new, e_old):
    dl = e_new[BOOT].std(1) - e_old[BOOT].std(1)
    return round(float(e_new.std() - e_old.std()), 5), [round(float(v), 5) for v in np.percentile(dl, [2.5, 97.5])]


# ---------------------------------------------------------------- 1. plateau
MODELS = {'Ridge': lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
          'RandomForest': lambda: RF(n_estimators=60, min_samples_leaf=25, max_depth=18, n_jobs=4, random_state=42),
          'GradBoost': lambda: GB(),
          'MLP': lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=120,
                                                                      random_state=42, early_stopping=True)),
          'kNN (k=50)': lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=50, weights='distance'))}
with parallel_config(backend='loky', inner_max_num_threads=4):
    res = Parallel(n_jobs=min(NJOBS, len(MODELS)))(delayed(lambda k: (k, float(resid(NINE, MODELS[k]()).std())))(k)
                                                   for k in MODELS)
R['plateau'] = {k: round(v, 4) for k, v in res}
print('plateau:', R['plateau'])

# ---------------------------------------------------------------- 2. floor (control first)
def gamma(X, yy, K=10):
    Z = StandardScaler().fit_transform(X)
    P = K + 8                                                   # spare candidates for skipped copies
    dist, idx = NearestNeighbors(n_neighbors=P).fit(Z).kneighbors(Z)
    same = shower[idx] == shower[:, None]                       # never use a copy of the same shower
    keep = np.where(~same, np.arange(P)[None, :], 999); keep[:, 0] = 999
    order = np.sort(keep, 1)[:, :K]
    okr = (order < 999).all(1)
    D2 = np.take_along_axis(dist, np.minimum(order, P - 1), 1)[okr]
    I2 = np.take_along_axis(idx, np.minimum(order, P - 1), 1)[okr]
    g = 0.5 * ((yy[okr][:, None] - yy[I2]) ** 2).mean(0); dl = (D2 ** 2).mean(0)
    return float(np.sqrt(max(np.polyfit(dl, g, 1)[1], 0))), float(okr.mean())


Zc = StandardScaler().fit_transform(NINE)
mtrue = 0.9 * Zc[:, 0] + 0.35 * Zc[:, 1] + 0.08 * np.tanh(2 * Zc[:, 3]) + 0.05 * Zc[:, 0] * Zc[:, 3]
mtrue = (mtrue - mtrue.mean()) / mtrue.std() * y.std() * 0.85
ctrl = []
for s_ in (0.10, 0.15):
    est, _ = gamma(NINE, mtrue + np.random.default_rng(0).normal(0, s_, n))
    ctrl.append({'true_sd': s_, 'est': round(est, 4)}); print('  floor control: true %.3f est %.4f' % (s_, est))
    assert abs(est - s_) / s_ < 0.10, 'Gamma test fails its control on KASCADE'
fl, frac = gamma(NINE, y)
R['floor'] = {'nine': round(fl, 4), 'controls': ctrl, 'frac_with_other_shower_neighbours': round(frac, 3)}
fo, _ = gamma(ORACLE, y)
R['floor']['oracle'] = round(fo, 4)
print('estimated floor, nine features: %.4f; with MC-true sizes: %.4f' % (fl, fo))

# ---------------------------------------------------------------- 3. ladder
PHI = np.column_stack([Ne, age, xc, yc])
steps = [('A', 'four fit parameters', PHI), ('B', '+ zenith angle', np.column_stack([PHI, zz, cz])),
         ('C', '+ core distance', np.column_stack([PHI, zz, cz, rc])),
         ('D', '+ muon size', np.column_stack([PHI, zz, cz, rc, Nmu])),
         ('E', '+ muon ratio (nine features)', np.column_stack([PHI, zz, cz, rc, Nmu, Nmu - Ne]))]
lad, prev = [], None
for key, lab, Xs in steps:
    e = resid(Xs); row = {'rung': key, 'label': lab, 'sigma': round(float(e.std()), 4)}
    if prev is not None:
        row['delta'], row['delta_ci'] = paired(e, prev)
    lad.append(row); prev = e
    print('  ladder %s %-28s %.4f %s' % (key, lab, row['sigma'], row.get('delta', '')))
e_nine = prev
R['ladder'] = lad

# ---------------------------------------------------------------- 4 + 5. oracle and intermediate representations
e_or = resid(ORACLE)
reps = {'radial e/gamma profile': np.column_stack([NINE, D['e_ann'][ids]]),
        'radial muon profile': np.column_stack([NINE, D['mu_ann'][ids]]),
        'both radial profiles': np.column_stack([NINE, D['e_ann'][ids], D['mu_ann'][ids]]),
        'profiles + counts, totals, timing': np.column_stack([NINE, D['e_ann'][ids], D['mu_ann'][ids],
                                                              D['e_nhit'][ids], D['mu_nhit'][ids], D['e_logsum'][ids],
                                                              D['mu_logsum'][ids], D['t_spread'][ids]])}
gap = e_nine.std() - e_or.std()
inter = {}
for k, Xr in reps.items():
    e = resid(Xr); dlt, ci = paired(e, e_nine)
    inter[k] = {'sigma': round(float(e.std()), 4), 'delta': dlt, 'ci': ci,
                'gap_recovered_pct': round(100 * (e_nine.std() - e.std()) / gap, 1)}
    print('  %-34s %.4f  delta %+.5f [%+.5f, %+.5f]  gap recovered %.1f%%'
          % (k, e.std(), dlt, *ci, inter[k]['gap_recovered_pct']))
R['oracle'] = {'sigma_nine': round(float(e_nine.std()), 4), 'sigma_true_sizes': round(float(e_or.std()), 4),
               'gap': round(float(gap), 4)}
R['intermediate'] = inter
print('oracle: nine %.4f -> MC-true sizes %.4f (gap %.4f)' % (e_nine.std(), e_or.std(), gap))
json.dump(R, open(OUT, 'w'), indent=2)
print('wrote %s' % OUT)
