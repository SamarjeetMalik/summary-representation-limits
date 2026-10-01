"""KCDC (KASCADE, CORSIKA + QGSJET-II-04) arm of the MLST paper.

Scope: this script does not measure an estimator plateau or a Bayes floor on KCDC. It does:

  B. Does the inherited-selection result transfer? Within narrow log10 Ne slices and over the
     full range: corr(age, log E), and resolution with a central-age selection versus none,
     retraining within the selection, plus the composition-bias spread.
  C. The degeneracy-probe calibration on a second experiment: phi = (Ne, Age, Xc, Yc),
     M = (Nmu, Ze); positive controls, noise injection and muon mixing.

Runs on CPU; parallel over slices with joblib. Input: data/kcdc/kcdc_kascade_qgs4.npz
(from convert_kcdc.py). Output: analysis/kcdc/kcdc_results.json.
"""
import json, os, sys, time
import numpy as np
from joblib import Parallel, delayed, parallel_config
from sklearn.ensemble import HistGradientBoostingRegressor as GBR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.environ.get('KCDC_NPZ', os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                                 'data', 'kcdc', 'kcdc_kascade_qgs4.npz'))
OUT = os.path.join(HERE, 'kcdc_results.json')
NJOBS = int(os.environ.get('NJOBS', '32'))
# Each worker gets a fixed thread budget. Without this, every loky worker inherits the parent's
# OMP/BLAS thread count, and 32 workers x 32+ threads on 32 cores ran for an hour instead of 30 s.
INNER = int(os.environ.get('KCDC_INNER_THREADS', '2'))
SPECIES = {14: 'H', 402: 'He', 1206: 'C', 2814: 'Si', 5626: 'Fe'}
R = {}
t0 = time.time()


def log(msg):
    print('[%5.0fs] %s' % (time.time() - t0, msg), flush=True)


def sigma(p, y):
    return float(np.std(p - y))


def r2(p, y):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def gbr(seed=0):
    return GBR(max_iter=600, learning_rate=.06, random_state=seed)


# ---------------------------------------------------------------- load + cuts
D = dict(np.load(NPZ))
ze = D['Ze'] if np.nanmax(D['Ze']) > 3.2 else np.degrees(D['Ze'])      # degrees either way
ok = np.ones(len(D['Ne']), bool)
for k in ('Ne', 'Nmu', 'Age', 'Xc', 'Yc', 'TrEP', 'TrNe', 'TrNm'):
    ok &= np.isfinite(D[k])
# Cleaning cuts: finite values and physically sensible log ranges.
ok &= (D['Ne'] > 1) & (D['Ne'] < 8) & (D['Nmu'] > 0) & (D['Nmu'] < 7) & (D['TrNe'] > 1) \
    & (D['TrNm'] > 0) & (D['TrEP'] > 13) & (D['TrEP'] < 18) & (D['Age'] > 0) & (D['Age'] < 2)
R['n_file'] = int(len(ok)); R['n_clean'] = int(ok.sum())
log('file %d events, %d after cleaning cuts' % (len(ok), ok.sum()))

y_all = D['TrEP'] - 9.0                                   # log10(E/GeV)
# One CORSIKA shower is thrown at 3 core positions in the KCDC KASCADE set (KCDC simulation
# manual, sec. 2.5). Copies share true energy and primary, so every split below is by shower.
_, SHOWER = np.unique(np.column_stack([D['TrEP'], D['TrPP'].astype(np.float32)]), axis=0,
                      return_inverse=True)
SHOWER = SHOWER.ravel()
R['n_showers'] = int(SHOWER.max() + 1)


def nine(Ne, Nmu, idx):
    """The nine-feature summary in the KCDC convention (muon ratio as log Nmu / log Ne)."""
    z = ze[idx]; xc, yc = D['Xc'][idx], D['Yc'][idx]
    return np.column_stack([Ne, Nmu, Nmu / Ne, D['Age'][idx], z, np.cos(np.radians(z)),
                            xc, yc, np.hypot(xc, yc)])


# ---------------------------------------------------------------- B. inherited selection
log('B. age selection within Ne slices and over the full range')
full = np.where(ok & (y_all >= 5) & (y_all <= 6))[0]
lo_ne, hi_ne = np.percentile(D['Ne'][full], [2, 98])
edges = np.round(np.arange(np.floor(lo_ne * 5) / 5, hi_ne, 0.4), 2)
slices = [('full range', full)] + [
    ('logNe %.1f-%.1f' % (e, e + .4), full[(D['Ne'][full] >= e) & (D['Ne'][full] < e + .4)])
    for e in edges]
slices = [(k, v) for k, v in slices if len(v) >= 40000]


def run_slice(label, ids, seed=0):
    r_ = np.random.default_rng(seed)
    ids = r_.permutation(ids)
    if len(ids) > 400000:
        ids = ids[:400000]
    X = nine(D['Ne'][ids], D['Nmu'][ids], ids); y = y_all[ids]; age = D['Age'][ids]
    pid = D['TrPP'][ids]
    # shower-disjoint 75/25 split: whole showers go to one side
    sh = SHOWER[ids]; ush = r_.permutation(np.unique(sh))
    in_tr = np.isin(sh, ush[:int(.75 * len(ush))])
    a, b = np.where(in_tr)[0], np.where(~in_tr)[0]
    q25, q75 = np.percentile(age[a], [25, 75])
    cuts = {'none': np.ones(len(ids), bool), 'central half of age': (age >= q25) & (age <= q75)}
    out = {'n': int(len(ids)), 'corr_age_logE': round(float(np.corrcoef(age, y)[0, 1]), 3),
           'age_iqr': [round(float(q25), 3), round(float(q75), 3)]}
    for cname, cm in cuts.items():
        trs, tes = a[cm[a]], b[cm[b]]
        p_all = gbr(seed).fit(X[a], y[a]).predict(X[tes])
        p_in = gbr(seed).fit(X[trs], y[trs]).predict(X[tes])
        bias = [float(np.mean(p_in[pid[tes] == k] - y[tes][pid[tes] == k]))
                for k in SPECIES if (pid[tes] == k).sum() > 200]
        out[cname] = {'kept': round(100 * cm[b].mean(), 1), 'sigma_trained_all': round(sigma(p_all, y[tes]), 4),
                      'sigma_trained_within': round(sigma(p_in, y[tes]), 4),
                      'bias_spread': round(max(bias) - min(bias), 4)}
    return label, out


with parallel_config(backend='loky', inner_max_num_threads=INNER):
    res = Parallel(n_jobs=min(NJOBS, len(slices)))(delayed(run_slice)(k, v) for k, v in slices)
R['selection'] = dict(res)
for k, v in res:
    log('  %-16s n=%7d corr=%.3f  none %.4f | central-half %.4f (within %.4f)  bias spread %.4f -> %.4f'
        % (k, v['n'], v['corr_age_logE'], v['none']['sigma_trained_within'],
           v['central half of age']['sigma_trained_all'], v['central half of age']['sigma_trained_within'],
           v['none']['bias_spread'], v['central half of age']['bias_spread']))

# ---------------------------------------------------------------- C. probe calibration
log('C. degeneracy probe calibration on KCDC')
ids = np.random.default_rng(1).permutation(full)[:300000]
PHI = np.column_stack([D['Ne'][ids], D['Age'][ids], D['Xc'][ids], D['Yc'][ids]])
M = np.column_stack([D['Nmu'][ids], ze[ids]])
y = y_all[ids]; sh = SHOWER[ids]
ush = np.random.default_rng(2).permutation(np.unique(sh))
in_tr = np.isin(sh, ush[:int(2 / 3 * len(ush))])
a, b = np.where(in_tr)[0], np.where(~in_tr)[0]          # shower-disjoint train / probe sets
GB_ = sh[b]


def probe(m, F, seed=0):
    oof = np.empty_like(m)
    for i, j in GroupKFold(5).split(F, groups=GB_):            # folds by shower
        oof[j] = GBR(max_iter=400, random_state=seed).fit(F[i], m[i]).predict(F[j])
    return r2(oof, m)


gb_phi = gbr().fit(PHI[a], y[a]).predict(PHI[b])
gb_all = gbr().fit(np.column_stack([PHI, M])[a], y[a]).predict(np.column_stack([PHI, M])[b])
mu_only = gbr().fit(M[a], y[a]).predict(M[b])
mu_only = (mu_only - mu_only.mean()) / mu_only.std() * gb_phi.std() + gb_phi.mean()
rng = np.random.default_rng(0)
outputs = {'GB on phi': gb_phi, 'GB on phi + M': gb_all, 'true energy': y[b]}
for tau in (0.02, 0.05, 0.10):
    outputs['GB on phi + N(0,%.2f)' % tau] = gb_phi + rng.normal(0, tau, len(gb_phi))
for lam in (0.1, 0.3, 0.5):
    outputs['(1-%.1f) GB(phi) + %.1f u(M)' % (lam, lam)] = (1 - lam) * gb_phi + lam * mu_only
Fphi, FphiM = PHI[b], np.column_stack([PHI, M])[b]
with parallel_config(backend='loky', inner_max_num_threads=INNER):
    pr = Parallel(n_jobs=min(NJOBS, 2 * len(outputs)))(
        delayed(probe)(m, F) for m in outputs.values() for F in (Fphi, FphiM))
R['probe'] = {}
for i, k in enumerate(outputs):
    p1, p2 = pr[2 * i], pr[2 * i + 1]
    R['probe'][k] = {'phi': round(p1, 4), 'phi_M': round(p2, 4), 'increment': round(p2 - p1, 4)}
    log('  %-32s R2(phi)=%.4f R2(phi,M)=%.4f inc=%+.4f' % (k, p1, p2, p2 - p1))

json.dump(R, open(OUT, 'w'), indent=2)
log('wrote %s' % OUT)
