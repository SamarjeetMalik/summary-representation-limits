"""Label-free checks of the KASCADE audit on measured data.

Measured events have no true energy, so no floor or plateau can be measured on them. What can be
checked without labels:

  A. Data against simulation. In decades of KCDC's reconstructed energy E, the distributions of
     log Ne, log Nmu, age and zenith in the data are compared with each interaction model
     (QGSJET-II-04, EPOS-LHC, SIBYLL-2.3d) for pure proton, pure iron and an equal five-species
     mixture, reweighted from the generated E^-2 spectrum to a knee spectrum (index -2.7 below
     10^15.6 eV, -3.1 above). Distance: Wasserstein-1 between weighted distributions.
  B. Where the data sit between proton and iron. In bins of log Ne (zenith < 30 deg), the mean
     log Nmu of the data placed on the proton-to-iron scale of each model:
     z = (data - p) / (Fe - p); z in [0, 1] means the data lie inside the model's mass bracket.
     This is a consistency check of the simulation, not a composition measurement.
  C. The degeneracy probe on measured data. Gradient boosting trained on each simulation on
     phi = (Ne, Age, Xc, Yc) and on phi + M with M = (Nmu, Ze), applied to the data; the probe
     R2(phi) and R2(phi, M) of each output is computed on the data and on held-out simulation.
     If the network's dependence on the muon channel is the same in data and simulation, the two
     agree. Controls as in run_kcdc.py (noise injection, muon mixing).

Cuts on both data and simulation: 14 <= E < 18 (reconstructed, log10 eV), zenith < 40 deg,
core within 91 m of the array centre, log10 Ne >= 4.0 (above the measured release's size
threshold), all quantities finite.

Closure test (no measured data needed): MEASURED_TEST=<gen> draws pseudo-data from half of that
generator's showers, resampled to the knee spectrum with the equal mixture, and compares it with
the other half and with the other generators. Expected: small distance to <gen> mix, z near the
mixture's value for <gen>, probe values equal within noise.

Input: data/kcdc/kcdc_measured.npz (convert_kcdc_measured.py) and kcdc_kascade_{qgs4,epos,sibyll}.npz.
Output: analysis/kcdc/kcdc_measured_results.json (kcdc_measured_test_<gen>.json in test mode).
"""
import json, os, sys, time
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.ensemble import HistGradientBoostingRegressor as GBR
from sklearn.model_selection import GroupKFold, KFold

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'data', 'kcdc')
TEST = os.environ.get('MEASURED_TEST', '')
OUT = os.path.join(HERE, 'kcdc_measured_test_%s.json' % TEST if TEST else 'kcdc_measured_results.json')
GENS = {'QGSJET-II-04': 'qgs4', 'EPOS-LHC': 'epos', 'SIBYLL-2.3d': 'sibyll'}
SPECIES = {14: 'p', 402: 'He', 1206: 'C', 2814: 'Si', 5626: 'Fe'}
DECADES = [(14, 15), (15, 16), (16, 17), (17, 18)]
NE_BINS = np.arange(3.0, 7.01, 0.25)
VARS = ('Ne', 'Nmu', 'Age', 'Ze')
NMAX = int(os.environ.get('NMAX', '300000'))
# The measured release has a soft size threshold near log10 Ne = 3.6-3.8 that the simulation release
# does not (at the lowest reconstructed energies the data's median log10 Ne is about 0.5 above the
# simulation's, which keeps small, mostly heavy showers). A common cut above it, applied identically
# to data and simulation, makes the comparison like for like.
NE_MIN = float(os.environ.get('NE_MIN', '4.0'))
t0 = time.time()
R = {'test': TEST or None, 'common_cut_log10_Ne_min': NE_MIN}


def log(m):
    print('[%5.0fs] %s' % (time.time() - t0, m), flush=True)


def cuts(d):
    ok = np.ones(len(d['E']), bool)
    for k in ('E', 'Ne', 'Nmu', 'Age', 'Ze', 'Xc', 'Yc'):
        ok &= np.isfinite(d[k])
    ze = d['Ze'] if np.nanmax(d['Ze']) > 3.2 else np.degrees(d['Ze'])
    ok &= (d['E'] >= 14) & (d['E'] < 18) & (ze < 40) & (np.hypot(d['Xc'], d['Yc']) <= 91) & (d['Ne'] >= NE_MIN)
    out = {k: np.asarray(d[k])[ok] for k in d if np.ndim(d[k]) == 1 and len(d[k]) == len(ok)}
    out['Ze'] = ze[ok].astype(np.float32)
    return out


def knee_weight(lgE_eV):
    x = lgE_eV
    return np.where(x < 15.6, 10 ** (-0.7 * (x - 14)), 10 ** (-0.7 * 1.6 - 1.1 * (x - 15.6)))


def showers(d):
    _, s = np.unique(np.column_stack([d['TrEP'], d['TrPP'].astype(np.float32)]), axis=0, return_inverse=True)
    return s.ravel()


# ------------------------------------------------------------------ load
SIM = {}
for name, tag in GENS.items():
    d = cuts(dict(np.load(os.path.join(DATA, 'kcdc_kascade_%s.npz' % tag))))
    d['w'] = knee_weight(d['TrEP'])
    d['sh'] = showers(d)
    SIM[name] = d
    log('%-13s %d events after cuts' % (name, len(d['E'])))

if TEST:
    name = [n for n, t in GENS.items() if t == TEST][0]
    d = SIM[name]
    ush = np.random.default_rng(7).permutation(np.unique(d['sh']))
    half = np.isin(d['sh'], ush[:len(ush) // 2])
    pool = {k: v[half] for k, v in d.items()}
    SIM[name] = {k: v[~half] for k, v in d.items()}
    rng = np.random.default_rng(8)
    keep = rng.random(len(pool['w'])) < pool['w'] / pool['w'].max()
    MEAS = {k: pool[k][keep] for k in ('E', 'Ne', 'Nmu', 'Age', 'Ze', 'Xc', 'Yc')}
    log('closure test: pseudo-data from %s, %d events (knee spectrum, equal mixture)' % (name, len(MEAS['E'])))
else:
    f = os.path.join(DATA, 'kcdc_measured.npz')
    if not os.path.exists(f):
        sys.exit('no %s yet: run convert_kcdc_measured.py on the Data Shop download' % f)
    raw = dict(np.load(f))
    R['n_total_in_file'] = int(raw.pop('n_total')); R['keep_fraction'] = float(raw.pop('keep'))
    MEAS = cuts(raw)
    log('measured: %d events after cuts (file total %d, kept fraction %.4f)'
        % (len(MEAS['E']), R['n_total_in_file'], R['keep_fraction']))
R['n_measured_after_cuts'] = int(len(MEAS['E']))

# ------------------------------------------------------------------ A. distributions per decade
log('A. data against simulation, per reconstructed-energy decade')
rng = np.random.default_rng(0)
R['distributions'] = {}
for lo, hi in DECADES:
    key = '%d-%d' % (lo, hi)
    m = (MEAS['E'] >= lo) & (MEAS['E'] < hi)
    if m.sum() < 200:
        continue
    idx = np.where(m)[0]
    if len(idx) > NMAX:
        idx = rng.choice(idx, NMAX, replace=False)
    row = {'n_data': int(m.sum()), 'data': {v: [round(float(MEAS[v][idx].mean()), 4), round(float(MEAS[v][idx].std()), 4)] for v in VARS}}
    for name, d in SIM.items():
        s = (d['E'] >= lo) & (d['E'] < hi)
        comp = {'mix': s}
        comp.update({SPECIES[p]: s & (d['TrPP'] == p) for p in (14, 5626)})
        row[name] = {}
        for cname, cm in comp.items():
            if cm.sum() < 50:
                continue
            w = d['w'][cm]
            row[name][cname] = {
                'n_sim': int(cm.sum()),
                'mean': {v: round(float(np.average(d[v][cm], weights=w)), 4) for v in VARS},
                'w1': {v: round(float(wasserstein_distance(MEAS[v][idx], d[v][cm], None, w)), 4) for v in VARS}}
    R['distributions'][key] = row
    log('  %s: n_data %d; W1(log Nmu) to mix: %s' % (key, m.sum(), {n: row[n].get('mix', {}).get('w1', {}).get('Nmu') for n in SIM}))

# ------------------------------------------------------------------ B. proton-to-iron scale of log Nmu at fixed log Ne
log('B. mean log Nmu at fixed log Ne on the proton-to-iron scale (zenith < 30 deg)')
R['muon_scale'] = {'ne_bins': NE_BINS.tolist(), 'models': {}}
vd = MEAS['Ze'] < 30
for name, d in SIM.items():
    vs = d['Ze'] < 30
    rows = []
    for a, b in zip(NE_BINS[:-1], NE_BINS[1:]):
        md = vd & (MEAS['Ne'] >= a) & (MEAS['Ne'] < b)
        mp = vs & (d['Ne'] >= a) & (d['Ne'] < b) & (d['TrPP'] == 14)
        mf = vs & (d['Ne'] >= a) & (d['Ne'] < b) & (d['TrPP'] == 5626)
        mm = vs & (d['Ne'] >= a) & (d['Ne'] < b)
        if md.sum() < 100 or mp.sum() < 30 or mf.sum() < 30:
            continue
        p = np.average(d['Nmu'][mp], weights=d['w'][mp]); fe = np.average(d['Nmu'][mf], weights=d['w'][mf])
        mix = np.average(d['Nmu'][mm], weights=d['w'][mm]); x = MEAS['Nmu'][md].mean()
        rows.append({'ne': [float(a), float(b)], 'n_data': int(md.sum()), 'median_E_data': round(float(np.median(MEAS['E'][md])), 3),
                     'p': round(float(p), 4), 'Fe': round(float(fe), 4), 'data': round(float(x), 4),
                     'z_data': round(float((x - p) / (fe - p)), 3), 'z_mix': round(float((mix - p) / (fe - p)), 3)})
    R['muon_scale']['models'][name] = rows
    log('  %-13s z_data by log Ne bin: %s' % (name, [r['z_data'] for r in rows]))
    log('  %-13s z_mix  by log Ne bin: %s' % (name, [r['z_mix'] for r in rows]))

# ------------------------------------------------------------------ C. degeneracy probe on data and on held-out simulation
log('C. degeneracy probe: simulation-trained outputs probed on data and on held-out simulation')


def r2(p, y):
    return float(1 - np.mean((y - p) ** 2) / np.var(y))


def probe(m, F, groups=None):
    oof = np.empty_like(m)
    split = GroupKFold(5).split(F, groups=groups) if groups is not None else KFold(5, shuffle=True, random_state=0).split(F)
    for i, j in split:
        oof[j] = GBR(max_iter=300, random_state=0).fit(F[i], m[i]).predict(F[j])
    return r2(oof, m)


feat = lambda d: (np.column_stack([d['Ne'], d['Age'], d['Xc'], d['Yc']]), np.column_stack([d['Nmu'], d['Ze']]))
mi = rng.choice(len(MEAS['E']), min(NMAX // 2, len(MEAS['E'])), replace=False)
Pd, Md = feat({k: v[mi] for k, v in MEAS.items()})
R['probe'] = {}
for name, d in SIM.items():
    P, M = feat(d)
    ush = np.random.default_rng(2).permutation(np.unique(d['sh']))
    tr = np.isin(d['sh'], ush[:int(.75 * len(ush))])
    a = np.where(tr)[0]; b = np.where(~tr)[0]
    # held-out simulation resampled to the knee spectrum (equal mixture), the population the data
    # are compared with; the probe value depends on the population, so this must match part A
    b = b[np.random.default_rng(5).random(len(b)) < d['w'][b] / d['w'][b].max()]
    if len(b) > NMAX // 2:
        b = np.random.default_rng(3).choice(b, NMAX // 2, replace=False)
    y = d['TrEP'] - 9.0
    g_phi = GBR(max_iter=400, random_state=0).fit(P[a], y[a])
    g_all = GBR(max_iter=400, random_state=0).fit(np.column_stack([P, M])[a], y[a])
    g_mu = GBR(max_iter=400, random_state=0).fit(M[a], y[a])
    res = {}
    for where, (PP, MM, grp) in {'data': (Pd, Md, None), 'simulation': (P[b], M[b], d['sh'][b])}.items():
        o_phi = g_phi.predict(PP); o_all = g_all.predict(np.column_stack([PP, MM])); u = g_mu.predict(MM)
        u = (u - u.mean()) / u.std() * o_phi.std() + o_phi.mean()
        outs = {'GB on phi': o_phi, 'GB on phi + M': o_all,
                'GB on phi + N(0,0.05)': o_phi + np.random.default_rng(4).normal(0, .05, len(o_phi)),
                '0.7 GB(phi) + 0.3 u(M)': .7 * o_phi + .3 * u}
        FP, FPM = PP, np.column_stack([PP, MM])
        res[where] = {k: {'phi': round(probe(m, FP, grp), 4), 'phi_M': round(probe(m, FPM, grp), 4)} for k, m in outs.items()}
        for k in res[where]:
            res[where][k]['increment'] = round(res[where][k]['phi_M'] - res[where][k]['phi'], 4)
        e_kcdc = (MEAS['E'][mi] if where == 'data' else d['E'][b]) - 9.0     # KCDC's own estimate, log10(E/GeV)
        res[where]['mean_output_minus_E'] = round(float(np.mean(o_all - e_kcdc)), 4)
    R['probe'][name] = res
    log('  %-13s increment of GB(phi+M): data %+.4f  simulation %+.4f; mean(GB(phi+M) - KCDC E): data %+.4f  sim %+.4f'
        % (name, res['data']['GB on phi + M']['increment'], res['simulation']['GB on phi + M']['increment'],
           res['data']['mean_output_minus_E'], res['simulation']['mean_output_minus_E']))

json.dump(R, open(OUT, 'w'), indent=2)
log('wrote %s' % OUT)
