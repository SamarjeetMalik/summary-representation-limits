"""Small inputs for the figures of the physics-journal manuscript that need event-level data.

Writes analysis/kcdc/kcdc_figure_inputs.json, so that figures/make_figures_epjc.py can draw from
results files only:
  events     two simulated events (QGSJET-II-04) at nearly the same true energy, one proton and one
             iron, near-vertical and with the core near the array centre: station positions and
             their e/gamma deposit, muon deposit and arrival time, plus reconstructed and true sizes;
  lateral    mean log10(1 + signal) in the radial annuli used as station features, for protons and
             for iron in 5.4 <= log10(E/GeV) < 5.6, per interaction model;
  measured   histograms of log10 Nmu in decades of reconstructed energy for the measured data and
             for each model (proton, equal mixture, iron; knee spectrum), with the same cuts as
             run_kcdc_measured.py (including log10 Ne >= 4).
"""
import csv, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'data', 'kcdc')
R = {}

# ---- two example events
F = np.load(os.path.join(D, 'kcdc_station_features.npz')); Dn = np.load(os.path.join(D, 'kcdc_station_dense.npz'))
POS = np.zeros((252, 2))
for r in csv.DictReader(open(os.path.join(HERE, 'kascade_station_coordinates.csv'))):
    POS[int(r['station_id']) - 1] = [float(r['x_m']), float(r['y_m'])]
ze = F['Ze'] if np.nanmax(F['Ze']) > 3.2 else np.degrees(F['Ze'])
base = (np.abs(F['TrEP'] - 15.5) < 0.03) & (ze < 15) & (np.hypot(F['Xc'], F['Yc']) < 30)
R['events'] = {'station_x': POS[:, 0].round(1).tolist(), 'station_y': POS[:, 1].round(1).tolist()}
for p, nm in ((14, 'proton'), (5626, 'iron')):
    i = int(np.where(base & (F['TrPP'] == p))[0][0])
    R['events'][nm] = {'e': np.round(Dn['dense_e'][i].astype(float), 2).tolist(),
                       'mu': np.round(Dn['dense_mu'][i].astype(float), 2).tolist(),
                       't': np.round(Dn['dense_t'][i].astype(float), 1).tolist(),
                       'lgE_eV': round(float(F['TrEP'][i]), 3), 'zenith': round(float(ze[i]), 1),
                       'core': [round(float(F['Xc'][i]), 1), round(float(F['Yc'][i]), 1)],
                       'Ne': round(float(F['Ne'][i]), 2), 'Nmu': round(float(F['Nmu'][i]), 2),
                       'TrNe': round(float(F['TrNe'][i]), 2), 'TrNm': round(float(F['TrNm'][i]), 2)}

# ---- mean lateral profiles: per-station signal against distance to the reconstructed core
RB = np.arange(0, 141, 10.0)
R['lateral'] = {'r_edges': RB.tolist(), 'models': {}}
for name, suf in (('QGSJET-II-04', ''), ('EPOS-LHC', '_epos'), ('SIBYLL-2.3d', '_sibyll')):
    G = np.load(os.path.join(D, 'kcdc_station_features%s.npz' % suf)); Gd = np.load(os.path.join(D, 'kcdc_station_dense%s.npz' % suf))
    zg = G['Ze'] if np.nanmax(G['Ze']) > 3.2 else np.degrees(G['Ze'])
    w = (G['TrEP'] - 9 >= 5.4) & (G['TrEP'] - 9 < 5.6) & (zg < 30) & (np.hypot(G['Xc'], G['Yc']) < 60)
    out = {}
    for p, nm in ((14, 'proton'), (5626, 'iron')):
        idx = np.where(w & (G['TrPP'] == p))[0][:20000]
        r = np.hypot(POS[None, :, 0] - G['Xc'][idx, None], POS[None, :, 1] - G['Yc'][idx, None])
        b = np.digitize(r, RB) - 1; ok = (b >= 0) & (b < len(RB) - 1)
        prof = {}
        for key in ('e', 'mu'):
            v = np.log10(1 + np.maximum(Gd['dense_' + key][idx].astype(float), 0))
            prof[key] = (np.bincount(b[ok], v[ok], len(RB) - 1) / np.maximum(np.bincount(b[ok], None, len(RB) - 1), 1)).round(4).tolist()
        prof['n'] = int(len(idx)); out[nm] = prof
    R['lateral']['models'][name] = out

# ---- measured against simulation: log10 Nmu histograms per decade of reconstructed energy
def cut(d):
    ok = (d['E'] >= 14) & (d['E'] < 18) & (d['Ze'] < 40) & (np.hypot(d['Xc'], d['Yc']) <= 91) & (d['Ne'] >= 4.0)
    for k in ('E', 'Ne', 'Nmu', 'Age', 'Ze', 'Xc', 'Yc'):
        ok &= np.isfinite(d[k])
    return ok


def knee(x):
    return np.where(x < 15.6, 10 ** (-0.7 * (x - 14)), 10 ** (-0.7 * 1.6 - 1.1 * (x - 15.6)))


bins = np.arange(2.5, 6.51, 0.05)
M = np.load(os.path.join(D, 'kcdc_measured.npz')); okm = cut(M)
R['measured'] = {'bins': bins.round(2).tolist(), 'decades': {}}
for lo in (14, 15, 16):
    m = okm & (M['E'] >= lo) & (M['E'] < lo + 1)
    h, _ = np.histogram(M['Nmu'][m], bins); row = {'data': (h / h.sum()).round(5).tolist(), 'n_data': int(m.sum())}
    for name, tag in (('QGSJET-II-04', 'qgs4'), ('EPOS-LHC', 'epos'), ('SIBYLL-2.3d', 'sibyll')):
        S = np.load(os.path.join(D, 'kcdc_kascade_%s.npz' % tag)); ok = cut(S) & (S['E'] >= lo) & (S['E'] < lo + 1)
        w = knee(S['TrEP'])
        row[name] = {}
        for nm, sel in (('proton', S['TrPP'] == 14), ('mixture', np.ones(len(ok), bool)), ('iron', S['TrPP'] == 5626)):
            k = ok & sel; hs, _ = np.histogram(S['Nmu'][k], bins, weights=w[k])
            row[name][nm] = (hs / hs.sum()).round(5).tolist()
    R['measured']['decades']['%d-%d' % (lo, lo + 1)] = row
json.dump(R, open(os.path.join(HERE, 'kcdc_figure_inputs.json'), 'w'))
print('wrote kcdc_figure_inputs.json;', {k: R['events'][k]['lgE_eV'] for k in ('proton', 'iron')})
