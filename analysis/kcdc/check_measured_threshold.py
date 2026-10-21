"""Is the measured release's lack of small showers a selection, or the composition?

At the lowest reconstructed energies (14 <= log10 E_rec/eV < 14.2; zenith < 40 deg, core within
91 m), compare the fraction of events with log10 Ne < 3.5 in the measured data with the same fraction
in each simulation for pure proton and pure iron, reweighted to the knee spectrum. If even pure
protons have far more small showers than the data, composition cannot explain the difference.
Output: analysis/kcdc/kcdc_measured_threshold.json.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'data', 'kcdc')


def cut(d):
    return (d['E'] >= 14) & (d['E'] < 14.2) & (d['Ze'] < 40) & (np.hypot(d['Xc'], d['Yc']) <= 91)


m = np.load(os.path.join(D, 'kcdc_measured.npz')); ok = cut(m)
R = {'window_log10E_rec': [14.0, 14.2], 'data_frac_below_3.5': round(float((m['Ne'][ok] < 3.5).mean()), 4),
     'data_n': int(ok.sum()), 'models': {}}
for name, tag in (('QGSJET-II-04', 'qgs4'), ('EPOS-LHC', 'epos'), ('SIBYLL-2.3d', 'sibyll')):
    s = np.load(os.path.join(D, 'kcdc_kascade_%s.npz' % tag)); ok = cut(s); x = s['TrEP'][ok]
    w = np.where(x < 15.6, 10 ** (-0.7 * (x - 14)), 10 ** (-0.7 * 1.6 - 1.1 * (x - 15.6)))
    R['models'][name] = {nm: round(float(np.average(s['Ne'][ok][s['TrPP'][ok] == p] < 3.5, weights=w[s['TrPP'][ok] == p])), 3)
                         for p, nm in ((14, 'proton'), (5626, 'iron'))}
print(R)
json.dump(R, open(os.path.join(HERE, 'kcdc_measured_threshold.json'), 'w'), indent=2)
