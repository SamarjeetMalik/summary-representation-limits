"""Validate the KCDC station-ID -> position table (KCDC Simulation Manual, Appendix A).

With the tabulated positions, the signal-weighted centroid of the e/gamma deposits of each event
should track the reconstructed shower core. Writes station_mapping_check.json.
Run (needs the KCDC ROOT file in data/kcdc/):  python analysis/kcdc/check_station_mapping.py
"""
import csv, glob, json, os
import numpy as np
import awkward as ak
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'data', 'kcdc')
X = np.full(260, np.nan); Y = np.full(260, np.nan)
for r in csv.DictReader(open(os.path.join(HERE, 'kascade_station_coordinates.csv'))):
    X[int(r['station_id'])] = float(r['x_m']); Y[int(r['station_id'])] = float(r['y_m'])
f = uproot.open(sorted(glob.glob(os.path.join(ROOTDIR, '*.root')))[0])
N = f['array'].num_entries; out = []
for o in np.linspace(0, N - 5000, 6).astype(int):
    g = f['general'].arrays(['EDeposit', 'EDepositS'], entry_start=int(o), entry_stop=int(o) + 5000)
    a = f['array'].arrays(['Xc', 'Yc'], entry_start=int(o), entry_stop=int(o) + 5000, library='np')
    cnt = ak.to_numpy(ak.num(g['EDepositS'])); ev = np.repeat(np.arange(len(cnt)), cnt)
    e = np.maximum(ak.to_numpy(ak.flatten(g['EDeposit'])).astype(float), 0); s = ak.to_numpy(ak.flatten(g['EDepositS']))
    w = np.bincount(ev, e, len(cnt)); okw = w > 0
    cx = np.bincount(ev, e * X[s], len(cnt))[okw] / w[okw]; cy = np.bincount(ev, e * Y[s], len(cnt))[okw] / w[okw]
    out.append((cx, cy, a['Xc'][okw], a['Yc'][okw]))
cx, cy, xc, yc = (np.concatenate([o[i] for o in out]) for i in range(4))
res = {'n_events': int(len(cx)), 'corr_x': round(float(np.corrcoef(cx, xc)[0, 1]), 3),
       'corr_y': round(float(np.corrcoef(cy, yc)[0, 1]), 3),
       'median_offset_m': round(float(np.median(np.hypot(cx - xc, cy - yc))), 1)}
json.dump(res, open(os.path.join(HERE, 'station_mapping_check.json'), 'w'), indent=2)
print(res)
