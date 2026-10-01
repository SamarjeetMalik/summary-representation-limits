"""Per-station features for the KCDC KASCADE simulation, using the true detector positions.

Station coordinates come from the KCDC Simulation Manual, Appendix A (parsed into
kascade_station_coordinates.csv); with them the deposit-weighted centroid of each event tracks
the reconstructed core at r = 0.97, which validates the mapping. For each event we build
intermediate representations between "four fitted parameters" and "all 252 stations":

  e/gamma radial profile: summed EDeposit in annuli around the reconstructed core,
      r in [0,20), [20,40), [40,60), [60,80), [80,110), [110,inf) m   (log10(1 + sum))
  muon radial profile:    summed MDeposit in annuli [0,40), [40,70), [70,100), [100,inf) m
  counts and totals:      number of e/gamma and muon stations with signal, log total deposits
  timing:                 spread of arrival times over stations with signal

The file is ordered by primary, so events are read in evenly spread chunks. Output:
data/kcdc/kcdc_station_features.npz (with the summary, the MC truth and Run/Ev).
Run:  python analysis/kcdc/extract_stations.py
"""
import csv, glob, os
import numpy as np
import awkward as ak
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'data', 'kcdc')
OUT = os.path.join(ROOTDIR, 'kcdc_station_features.npz')
NCHUNK, CH = int(os.environ.get('NCHUNK', '24')), int(os.environ.get('CHUNK', '20000'))
E_EDGES = [0, 20, 40, 60, 80, 110, 1e9]
M_EDGES = [0, 40, 70, 100, 1e9]

X = np.full(260, np.nan); Y = np.full(260, np.nan)
for r in csv.DictReader(open(os.path.join(HERE, 'kascade_station_coordinates.csv'))):
    X[int(r['station_id'])] = float(r['x_m']); Y[int(r['station_id'])] = float(r['y_m'])

f = uproot.open(sorted(glob.glob(os.path.join(ROOTDIR, '*.root')))[0])
N = f['array'].num_entries
offs = np.linspace(0, N - CH, NCHUNK).astype(int)


def annuli(dep, sid, ev, n, xc, yc, edges):
    r = np.hypot(X[sid] - xc[ev], Y[sid] - yc[ev])
    out = np.zeros((n, len(edges) - 1))
    b = np.clip(np.searchsorted(edges, r, side='right') - 1, 0, len(edges) - 2)
    np.add.at(out, (ev, b), dep)
    return np.log10(1.0 + np.maximum(out, 0))


parts = []
for k, o in enumerate(offs):
    sl = dict(entry_start=int(o), entry_stop=int(o) + CH)
    g = f['general'].arrays(['Run', 'Ev', 'EDeposit', 'EDepositS', 'MDeposit', 'MDepositS', 'Arrival', 'ArrivalS'], **sl)
    a = f['array'].arrays(['Ne', 'Nmu', 'Age', 'Ze', 'Xc', 'Yc'], library='np', **sl)
    t = f['trmc'].arrays(['TrEP', 'TrPP', 'TrNe', 'TrNm'], library='np', **sl)
    n = len(a['Ne']); xc, yc = a['Xc'].astype(float), a['Yc'].astype(float)
    feats = {}
    for name, dkey, skey, edges in (('e', 'EDeposit', 'EDepositS', E_EDGES), ('mu', 'MDeposit', 'MDepositS', M_EDGES)):
        cnt = ak.to_numpy(ak.num(g[skey])); ev = np.repeat(np.arange(n), cnt)
        dep = ak.to_numpy(ak.flatten(g[dkey])).astype(float); sid = ak.to_numpy(ak.flatten(g[skey])).astype(int)
        feats[name + '_ann'] = annuli(dep, sid, ev, n, xc, yc, edges)
        feats[name + '_nhit'] = np.bincount(ev, dep > 0, n).astype(float)
        feats[name + '_logsum'] = np.log10(1.0 + np.maximum(np.bincount(ev, dep, n), 0))
    cnt = ak.to_numpy(ak.num(g['Arrival'])); ev = np.repeat(np.arange(n), cnt)
    tt = ak.to_numpy(ak.flatten(g['Arrival'])).astype(float)
    good = np.isfinite(tt) & (tt != 0)
    s1 = np.bincount(ev[good], tt[good], n); s2 = np.bincount(ev[good], tt[good] ** 2, n); c_ = np.bincount(ev[good], None, n)
    feats['t_spread'] = np.sqrt(np.maximum(s2 / np.maximum(c_, 1) - (s1 / np.maximum(c_, 1)) ** 2, 0))
    rec = dict(a); rec.update(t); rec.update(feats)
    rec['Run'] = ak.to_numpy(g['Run']); rec['Ev'] = ak.to_numpy(g['Ev'])
    parts.append(rec)
    print('  chunk %2d/%d at %d: %d events' % (k + 1, NCHUNK, o, n), flush=True)

D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
np.savez_compressed(OUT, **D, e_edges=np.array(E_EDGES), mu_edges=np.array(M_EDGES))
print('wrote %s: %d events, %.0f MB' % (OUT, len(D['Ne']), os.path.getsize(OUT) / 1e6))
