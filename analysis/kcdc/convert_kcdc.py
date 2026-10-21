"""Convert the KCDC KASCADE QGSJET-II-04 simulation ROOT file(s) to one compact npz.

Input: the KASCADE "download All" file(s) from kcdc.iap.kit.edu/simul/qgsjet4/, placed in
data/kcdc/ (unzipped). Trees used: `array` (reconstructed: Ne, Nmu, Age, Ze, Az, Xc, Yc)
and `trmc` (MC truth: TrEP, TrPP, TrNe, TrNm). Ne, Nmu, TrNe, TrNm are stored as log10;
TrEP is log10(E/eV).

The two trees must be row-aligned. This script checks that through `row_map` when it is
present (and stops if it disagrees), so no event is ever paired with another event's truth.

Output: data/kcdc/kcdc_kascade_qgs4.npz (collaboration-public KCDC data, but keep it out of
the code release with the rest of data/).  Run:  python analysis/kcdc/convert_kcdc.py
"""
import glob, os, sys
import numpy as np
import uproot

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# KCDC_GEN selects the interaction model: unset = QGSJET-II-04 (data/kcdc/*.root, the original);
# 'epos' or 'sibyll' = data/kcdc/<gen>/*.root, written to kcdc_kascade_<gen>.npz.
GEN = os.environ.get('KCDC_GEN', '')
DIR = os.path.join(ROOT, 'data', 'kcdc', GEN) if GEN else os.path.join(ROOT, 'data', 'kcdc')
OUT = os.path.join(ROOT, 'data', 'kcdc', 'kcdc_kascade_%s.npz' % (GEN or 'qgs4'))
ARRAY = ['Ne', 'Nmu', 'Age', 'Ze', 'Az', 'Xc', 'Yc', 'E']        # E: KCDC's reconstructed log10(E/eV)
TRMC = ['TrEP', 'TrPP', 'TrNe', 'TrNm']

files = sorted(glob.glob(os.path.join(DIR, '*.root')))
if not files:
    sys.exit('no .root files under %s' % DIR)

cols = {k: [] for k in ARRAY + TRMC}
for fn in files:
    f = uproot.open(fn)
    print('%s\n  trees: %s' % (os.path.basename(fn), sorted(k.split(';')[0] for k in f.keys())))
    na, nt = f['array'].num_entries, f['trmc'].num_entries
    assert na == nt, 'array (%d) and trmc (%d) differ in length' % (na, nt)
    if 'row_map' in f:
        rm = f['row_map']
        ra, rt = rm['array'].array(library='np'), rm['trmc'].array(library='np')
        assert len(ra) == na and np.array_equal(ra, rt), 'row_map: array and trmc rows are not aligned'
        print('  row_map: array and trmc aligned on all %d rows' % len(ra))
    for k, arr in f['array'].arrays(ARRAY, library='np').items():
        cols[k].append(arr)
    for k, arr in f['trmc'].arrays(TRMC, library='np').items():
        cols[k].append(arr)
    print('  %d events' % na)

d = {k: np.concatenate(v) for k, v in cols.items()}
d['TrPP'] = d['TrPP'].astype(np.int32)
for k in d:
    if k != 'TrPP':
        d[k] = d[k].astype(np.float32)

# Alignment sanity: reconstructed and true sizes of the SAME event must correlate strongly.
fin = np.isfinite(d['Ne']) & np.isfinite(d['TrNe'])
r = np.corrcoef(d['Ne'][fin], d['TrNe'][fin])[0, 1]
print('corr(reco log Ne, true log Ne) = %.3f' % r)
assert r > 0.8, 'array and trmc look misaligned'
u, n = np.unique(d['TrPP'], return_counts=True)
print('primaries: %s' % dict(zip(u.tolist(), n.tolist())))
print('log10(E/eV): %.2f to %.2f' % (np.nanmin(d['TrEP']), np.nanmax(d['TrEP'])))
np.savez_compressed(OUT, **d)
print('wrote %s (%.0f MB)' % (OUT, os.path.getsize(OUT) / 1e6))
