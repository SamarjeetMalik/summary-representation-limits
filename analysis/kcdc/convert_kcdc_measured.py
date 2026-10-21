"""Convert a KCDC KASCADE Data Shop download (measured events, ROOT) to one compact npz.

Input: the unzipped Data Shop file(s) in data/kcdc/measured/ (or MEASURED_DIR). The request was
KASCADE E, Xc, Yc, Ze, Az, Ne, Nmu, Age plus General Info Run and Ev, reconstructed energy cut
14 <= log10(E/eV) <= 18, everything else full range. The simulation files use trees `array`
and `general` with these branch names; the Data Shop layout is not documented for us, so
branches are located by name in whichever trees hold them, and trees that are read together
must have the same number of entries (checked).

Measured data can be large. Events are kept with probability KEEP (Bernoulli, fixed seed), chosen
so that about TARGET events are kept; the total count and KEEP are stored with the output so
rates can be scaled back. Measured events are independent showers, so a random subsample is a
fair sample (unlike block reads of the ordered simulation files).

Output: data/kcdc/kcdc_measured.npz.  Run:  python analysis/kcdc/convert_kcdc_measured.py
Test on a simulation file:  MEASURED_DIR=data/kcdc/epos MEASURED_OUT=/tmp/x.npz python ...
"""
import glob, os, sys
import numpy as np
import uproot

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR = os.environ.get('MEASURED_DIR', os.path.join(ROOT, 'data', 'kcdc', 'measured'))
OUT = os.environ.get('MEASURED_OUT', os.path.join(ROOT, 'data', 'kcdc', 'kcdc_measured.npz'))
TARGET = int(os.environ.get('TARGET', '20000000'))
WANT = ['E', 'Xc', 'Yc', 'Ze', 'Az', 'Ne', 'Nmu', 'Age', 'Run', 'Ev']
ALIAS = {'Run': ['Run', 'R']}               # the Data Shop names the run-number branch "R" (info.txt)

files = sorted(glob.glob(os.path.join(DIR, '**', '*.root'), recursive=True))
if not files:
    sys.exit('no .root files under %s' % DIR)


def locate(f):
    """Map each wanted branch to the tree that holds it (exact name first, then case-insensitive)."""
    where = {}
    trees = [k.split(';')[0] for k, c in f.classnames().items() if c.startswith('TTree')]
    for w in WANT:
        for exact in (True, False):
            for t in trees:
                names = f[t].keys()
                hit = [b for b in names for ww in ALIAS.get(w, [w]) if (b == ww if exact else b.lower() == ww.lower())]
                if hit and w not in where:
                    where[w] = (t, hit[0])
    return trees, where


total = 0
plan = []
for fn in files:
    f = uproot.open(fn)
    trees, where = locate(f)
    missing = [w for w in WANT if w not in where]
    print('%s\n  trees %s\n  found %s' % (os.path.basename(fn), trees, {w: '%s/%s' % v for w, v in where.items()}))
    if missing:
        sys.exit('  missing branches %s: send me the tree listing above' % missing)
    used = sorted({t for t, _ in where.values()})
    n = {t: f[t].num_entries for t in used}
    assert len(set(n.values())) == 1, 'trees to be read together differ in length: %s' % n
    total += next(iter(n.values()))
    plan.append((fn, where, used))

keep = min(1.0, TARGET / max(total, 1))
print('total events %d; keeping each with probability %.4f' % (total, keep))
rng = np.random.default_rng(20261016)
cols = {w: [] for w in WANT}
for fn, where, used in plan:
    f = uproot.open(fn)
    n = f[used[0]].num_entries
    step = 2_000_000
    for s in range(0, n, step):
        e = min(n, s + step)
        sel = rng.random(e - s) < keep
        if 'row_map' in f and len(used) > 1:      # the trees read together must be row-aligned
            rm = f['row_map'].arrays(used, library='np', entry_start=s, entry_stop=e)
            assert all(np.array_equal(rm[used[0]], rm[t]) for t in used[1:]), 'row_map: trees misaligned at %d' % s
        for t in used:
            br = [b for w, (tt, b) in where.items() if tt == t]
            arr = f[t].arrays(br, library='np', entry_start=s, entry_stop=e)
            for w, (tt, b) in where.items():
                if tt == t:
                    cols[w].append(arr[b][sel])
        print('  %s: %d / %d' % (os.path.basename(fn), e, n), flush=True)

d = {w: np.concatenate(v) for w, v in cols.items()}
for w in WANT:
    d[w] = d[w].astype(np.int64 if w in ('Run', 'Ev') else np.float32)
key = d['Run'] * 10_000_000 + d['Ev']
ndup = len(key) - len(np.unique(key))
print('kept %d events; duplicate (Run, Ev) pairs among them: %d' % (len(key), ndup))
for w in ('E', 'Ne', 'Nmu', 'Age', 'Ze', 'Xc', 'Yc'):
    print('  %-4s %s' % (w, np.nanpercentile(d[w], [0, 1, 50, 99, 100]).round(3)))
np.savez_compressed(OUT, **d, n_total=np.int64(total), keep=np.float64(keep))
print('wrote %s (%.0f MB)' % (OUT, os.path.getsize(OUT) / 1e6))
