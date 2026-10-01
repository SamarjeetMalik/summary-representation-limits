"""Shared data loading for the R/PS 2026 Research Note.

Design note
-----------
Every estimator in the manuscript is evaluated on ONE identical population: the
34,356 events that appear in both the ROOT simulation files and the archived
prediction table (grapes3_ml_results.csv). Machine-learning models are trained on
the complement (137,420 events) and evaluated on that matched set; the NKG
analytic reconstruction and the archived CNN/ViT predictions are read directly
from the CSV for the same events.

This removes the population mismatch that would otherwise make a sigma table
incomparable across estimators.
"""
import csv
import os
import numpy as np
import uproot
import awkward as ak

# Data location. Resolved in this order:
#   1. the GRAPES3_DATA_DIR environment variable, if set;
#   2. the data/ directory at the root of this bundle (the default).
# Nothing here depends on the machine the bundle was built on.
_BUNDLE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get('GRAPES3_DATA_DIR', os.path.join(_BUNDLE, 'data'))

ROOT_FILES = [
    os.path.join(DATA_DIR, 'feature_24.6_5.0.root'),
    os.path.join(DATA_DIR, 'feature_24.6_4.8.root'),
]
CSV_FILE = os.path.join(DATA_DIR, 'grapes3_ml_results.csv')

for _f in ROOT_FILES + [CSV_FILE]:
    if not os.path.exists(_f):
        raise FileNotFoundError(
            'Missing data file: %s\n'
            'Expected the three files in %s (see DATASETS.md), or set '
            'GRAPES3_DATA_DIR to the directory that holds them.' % (_f, DATA_DIR))

SPECIES = {14: 'proton', 402: 'helium', 1407: 'nitrogen',
           2713: 'aluminium', 5626: 'iron'}
SEEDS = [42, 7, 13, 21, 37, 55, 71, 89]

FEATURE_NAMES = ['log_Ne', 'age', 'log_Nmu', 'muon_ratio',
                 'theta', 'cos_theta', 'Xc', 'Yc', 'R_core']
SCALAR4 = ['log_Ne', 'age', 'Xc', 'Yc']


def load_root():
    """Return a dict of per-event arrays over all 171,776 simulated events."""
    cols = {}
    for fn in ROOT_FILES:
        t = uproot.open(fn)['param']
        for b in ['PrimaryEnergy', 'PrimaryID', 'Theta1', 'NKGX', 'NKGY',
                  'NKGSize', 'Age', 'nMuons']:
            cols.setdefault(b, []).append(t[b].array(library='np'))
        cols.setdefault('MuonsLarge', []).append(
            ak.to_numpy(t['MuonsLarge'].array()))
    d = {k: np.concatenate(v) for k, v in cols.items()}

    d['y'] = np.log10(d['PrimaryEnergy'])          # log10(E/GeV)
    d['log_Ne'] = np.log10(d['NKGSize'])
    d['mu_total'] = d['MuonsLarge'].sum(1).astype(float)
    d['log_Nmu'] = np.log10(1.0 + d['mu_total'])
    return d


def build_features(d):
    """The nine-feature summary representation."""
    return np.column_stack([
        d['log_Ne'], d['Age'], d['log_Nmu'], d['log_Nmu'] - d['log_Ne'],
        d['Theta1'], np.cos(np.radians(d['Theta1'])),
        d['NKGX'], d['NKGY'], np.hypot(d['NKGX'], d['NKGY']),
    ])


def load_csv():
    """Archived predictions: truth, CNN/ViT output, NKG analytic output, age."""
    rows = list(csv.DictReader(open(CSV_FILE)))
    return {
        'true': np.array([float(r['true_logE']) for r in rows]),
        'cnnvit': np.array([float(r['ml_pred_logE']) for r in rows]),
        'nkg': np.array([float(r['nkg_pred_logE']) for r in rows]),
        'age': np.array([float(r['age']) for r in rows]),
        'log_Ne': np.array([float(r['log_nkg_size']) for r in rows]),
        'Xc': np.array([float(r['core_x_true']) for r in rows]),
        'Yc': np.array([float(r['core_y_true']) for r in rows]),
    }


def match(d, c, tol=1e-12):
    """Match each archived row to its simulated event: Age exactly, log10 E within tol.

    An exact match on log10 E (the original protocol) depended on the last bit of a
    computed logarithm: it found 33,876 of 34,356 rows with one numpy build and 33,874 with
    another, and the 480 "unmatched" rows were blamed on round-trip loss in the CSV. With a
    1e-12 tolerance every row matches exactly one event, on any build, and that is
    asserted below. log_nkg_size is recomputed differently in the CSV and is not a key.
    Returns (root_index, csv_index) aligned pairs.
    """
    by_age = {}
    for i, a in enumerate(d['Age']):
        by_age.setdefault(a, []).append(i)
    r_idx, c_idx = [], []
    for j, (e, a) in enumerate(zip(c['true'], c['age'])):
        cand = [i for i in by_age.get(a, ()) if abs(d['y'][i] - e) <= tol]
        assert len(cand) <= 1, 'archived row %d matches %d events' % (j, len(cand))
        if cand:
            r_idx.append(cand[0])
            c_idx.append(j)
    assert len(set(r_idx)) == len(r_idx), 'two archived rows matched the same event'
    return np.array(r_idx), np.array(c_idx)


def sigma(pred, truth):
    return float((np.asarray(pred) - np.asarray(truth)).std())


def bias(pred, truth):
    return float((np.asarray(pred) - np.asarray(truth)).mean())


def r2(pred, truth):
    pred, truth = np.asarray(pred), np.asarray(truth)
    return float(1 - ((truth - pred) ** 2).sum() / ((truth - truth.mean()) ** 2).sum())


def bootstrap_ci(values, n=4000, alpha=0.05, seed=0):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = [rng.choice(v, len(v), replace=True).mean() for _ in range(n)]
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
