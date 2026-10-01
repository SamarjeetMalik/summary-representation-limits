"""How the age-energy correlation depends on the width of the shower-size window.

run_all.py reports corr(s, log10 E) in three windows of equal width (0.10 in log10 Ne) at
different positions, and over the whole 0.4-wide slice. That shows the correlation is stronger
inside narrow windows, but not how it depends on width. This script measures it: for window
widths 0.025 .. 0.40 centred at several positions inside the slice, the correlation between age
and log10 E among the events in the window (mean and spread over centres).

Extends results.json.  Run:  python analysis/run_mechanism.py
"""
import json, os
import numpy as np

import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results.json')
R = json.load(open(OUT))
d = C.load_root(); y = d['y']; lne, s = d['log_Ne'], d['Age']

WIDTHS = [0.025, 0.05, 0.10, 0.20, 0.40]
rows = []
for w in WIDTHS:
    centres = np.arange(4.8 + w / 2, 5.2 - w / 2 + 1e-9, 0.025) if w < 0.4 else np.array([5.0])
    cs = []
    for c in centres:
        m = (lne >= c - w / 2) & (lne < c + w / 2)
        if m.sum() > 2000:
            cs.append(float(np.corrcoef(s[m], y[m])[0, 1]))
    rows.append({'width': w, 'n_centres': len(cs), 'corr_mean': round(float(np.mean(cs)), 3),
                 'corr_min': round(float(np.min(cs)), 3), 'corr_max': round(float(np.max(cs)), 3)})
    print('  width %.3f: corr %.3f (range %.3f-%.3f over %d centres)' % (w, *[rows[-1][k] for k in
          ('corr_mean', 'corr_min', 'corr_max')], len(cs)))
R['mechanism_width'] = rows
json.dump(R, open(OUT, 'w'), indent=2)
print('extended %s' % OUT)
