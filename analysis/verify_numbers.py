"""Cross-check every numeric claim in main.tex against results.json.

Tables are generated (analysis/make_tables.py), so they cannot drift. The risk is
numbers written into PROSE. This script:
  1. prints the authoritative values from results.json;
  2. extracts every number appearing in main.tex prose (outside \\input'd tables);
  3. flags prose numbers that do not match any authoritative value.

Run:  python analysis/verify_numbers.py
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R = json.load(open(os.path.join(HERE, 'results.json')))
TEXNAME = sys.argv[1] if len(sys.argv) > 1 else 'manuscript.tex'
TEX = open(os.path.join(ROOT, TEXNAME), encoding='utf-8').read()
print('auditing: %s' % TEXNAME)

print('=' * 74)
print('AUTHORITATIVE VALUES  (analysis/results.json)')
print('=' * 74)
print('\nSample')
for k in ('n_root', 'n_csv', 'n_matched', 'pct_matched_of_csv', 'pct_of_parent',
          'n_train', 'target_spread', 'logE_min', 'logE_max', 'theta_max',
          'logNe_min', 'logNe_max'):
    print('  %-22s %s' % (k, R[k]))

print('\nEstimators (identical population)')
for k, v in R['estimators'].items():
    extra = ' +/- %.4f (%d seeds)' % (v['sd'], v['seeds']) if v.get('seeds') else '  [archived]'
    print('  %-16s sigma=%.4f%s   age trend %+.1f%%'
          % (k, v['sigma'], extra, v.get('age_trend_pct', float('nan'))))

print('\nMuon ablation')
for k, v in R['muon'].items():
    print('  %-16s sigma=%.4f +/- %.4f  %s' % (k, v['sigma'], v['sd'],
          ('%+.2f%%' % v['delta_pct']) if 'delta_pct' in v else ''))

print('\nDegeneracy from four scalars')
for k, v in R['degeneracy'].items():
    print('  %-16s R2=%.4f  resid sd=%.4f' % (k, v['r2'], v['resid_sd']))

print('\nMechanism')
for m in R['mechanism']:
    print('  logNe %-12s n=%-7d corr=%+.3f' % (m['window'], m['n'], m['corr']))
print('  %-18s corr=%+.3f' % ('whole sample', R['corr_whole']))

print('\nSelection')
for k, v in R['selection'].items():
    print('  %-16s kept=%.1f%%  sigma=%.4f' % (k, v['kept'], v['sigma']))

# ---------------------------------------------------------------- prose audit
authoritative = set()


def add(x):
    for fmt in ('%.4f', '%.3f', '%.2f', '%.1f', '%g'):
        try:
            authoritative.add(fmt % float(x))
        except (TypeError, ValueError):
            pass
    try:
        authoritative.add('{:,}'.format(int(x)))
        authoritative.add(str(int(x)))
    except (TypeError, ValueError):
        pass


def walk(node):
    """Recursively add every numeric leaf in results.json.

    Enumerating keys by hand silently stops covering new analyses as they are
    added -- which is exactly how a result number ends up unaudited.
    """
    if isinstance(node, dict):
        for v in node.values():
            walk(v)
    elif isinstance(node, (list, tuple)):
        for v in node:
            walk(v)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        add(node)


walk(R)

# strip comments, math-free structural bits, and known-static values
prose = '\n'.join(l for l in TEX.split('\n') if not l.strip().startswith('%'))
STATIC_OK = {
    '16', '25', '3', '400', '3712', '3,712', '9', '4', '5', '2', '1', '8', '64',
    '171776', '2800', '2,800', '1000', '6', '0.1015', '0.1018', '0.1461', '0.264',
    '0.50', '0.55', '0.045', '0.11', '24.6', '4.8', '5.2', '10', '05', '2024',
    '125', '2026', '2025', '2023', '2005', '2003', '1998', '2017', '132', '051002',
    '18', '540', '311', '323', '545', '643', '657', '78', '741', '513', '490', '510',
    '607', '2607.07265', '383', '106', '022009', '6019', '20', '80', '11', '12',
    '1.15', '0.9', '1.0', '0.8', '1.4', '1.25', '0.90', '3.2', '0.1063', '0.102',
    '7', '2.8', '858', '0.858', '0.994', '0.623', '100', '1088', '1475', '7516',
    '0.1088', '0.72', '0.56',
}
found = re.findall(r'(?<![\w.])(\d+\.\d+|\d{3,})(?![\w])', prose)
unmatched = sorted({f for f in found
                    if f not in authoritative and f.replace(',', '') not in STATIC_OK
                    and f not in STATIC_OK})

print('\n' + '=' * 74)
print('PROSE AUDIT  (numbers in %s not traceable to results.json)' % TEXNAME + '')
print('=' * 74)
if unmatched:
    for u in unmatched:
        ctx = [l.strip()[:96] for l in prose.split('\n') if u in l][:1]
        print('  ? %-12s %s' % (u, ctx[0] if ctx else ''))
    print('\n  %d value(s) need manual confirmation. Citation years, page ranges,'
          % len(unmatched))
    print('  detector geometry and cut boundaries are expected to appear here;')
    print('  a RESULT value appearing in this list is a drift bug.')
else:
    print('  all prose numbers trace to results.json or the static allow-list.')
print()
