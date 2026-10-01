"""One entry point for the whole paper.

    python run_pipeline.py --check     environment + data integrity only (~1 min)
    python run_pipeline.py             regenerate tables, figures, audit, PDFs (~3 min)
    python run_pipeline.py --full      recompute every result from the data first (~2-3 h CPU)

The default (fast) mode uses the shipped analysis/results.json, which already holds
every number in the papers. --full rebuilds that file from the ROOT data.

Order matters and is encoded here so nobody has to remember it:
  * run_all.py REWRITES results.json from scratch; the other run_* scripts EXTEND it.
  * make_tables.py writes tables/values.tex fresh; the other make_tables_* APPEND to it.
Running any of them out of order produces stale or duplicated numbers.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
ANA = os.path.join(ROOT, 'analysis')
FIG = os.path.join(ROOT, 'figures')
DATA = os.environ.get('GRAPES3_DATA_DIR', os.path.join(ROOT, 'data'))
PAPERS = ['rps_short', 'mlst', 'main', 'ml4ps']

ENV = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')


def banner(msg):
    print('\n' + '=' * 72 + '\n' + msg + '\n' + '=' * 72, flush=True)


def run(script, cwd, label=None):
    t0 = time.time()
    print('  -> %-28s' % (label or script), end=' ', flush=True)
    r = subprocess.run([sys.executable, script], cwd=cwd, env=ENV,
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode != 0:
        print('FAILED')
        # the NumPy-1.x/2.x ABI warning is noise; show only the real error tail
        tail = [l for l in r.stderr.splitlines()
                if 'NumPy' not in l and 'pybind11' not in l and l.strip()]
        print('\n'.join(tail[-25:]))
        sys.exit('stage failed: %s' % script)
    print('ok  (%.0fs)' % (time.time() - t0), flush=True)
    return r.stdout


# --------------------------------------------------------------------- check
def check():
    banner('CHECK: environment and data integrity')
    print('  python %s' % sys.version.split()[0])
    missing = []
    for m in ('numpy', 'scipy', 'sklearn', 'uproot', 'awkward', 'matplotlib'):
        try:
            __import__(m)
        except Exception:
            missing.append(m)
    try:
        import torch  # noqa: F401  only needed for --full
        print('  torch present (needed only for --full)')
    except Exception:
        print('  torch ABSENT: fine for the default mode, required for --full')
    if missing:
        sys.exit('  missing packages: %s  ->  pip install -r requirements.txt' % missing)
    print('  core packages present')

    sums = os.path.join(DATA, 'SHA256SUMS.txt')
    if not os.path.exists(sums):
        sys.exit('  no %s -- is GRAPES3_DATA_DIR right?' % sums)
    for line in open(sums):
        h, name = line.split()
        p = os.path.join(DATA, name)
        if not os.path.exists(p):
            sys.exit('  MISSING data file %s' % p)
        got = hashlib.sha256(open(p, 'rb').read()).hexdigest()
        print('  %-28s %s' % (name, 'sha256 OK' if got == h else 'HASH MISMATCH'))
        if got != h:
            sys.exit('  data file corrupted or replaced: %s' % name)

    sys.path.insert(0, ANA)
    import common as C
    d = C.load_root()
    c = C.load_csv()
    r_idx, _ = C.match(d, c)
    print('  ROOT events     %d   (expect 171776)' % len(d['y']))
    print('  CSV rows        %d   (expect 34356)' % len(c['true']))
    print('  matched events  %d   (expect 34356)' % len(r_idx))
    if (len(d['y']), len(c['true']), len(r_idx)) != (171776, 34356, 34356):
        sys.exit('  event counts do not match the paper -- stop and investigate')
    print('  all checks passed')


# --------------------------------------------------------------------- stages
def compute():
    banner('COMPUTE: rebuilding results.json from the data (slow)')
    print('  run_all rewrites results.json; each later script extends it.')
    for s, lbl in (('run_all.py', 'run_all (~35 min, RF slow)'),
                   ('run_extra.py', 'run_extra (~10 min)'),
                   ('run_eventci.py', 'run_eventci (~5 min)'),
                   ('run_nkg_baseline.py', 'run_nkg_baseline (~1 min)'),
                   ('run_bayes.py', 'run_bayes (~1 min)'),
                   ('run_rendering.py', 'run_rendering (~30 min, torch)'),
                   ('run_rendering_controls.py', 'run_rendering_controls (~2 min)'),
                   ('run_sweep.py', 'run_sweep (~35 min, torch)'),
                   ('run_probe_calibration.py', 'run_probe_calibration (~5 min; needs rendering preds)'),
                   ('run_selection_robustness.py', 'run_selection_robustness (~5 min)'),
                   ('run_shower_groups.py', 'run_shower_groups (~3 min)')):
        run(s, ANA, lbl)
    # KCDC arm: needs data/kcdc/kcdc_kascade_qgs4.npz (public KCDC download + convert_kcdc.py)
    if os.path.exists(os.path.join(ROOT, 'data', 'kcdc', 'kcdc_kascade_qgs4.npz')):
        run(os.path.join('kcdc', 'run_kcdc.py'), ANA, 'run_kcdc (~1 min on 32 cores)')
    else:
        print('  KCDC npz absent: skipping the KCDC arm (see analysis/kcdc/convert_kcdc.py)')


def tables():
    banner('TABLES + prose macros from results.json')
    for s in ('make_tables.py', 'make_tables_extra.py', 'make_tables_compact.py',
              'make_tables_render.py', 'make_tables_journal.py', 'dedupe_values.py'):
        run(s, ANA)


def figures():
    banner('FIGURES')
    run('make_figures.py', FIG)
    run('make_figures_extra.py', FIG)


def verify():
    banner('AUDIT: every prose number traced to results.json')
    for p in PAPERS:
        r = subprocess.run([sys.executable, os.path.join('analysis', 'verify_numbers.py'),
                            p + '.tex'], cwd=ROOT, env=ENV, capture_output=True,
                           text=True, encoding='utf-8', errors='replace')
        last = [l for l in r.stdout.splitlines() if 'value(s) need' in l or 'all prose' in l]
        print('  %-12s %s' % (p, last[0].strip() if last else '(no summary)'))
    r = subprocess.run([sys.executable, os.path.join('tools', 'check_refs.py')], cwd=ROOT,
                       env=ENV, capture_output=True, text=True, encoding='utf-8', errors='replace')
    print(r.stdout.rstrip())
    if r.returncode:
        sys.exit('  check_refs failed: broken \\ref, uncited \\bibitem or unknown cite key')


def latex():
    banner('LATEX')
    if not shutil.which('pdflatex'):
        print('  pdflatex not on PATH -- skipping. Upload overleaf/RPS2026_overleaf.zip to')
        print('  Overleaf instead, or rebuild it with  python tools/build_overleaf_zip.py')
        return
    for p in PAPERS:
        for _ in range(2):                       # second pass resolves references
            subprocess.run(['pdflatex', '-interaction=nonstopmode', p + '.tex'], cwd=ROOT,
                           capture_output=True)
        log = open(os.path.join(ROOT, p + '.log'), encoding='latin-1').read()
        errs = log.count('\n!')
        undef = log.lower().count('undefined')
        over = log.count('Overfull')
        print('  %-12s errors=%d undefined=%d overfull=%d' % (p, errs, undef, over))
        for ext in ('.aux', '.log', '.out'):
            try:
                os.remove(os.path.join(ROOT, p + ext))
            except OSError:
                pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true', help='environment + data only')
    ap.add_argument('--full', action='store_true', help='recompute results.json first')
    a = ap.parse_args()

    check()
    if a.check:
        sys.exit(0)
    if a.full:
        compute()
    tables()
    figures()
    verify()
    latex()
    banner('DONE')
