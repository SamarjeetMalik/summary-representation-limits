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
PAPERS = ['manuscript']

ENV = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')


def banner(msg):
    print('\n' + '=' * 72 + '\n' + msg + '\n' + '=' * 72, flush=True)


def run_env(script, cwd, extra, label=None):
    """run() with extra environment variables (KCDC_GEN selects the interaction model)."""
    global ENV
    saved = ENV
    ENV = dict(saved, **extra)
    try:
        return run(script, cwd, label)
    finally:
        ENV = saved


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
                   ('run_shower_groups.py', 'run_shower_groups (~3 min)'),
                   ('run_ladder.py', 'run_ladder (~2 min)'),
                   ('run_floor_robust.py', 'run_floor_robust (~2 min)'),
                   ('run_regimes.py', 'run_regimes (~2 min)'),
                   ('run_diagnostics.py', 'run_diagnostics (~2 min)'),
                   ('run_cnn_curves.py', 'run_cnn_curves (~8 min, torch)'),
                   ('run_group_checks.py', 'run_group_checks (~3 min)'),
                   ('run_mechanism.py', 'run_mechanism (~1 min)'),
                   ('run_tmlr_checks.py', 'run_tmlr_checks (~1 min)'),
                   ('run_render_seeds.py', 'run_render_seeds (4 seeds x 2 scripts, torch)'),
                   ('run_synthetic.py', 'run_synthetic (~10 min, torch)'),
                   ('run_synthetic_cls.py', 'run_synthetic_cls (~15 min, torch)')):
        run(s, ANA, lbl)
    # KCDC arm: needs data/kcdc/kcdc_kascade_qgs4.npz (public KCDC download + convert_kcdc.py)
    # shower-disjoint cross-fit: five fold runs of the networks, then the tabular cross-fit
    for k in range(5):
        subprocess.run([sys.executable, 'run_rendering.py'], cwd=ANA, env=dict(ENV, RENDER_FOLD=str(k)),
                       capture_output=True)
    run('run_shower_crossfit.py', ANA, 'run_shower_crossfit (~5 min; needs the five fold runs)')
    run('run_crossfit_secondary.py', ANA, 'run_crossfit_secondary (~4 min)')
    run('run_selection_fixedpop.py', ANA, 'run_selection_fixedpop (~5 min)')
    run('run_cqr_shower.py', ANA, 'run_cqr_shower (~1 min)')
    if os.environ.get('SKIP_KCDC'):                 # KASCADE is untouched by GRAPES-3 changes
        print('  SKIP_KCDC set: keeping analysis/kcdc/*.json as they are')
        return
    if os.path.exists(os.path.join(ROOT, 'data', 'kcdc', 'kcdc_kascade_qgs4.npz')):
        run(os.path.join('kcdc', 'run_kcdc.py'), ANA, 'run_kcdc (~1 min on 32 cores)')
        if os.path.exists(os.path.join(ROOT, 'data', 'kcdc', 'kcdc_station_features.npz')):
            run(os.path.join('kcdc', 'run_kcdc_floor.py'), ANA, 'run_kcdc_floor (~15 min)')
            run(os.path.join('kcdc', 'run_kcdc_floor_sens.py'), ANA, 'run_kcdc_floor_sens (~1 min)')
        if os.path.exists(os.path.join(ROOT, 'data', 'kcdc', 'generators', 'perstation_epos.npz')):
            run(os.path.join('kcdc', 'run_kcdc_generators.py'), ANA, 'run_kcdc_generators (~1 min)')
        if os.path.exists(os.path.join(ROOT, 'data', 'kcdc', 'kcdc_station_dense.npz')):
            run(os.path.join('kcdc', 'run_kcdc_cnn.py'), ANA, 'run_kcdc_cnn (~10 min on one GPU)')
        # the physics-journal arm: per-decade audit on all three interaction models, station-level
        # audit on EPOS-LHC and SIBYLL-2.3d (KCDC_GEN), and the label-free checks on measured data
        for g in ('', 'epos', 'sibyll'):
            tag = g or 'qgs4'
            if os.path.exists(os.path.join(ROOT, 'data', 'kcdc', 'kcdc_kascade_%s.npz' % tag)):
                run_env(os.path.join('kcdc', 'run_kcdc_energy.py'), ANA, {'KCDC_GEN': g}, 'run_kcdc_energy %s (~10 min)' % tag)
            if g and os.path.exists(os.path.join(ROOT, 'data', 'kcdc', 'kcdc_station_features_%s.npz' % g)):
                run_env(os.path.join('kcdc', 'run_kcdc_floor.py'), ANA, {'KCDC_GEN': g}, 'run_kcdc_floor %s (~15 min)' % g)
            for dec in ('', '14-15', '15-16', '16-17'):     # floor accuracy on a semi-synthetic known-floor target
                src = ('kcdc_kascade_%s.npz' % tag) if dec else ('kcdc_station_features%s.npz' % ('_' + g if g else ''))
                if os.path.exists(os.path.join(ROOT, 'data', 'kcdc', src)):
                    run_env(os.path.join('kcdc', 'run_kcdc_floor_benchmark.py'), ANA, {'KCDC_GEN': g, 'KCDC_DECADE': dec},
                            'run_kcdc_floor_benchmark %s %s (~10 min)' % (tag, dec or 'stations'))
            for dec in ('', '14-15', '15-16', '16-17'):     # floor-estimate sensitivity, per model and sample
                src = ('kcdc_kascade_%s.npz' % tag) if dec else ('kcdc_station_features%s.npz' % ('_' + g if g else ''))
                if (g or dec) and os.path.exists(os.path.join(ROOT, 'data', 'kcdc', src)):
                    run_env(os.path.join('kcdc', 'run_kcdc_floor_sens.py'), ANA, {'KCDC_GEN': g, 'KCDC_DECADE': dec},
                            'run_kcdc_floor_sens %s %s (~3 min)' % (tag, dec or 'stations'))
            for lge in ('6-7',):                            # station-level audit one decade higher
                wf = 'kcdc_station_features%s_e%s.npz' % ('_' + g if g else '', lge.replace('-', ''))
                if os.path.exists(os.path.join(ROOT, 'data', 'kcdc', wf)):
                    run_env(os.path.join('kcdc', 'run_kcdc_floor.py'), ANA, {'KCDC_GEN': g, 'KCDC_LGE': lge},
                            'run_kcdc_floor %s %s (~10 min)' % (tag, lge))
                    run_env(os.path.join('kcdc', 'run_kcdc_cnn.py'), ANA, {'KCDC_GEN': g, 'KCDC_LGE': lge},
                            'run_kcdc_cnn %s %s (~10 min on one GPU)' % (tag, lge))
            if g and os.path.exists(os.path.join(ROOT, 'data', 'kcdc', 'kcdc_station_dense_%s.npz' % g)):
                run_env(os.path.join('kcdc', 'run_kcdc_cnn.py'), ANA, {'KCDC_GEN': g}, 'run_kcdc_cnn %s (~10 min on one GPU)' % g)
        if os.path.exists(os.path.join(ROOT, 'data', 'kcdc', 'kcdc_measured.npz')):
            run(os.path.join('kcdc', 'run_kcdc_measured.py'), ANA, 'run_kcdc_measured (~7 min)')
            run(os.path.join('kcdc', 'check_measured_threshold.py'), ANA, 'check_measured_threshold (~1 min)')
        if os.path.exists(os.path.join(ROOT, 'data', 'kcdc', 'kcdc_kascade_epos.npz')):
            run_env(os.path.join('kcdc', 'run_kcdc_measured.py'), ANA, {'MEASURED_TEST': 'epos'},
                    'run_kcdc_measured closure test (~17 min)')
        run(os.path.join('kcdc', 'check_station_samples.py'), ANA, 'check_station_samples (~1 min)')
        for g in ('', 'epos', 'sibyll'):                    # the audit without knowing the true decade
            if os.path.exists(os.path.join(ROOT, 'data', 'kcdc', 'kcdc_kascade_%s.npz' % (g or 'qgs4'))):
                for flat in ('', '1'):
                    run_env(os.path.join('kcdc', 'run_kcdc_fullrange.py'), ANA, {'KCDC_GEN': g, 'FLAT': flat},
                            'run_kcdc_fullrange %s%s (~15 min)' % (g or 'qgs4', ' flat' if flat else ''))
        if os.path.exists(os.path.join(ROOT, 'data', 'kcdc', 'kcdc_measured.npz')):
            run(os.path.join('kcdc', 'make_figure_inputs_epjc.py'), ANA, 'make_figure_inputs_epjc (~2 min)')
    else:
        print('  KCDC npz absent: skipping the KCDC arm (see analysis/kcdc/convert_kcdc.py)')


def tables():
    banner('TABLES + prose macros from results.json')
    for s in ('make_tables.py', 'make_tables_extra.py', 'make_tables_compact.py',
              'make_tables_render.py', 'make_tables_journal.py', 'make_tables_tmlr.py',
              'make_tables_epjc.py', 'dedupe_values.py'):
        run(s, ANA)


def figures():
    banner('FIGURES')
    run('make_figures.py', FIG)
    run('make_figures_extra.py', FIG)
    run('make_figures_review.py', FIG)
    run('make_figures_epjc.py', FIG)


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
        for i in range(3 if p == 'manuscript' else 2):   # second pass resolves references
            subprocess.run(['pdflatex', '-interaction=nonstopmode', p + '.tex'], cwd=ROOT,
                           capture_output=True)
            if p == 'manuscript' and i == 0:   # both take their references from a .bib file
                subprocess.run(['bibtex', p], cwd=ROOT, capture_output=True)
        log = open(os.path.join(ROOT, p + '.log'), encoding='latin-1').read()
        errs = log.count('\n!')
        undef = log.lower().count('undefined')
        over = log.count('Overfull')
        print('  %-12s errors=%d undefined=%d overfull=%d' % (p, errs, undef, over))
        if p == 'manuscript':                         # needs the .aux, so before clean-up
            r = subprocess.run([sys.executable, os.path.join('tools', 'check_floats.py'), p], cwd=ROOT,
                               capture_output=True, text=True)
            print('  ' + r.stdout.strip().splitlines()[-1])
            if r.returncode:
                sys.exit(r.stdout)
        for ext in ('.aux', '.log', '.out', '.blg'):
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
