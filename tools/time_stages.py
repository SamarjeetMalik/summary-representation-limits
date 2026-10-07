"""Wall-clock cost of each audit step on this machine.

Runs the scripts that implement the steps on GRAPES-3 in a temporary copy of analysis/, so the
shipped results.json is not touched, and writes analysis/timing.json. Timings depend on the
machine and its load; the file records the core count and platform next to them.

    python tools/time_stages.py            all stages (the network stage takes the longest)
    python tools/time_stages.py --no-net   skip the network trainings of step 3
"""
import json, os, platform, shutil, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, 'analysis')
STAGES = [('floor', 'run_tmlr_checks.py', 'step 1: floor estimates with shower-level intervals and controls'),
          ('ladder', 'run_ladder.py', 'step 2: seven rungs, three seeds each, paired bootstrap, permutation control'),
          ('probe', 'run_probe_calibration.py', 'step 4: probe on fourteen outputs and controls, two columns each'),
          ('selection', 'run_selection_robustness.py', 'step 5: three selections, retraining, four seeds'),
          ('crossfit', 'run_shower_crossfit.py', 'steps 1, 2 and 5 again under the five-fold shower cross-fit'),
          ('networks', 'run_rendering.py', 'step 3: three CNN trainings on rendered images (the inputs being audited)')]

tmp = tempfile.mkdtemp(prefix='audit_timing_')
work = os.path.join(tmp, 'analysis')
shutil.copytree(ANA, work, ignore=shutil.ignore_patterns('__pycache__', 'boxlogs', '*.log'))
env = dict(os.environ, GRAPES3_DATA_DIR=os.path.join(ROOT, 'data'), PYTHONUNBUFFERED='1')
out = {'machine': {'cores': os.cpu_count(), 'platform': platform.platform(), 'processor': platform.machine()},
       'stages': {}}
for key, script, what in STAGES:
    if key == 'networks' and '--no-net' in sys.argv:
        continue
    t0 = time.time()
    r = subprocess.run([sys.executable, script], cwd=work, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode:
        sys.exit('%s failed:\n%s' % (script, r.stderr[-1500:]))
    out['stages'][key] = {'script': script, 'what': what, 'seconds': round(dt, 1)}
    print('%-10s %7.1f s   %s' % (key, dt, what), flush=True)
    json.dump(out, open(os.path.join(ANA, 'timing.json'), 'w'), indent=2)
shutil.rmtree(tmp, ignore_errors=True)
print('wrote analysis/timing.json')
