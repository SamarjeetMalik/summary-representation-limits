"""Seed study for the networks trained on rendered input.

The main rendering experiment (run_rendering.py, run_sweep.py) trains each network once, with
seed 42, and resolves differences between networks by a paired bootstrap over test events. That
does not include the variation between training runs. This script retrains the CNN on the
rendered image (row B), the CNN with the measured muon channel (row C) and the vision
transformer (row G) with further seeds, one process per seed, and records the spread.

    SEED_JOBS   processes run at once (default 4); each uses TORCH_THREADS threads

Extends results.json.  Run:  python analysis/run_render_seeds.py
"""
import glob, json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results.json')
SEEDS = [43, 44, 45, 46]                     # seed 42 is the main run
JOBS = int(os.environ.get('SEED_JOBS', '4'))

todo = [(s, f) for s in SEEDS for f in ('run_rendering.py', 'run_sweep.py')
        if not os.path.exists(os.path.join(HERE, 'seeds', '%s_%d.json'
                                           % ('render' if f == 'run_rendering.py' else 'sweep', s)))]
running = []
while todo or running:
    while todo and len(running) < JOBS:
        s, f = todo.pop(0)
        env = dict(os.environ, RENDER_SEED=str(s))
        env.setdefault('TORCH_THREADS', '8')
        log = open(os.path.join(HERE, 'seeds_%s_%d.log' % (f[4:-3], s)), 'w')
        running.append((subprocess.Popen([sys.executable, f], cwd=HERE, env=env, stdout=log,
                                         stderr=subprocess.STDOUT), f, s))
    p, f, s = running.pop(0)
    if p.wait() != 0:
        sys.exit('seed job failed: %s seed %d (see seeds_*.log)' % (f, s))

R = json.load(open(OUT))
main = {'B_rendered': R['rendering']['B_rendered'], 'C_rendered_muon': R['rendering']['C_rendered_muon'],
        'G_vit_patches': R['sweep']['G_vit_patches']}
vals = {k: [v] for k, v in main.items()}
for fn in sorted(glob.glob(os.path.join(HERE, 'seeds', '*.json'))):
    j = json.load(open(fn))
    for k in vals:
        if k in j:
            vals[k].append(j[k])
out = {'seeds': [42] + SEEDS, 'A_scalars4': R['rendering']['A_scalars4']}
for k, v in vals.items():
    out[k] = {'sigma': v, 'mean': round(float(np.mean(v)), 4), 'sd': round(float(np.std(v, ddof=1)), 4),
              'min': min(v), 'max': max(v)}
    print('%-18s n=%d  mean %.4f  sd %.4f  range %.4f-%.4f' % (k, len(v), out[k]['mean'], out[k]['sd'],
                                                                 min(v), max(v)))
R['render_seeds'] = out
json.dump(R, open(OUT, 'w'), indent=2)
print('extended %s' % OUT)
