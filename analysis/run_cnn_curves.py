"""Learning curves for the CNN on rendered images (rows B and C of the rendering table).

Trains the same CNN as run_rendering.py on 3,000 / 10,000 / 30,000 events (the 60,000-event
point is row B / row C itself) and scores it on the matched test events. If the rendered-input
network were data-starved, its error would still be falling at 60,000; if the representation
is the limit, it flattens onto the four-scalar floor. Geometry, rendering and network are
copied verbatim from run_rendering.py so that script stays untouched.

Extends results.json.  Run:  python analysis/run_cnn_curves.py
"""
import json, os, math, warnings
import numpy as np
warnings.filterwarnings('ignore')
from scipy.special import gammaln
import torch
import torch.nn as nn

import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results.json')
R = json.load(open(OUT))
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", max(1, (os.cpu_count() or 4) - 1))))

# ----------------------------------------------------------------- geometry
NROW, NCOL, PITCH = 16, 25, 8.0          # 400 scintillators, 8 m apart
R_MOLIERE = 103.0                        # metres, Ooty altitude 2200 m
gx = (np.arange(NCOL) - (NCOL - 1) / 2.0) * PITCH      # -96 .. 96 m
gy = (np.arange(NROW) - (NROW - 1) / 2.0) * PITCH      # -60 .. 60 m
GX, GY = np.meshgrid(gx, gy)                            # (16, 25)


def nkg_density(Ne, s, xc, yc):
    """NKG lateral distribution evaluated on the detector lattice.

    rho(r) = Ne/r_M^2 * C(s) * (r/r_M)^(s-2) * (1 + r/r_M)^(s-4.5)
    C(s)   = Gamma(4.5-s) / (2 pi Gamma(s) Gamma(4.5-2s))
    Returns (N, 16, 25) of log10(1 + rho).
    """
    r = np.sqrt((GX[None] - xc[:, None, None]) ** 2 +
                (GY[None] - yc[:, None, None]) ** 2)
    np.maximum(r, 1.0, out=r)                    # avoid the r->0 singularity
    lnC = gammaln(4.5 - s) - math.log(2 * math.pi) - gammaln(s) - gammaln(4.5 - 2 * s)
    x = r / R_MOLIERE
    lnrho = (np.log(Ne)[:, None, None] - 2 * math.log(R_MOLIERE) + lnC[:, None, None]
             + (s[:, None, None] - 2.0) * np.log(x)
             + (s[:, None, None] - 4.5) * np.log1p(x))
    return np.log10(1.0 + np.exp(lnrho)).astype(np.float32)


def t(a):
    """numpy -> torch without the broken from_numpy path."""
    # Zero-copy: numpy arrays expose the buffer protocol, so frombuffer can wrap
    # them directly. The previous tobytes()/bytearray() path made two full copies
    # and blew up on the two-channel tensor.
    a = np.ascontiguousarray(a, dtype=np.float32)
    return torch.frombuffer(a, dtype=torch.float32).reshape(a.shape)



d = C.load_root(); c = C.load_csv(); y = d['y']
r_idx, _ = C.match(d, c)
test = np.zeros(len(y), bool); test[r_idx] = True
pool = np.where(~test)[0]; te = r_idx
NFULL = R['rendering']['n_train']
full = np.random.default_rng(42).choice(pool, NFULL, replace=False)      # as run_rendering.py

img_all = nkg_density(d['NKGSize'][full], d['Age'][full], d['NKGX'][full], d['NKGY'][full])
img_te = nkg_density(d['NKGSize'][te], d['Age'][te], d['NKGX'][te], d['NKGY'][te])
mu16 = np.log10(1.0 + d['MuonsLarge'].astype(np.float32))
mu_all = np.repeat(mu16[full][:, :, None], NCOL, axis=2).astype(np.float32)
mu_te = np.repeat(mu16[te][:, :, None], NCOL, axis=2).astype(np.float32)
m_, s_ = img_all.mean(), img_all.std(); img_all = (img_all - m_) / s_; img_te = (img_te - m_) / s_
mm, ms = mu_all.mean(), mu_all.std(); mu_all = (mu_all - mm) / ms; mu_te = (mu_te - mm) / ms

class Net(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(cin, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(96, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


CURVES = {}
PREDS = {}


def predict_batched(net, X, bs=1024):
    """Chunked inference.

    Two constraints: a full-test forward pass allocates ~1.7 GB at the first
    conv layer, and this torch build cannot talk to NumPy 2.x in EITHER
    direction, so .numpy() is unavailable as well as from_numpy(). Tensors go
    out via .tolist().
    """
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.extend(net(t(X[i:i + bs])).tolist())
    return np.asarray(out, dtype=np.float32)



def fit(Xtr, ytr, Xte, epochs=28, bs=256, seed=42):
    torch.manual_seed(seed)
    net = Net(Xtr.shape[1]); ym, ys = ytr.mean(), ytr.std()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * math.ceil(len(Xtr) / bs))
    Xt, Yt = t(Xtr), t((ytr - ym) / ys); lossf = nn.MSELoss()
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            j = perm[i:i + bs]; opt.zero_grad(); L = lossf(net(Xt[j]), Yt[j]); L.backward(); opt.step(); sched.step()
    net.eval()
    return predict_batched(net, Xte) * ys + ym


out = {'n': [], 'B': [], 'C': []}
for n in (3000, 10000, 30000):
    sub = np.arange(n)                     # nested subsets of the 60,000-event training set
    pB = fit(img_all[sub][:, None], y[full][sub], img_te[:, None])
    pC = fit(np.stack([img_all[sub], mu_all[sub]], 1), y[full][sub], np.stack([img_te, mu_te], 1))
    out['n'].append(n); out['B'].append(round(float((pB - y[te]).std()), 4)); out['C'].append(round(float((pC - y[te]).std()), 4))
    print('  n=%6d  B %.4f  C %.4f' % (n, out['B'][-1], out['C'][-1]), flush=True)
out['n'].append(NFULL); out['B'].append(R['rendering']['B_rendered']); out['C'].append(R['rendering']['C_rendered_muon'])
R['cnn_curves'] = out
json.dump(R, open(OUT, 'w'), indent=2)
print('extended %s' % OUT)
