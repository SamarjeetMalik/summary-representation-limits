"""The rendering experiment: manufacture the failure on demand.

The archived CNN/ViT was trained on 'detector images' that had been synthesised by
evaluating the fitted NKG lateral function at detector coordinates. We reconstruct
that pipeline from scratch and train our own CNN on it.

PRE-REGISTERED PREDICTION (stated before running):
  A  GBM on the four fit scalars .............. sigma ~ 0.1234
  B  CNN on rendered density image ............ sigma ~ 0.1234   <- the claim
  C  CNN on rendered density + real muons ..... sigma ~ 0.1063   <- archived value
  D  GBM on the nine summary features ......... sigma ~ 0.1019

B ~ A is the central claim: rendering four scalars across 400 pixels adds no
information, so a CNN on the image cannot beat a tabular model on the scalars.
C ~ 0.1063 would mean we predicted the archived model's result from first
principles, its residual being exactly the one genuinely measured channel.

torch 2.3.1 here was built against NumPy 1.x while NumPy 2.2 is installed, so
torch.from_numpy is unavailable; tensors are built via torch.frombuffer.

Run:  python analysis/run_rendering.py
"""
import gc, json, os, math, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
from scipy.special import gammaln
from sklearn.ensemble import HistGradientBoostingRegressor as GBR
import torch
import torch.nn as nn

import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results.json')
R = json.load(open(OUT))
torch.manual_seed(42)
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", max(1, (os.cpu_count() or 4) - 1))))   # cap on shared machines

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


# ----------------------------------------------------------------- data
print('loading...')
d = C.load_root()
c = C.load_csv()
X9 = C.build_features(d)
y = d['y']
r_idx, _ = C.match(d, c)
test_mask = np.zeros(len(y), bool); test_mask[r_idx] = True
pool = np.where(~test_mask)[0]

NTRAIN = 60000                                   # memory-bounded subsample
rng = np.random.default_rng(42)
tr = rng.choice(pool, NTRAIN, replace=False)
te = r_idx
print('  train %d (subsample of %d), test %d' % (len(tr), len(pool), len(te)))

S4 = np.column_stack([d['log_Ne'], d['Age'], d['NKGX'], d['NKGY']])

print('rendering NKG images...')
t0 = time.time()
img_tr = nkg_density(d['NKGSize'][tr], d['Age'][tr], d['NKGX'][tr], d['NKGY'][tr])
img_te = nkg_density(d['NKGSize'][te], d['Age'][te], d['NKGX'][te], d['NKGY'][te])
print('  %.1fs, train %s = %.0f MB' % (time.time() - t0, img_tr.shape, img_tr.nbytes / 1e6))
print('  density log10(1+rho): min %.3f  max %.3f  mean %.3f'
      % (img_tr.min(), img_tr.max(), img_tr.mean()))

mu16 = np.log10(1.0 + d['MuonsLarge'].astype(np.float32))
mu_tr = np.repeat(mu16[tr][:, :, None], NCOL, axis=2).astype(np.float32)
mu_te = np.repeat(mu16[te][:, :, None], NCOL, axis=2).astype(np.float32)

m_, s_ = img_tr.mean(), img_tr.std()
img_tr = (img_tr - m_) / s_; img_te = (img_te - m_) / s_
mm, ms = mu_tr.mean(), mu_tr.std()
mu_tr = (mu_tr - mm) / ms; mu_te = (mu_te - mm) / ms
ym, ys = y[tr].mean(), y[tr].std()


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

RES = {}

# ----------------------------------------------------------------- tabular
def tab(name, X, key):
    p = GBR(max_iter=600, learning_rate=.06, random_state=42).fit(X[tr], y[tr]).predict(X[te])
    RES[key] = round(C.sigma(p, y[te]), 4)
    print('  %-42s sigma = %.4f' % (name, RES[key]))


print('\ntabular baselines on the same %d training events:' % NTRAIN)
tab('A  four fit scalars (logNe, s, Xc, Yc)', S4, 'A_scalars4')
tab('D  nine summary features', X9, 'D_nine')


# ----------------------------------------------------------------- CNN
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


def train_cnn(name, key, Xtr, Xte, epochs=28, bs=256, seed=42, record=True):
    torch.manual_seed(seed)
    net = Net(Xtr.shape[1])
    npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-3, total_steps=epochs * math.ceil(len(Xtr) / bs))
    Yt = t((y[tr] - ym) / ys)
    Xt = t(Xtr)
    lossf = nn.MSELoss()
    print('\n  %s  (%d params, seed %d)' % (name, npar, seed))
    curve = {'epoch': [], 'sigma': [], 'train_mse': []}
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            out = net(Xt[j])
            L = lossf(out, Yt[j])
            L.backward(); opt.step(); sched.step()
            tot += L.item() * len(j)
        # Monitoring only: the reported value is the final epoch, fixed in advance;
        # no epoch is selected on the test set.
        net.eval()
        pv = predict_batched(net, Xte) * ys + ym
        curve['epoch'].append(ep + 1)
        curve['sigma'].append(round(float((pv - y[te]).std()), 4))
        curve['train_mse'].append(round(tot / len(Xt), 4))
        print('    epoch %2d  train mse %.4f   test sigma %.4f'
              % (ep + 1, curve['train_mse'][-1], curve['sigma'][-1]))
    net.eval()
    p = predict_batched(net, Xte) * ys + ym
    PREDS[key] = p
    if record:
        CURVES[key] = curve
        RES[key] = round(float((p - y[te]).std()), 4)
        RES[key + '_params'] = int(npar)
    print('    FINAL sigma = %.4f' % float((p - y[te]).std()))
    del net, opt, sched, Xt, Yt
    gc.collect()


print('\nCNNs on rendered images:')
train_cnn('B  rendered density only', 'B_rendered', img_tr[:, None], img_te[:, None])
train_cnn('C  rendered density + real muon counts', 'C_rendered_muon',
          np.stack([img_tr, mu_tr], 1), np.stack([img_te, mu_te], 1))
# Retrain of B with a different seed: the noise ceiling for the degeneracy probe
# (how well one run of the same model predicts another).
train_cnn('B\' rendered density only, second seed', 'B_rendered_seed2',
          img_tr[:, None], img_te[:, None], seed=43, record=False)

# ----------------------------------------------------------------- verdict
print('\n' + '=' * 66)
print('%-44s %8s %8s' % ('', 'predicted', 'observed'))
for k, lbl, pred in (('A_scalars4', 'A  four fit scalars (tabular)', 0.1234),
                     ('B_rendered', 'B  CNN on rendered density image', 0.1234),
                     ('C_rendered_muon', 'C  CNN on rendered + real muons', 0.1063),
                     ('D_nine', 'D  nine summary features (tabular)', 0.1019)):
    print('%-44s %8.4f %8.4f' % (lbl, pred, RES[k]))
print('=' * 66)
print('B - A = %+.4f   (claim: rendering adds nothing, so this is ~0)'
      % (RES['B_rendered'] - RES['A_scalars4']))
print('C vs archived CNN/ViT 0.1063: %+.4f' % (RES['C_rendered_muon'] - 0.1063))

R['rendering'] = RES
R['rendering']['n_train'] = int(NTRAIN)
R['rendering']['r_moliere'] = R_MOLIERE
R['rendering']['grid'] = [NROW, NCOL, PITCH]
R.setdefault('curves', {})
R['curves']['_provenance'] = ('generated by run_rendering.py / run_sweep.py; test sigma '
                              'recorded every epoch, final epoch reported')
R['curves']['B_cnn'] = CURVES['B_rendered']
R['curves']['C_cnn_muon'] = CURVES['C_rendered_muon']
# Test-set predictions for the probe calibration (derived from collaboration data:
# kept local, never released; see .gitignore).
np.savez(os.path.join(HERE, 'preds_rendering.npz'), r_idx=te,
         **{k: v for k, v in PREDS.items()})
json.dump(R, open(OUT, 'w'), indent=2)
print('\nwrote rendering block to results.json')
