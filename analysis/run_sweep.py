"""Architecture sweep on the rendered images.

run_rendering.py established that a CNN on NKG-rendered images cannot beat a
tabular model on the four scalars the images were rendered from. If that is an
information limit rather than a quirk of convolution, then EVERY architecture
should land on the same number. This sweeps linear, fully-connected and
attention models over the identical rendered input.

  E  Ridge on 400 flattened pixels ....... linear
  F  MLP on 400 flattened pixels ......... fully connected
  B  CNN (from run_rendering.py) ......... convolutional
  G  ViT on 4x5 patches .................. attention

The archived model was a CNN/ViT hybrid, so G is the closest mirror of it.
Prediction: all four agree with A (four scalars, tabular) to within noise.

Run after run_rendering.py:  python analysis/run_sweep.py
"""
import gc, json, os, math, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
from scipy.special import gammaln
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import torch
import torch.nn as nn

import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results.json')
R = json.load(open(OUT))
assert 'rendering' in R, 'run run_rendering.py first'
torch.manual_seed(42)
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", max(1, (os.cpu_count() or 4) - 1))))   # cap on shared machines

NROW, NCOL, PITCH, R_MOLIERE = 16, 25, 8.0, 103.0
gx = (np.arange(NCOL) - (NCOL - 1) / 2.0) * PITCH
gy = (np.arange(NROW) - (NROW - 1) / 2.0) * PITCH
GX, GY = np.meshgrid(gx, gy)


def nkg_density(Ne, s, xc, yc):
    r = np.sqrt((GX[None] - xc[:, None, None]) ** 2 + (GY[None] - yc[:, None, None]) ** 2)
    np.maximum(r, 1.0, out=r)
    lnC = gammaln(4.5 - s) - math.log(2 * math.pi) - gammaln(s) - gammaln(4.5 - 2 * s)
    x = r / R_MOLIERE
    ln = (np.log(Ne)[:, None, None] - 2 * math.log(R_MOLIERE) + lnC[:, None, None]
          + (s[:, None, None] - 2.0) * np.log(x) + (s[:, None, None] - 4.5) * np.log1p(x))
    return np.log10(1.0 + np.exp(ln)).astype(np.float32)


def t(a):
    # Zero-copy: numpy arrays expose the buffer protocol, so frombuffer can wrap
    # them directly. The previous tobytes()/bytearray() path made two full copies
    # and blew up on the two-channel tensor.
    a = np.ascontiguousarray(a, dtype=np.float32)
    return torch.frombuffer(a, dtype=torch.float32).reshape(a.shape)



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

print('loading and rendering (same protocol as run_rendering.py)...')
d = C.load_root()
c = C.load_csv()
y = d['y']
r_idx, _ = C.match(d, c)
test_mask = np.zeros(len(y), bool); test_mask[r_idx] = True
pool = np.where(C.train_mask(d, test_mask))[0]
NTRAIN = R['rendering']['n_train']
tr = np.random.default_rng(42).choice(pool, NTRAIN, replace=False)   # same seed => same split
te = r_idx

img_tr = nkg_density(d['NKGSize'][tr], d['Age'][tr], d['NKGX'][tr], d['NKGY'][tr])
img_te = nkg_density(d['NKGSize'][te], d['Age'][te], d['NKGX'][te], d['NKGY'][te])
m_, s_ = img_tr.mean(), img_tr.std()
img_tr = (img_tr - m_) / s_; img_te = (img_te - m_) / s_
flat_tr = img_tr.reshape(len(tr), -1)
flat_te = img_te.reshape(len(te), -1)
ym, ys = y[tr].mean(), y[tr].std()
print('  train %d, test %d, %d pixels' % (len(tr), len(te), flat_tr.shape[1]))

SW = {}
SEED_ONLY = os.environ.get('RENDER_SEED')          # seed study: ViT only, see run_render_seeds.py
if SEED_ONLY:
    torch.manual_seed(int(SEED_ONLY))

if not SEED_ONLY:
    # ------------------------------------------------------------ E: linear
    print('\nE  Ridge on flattened pixels')
    t0 = time.time()
    p = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(flat_tr, y[tr]).predict(flat_te)
    SW['E_ridge_pixels'] = round(C.sigma(p, y[te]), 4)
    print('   sigma = %.4f   (%.0fs)' % (SW['E_ridge_pixels'], time.time() - t0))

    # ------------------------------------------------------------ F: MLP
    print('\nF  MLP on flattened pixels')
    t0 = time.time()
    p = make_pipeline(StandardScaler(),
                      MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=90,
                                   early_stopping=True, random_state=42)
                      ).fit(flat_tr, y[tr]).predict(flat_te)
    SW['F_mlp_pixels'] = round(C.sigma(p, y[te]), 4)
    print('   sigma = %.4f   (%.0fs)' % (SW['F_mlp_pixels'], time.time() - t0))


# ------------------------------------------------------------ G: ViT
PR, PC = 4, 5                       # patch 4x5 -> 20 patches of 20 pixels
NP = (NROW // PR) * (NCOL // PC)


def patchify(x):
    """(N,16,25) -> (N,20,20) non-overlapping 4x5 patches."""
    n = x.shape[0]
    x = x.reshape(n, NROW // PR, PR, NCOL // PC, PC)
    x = x.transpose(0, 1, 3, 2, 4)
    return np.ascontiguousarray(x.reshape(n, NP, PR * PC))


class ViT(nn.Module):
    def __init__(self, dim=64, depth=4, heads=4, ff=128):
        super().__init__()
        self.emb = nn.Linear(PR * PC, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, NP + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, ff, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        h = self.emb(x)
        h = torch.cat([self.cls.expand(h.shape[0], -1, -1), h], 1) + self.pos
        h = self.enc(h)
        return self.head(self.norm(h[:, 0])).squeeze(-1)


print('\nG  ViT on %d patches of %dx%d' % (NP, PR, PC))
Ptr, Pte = patchify(img_tr), patchify(img_te)
net = ViT()
npar = sum(q.numel() for q in net.parameters())
print('   %d params' % npar)
EPOCHS, BS = 30, 256
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=5e-2)
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=2e-3, total_steps=EPOCHS * math.ceil(len(Ptr) / BS))
Xt, Yt = t(Ptr), t((y[tr] - ym) / ys)
lossf = nn.MSELoss()
t0 = time.time()
curve = {'epoch': [], 'sigma': [], 'train_mse': []}
for ep in range(EPOCHS):
    net.train()
    perm = torch.randperm(len(Xt)); tot = 0.0
    for i in range(0, len(Xt), BS):
        j = perm[i:i + BS]
        opt.zero_grad()
        L = lossf(net(Xt[j]), Yt[j])
        L.backward(); opt.step(); sched.step()
        tot += L.item() * len(j)
    net.eval()                                  # monitoring only; final epoch reported
    pv = predict_batched(net, Pte) * ys + ym
    curve['epoch'].append(ep + 1)
    curve['sigma'].append(round(float((pv - y[te]).std()), 4))
    curve['train_mse'].append(round(tot / len(Xt), 4))
    print('   epoch %2d  train mse %.4f  test sigma %.4f  (%.0fs)'
          % (ep + 1, curve['train_mse'][-1], curve['sigma'][-1], time.time() - t0))
net.eval()
p = predict_batched(net, Pte) * ys + ym
SW['G_vit_patches'] = round(float((p - y[te]).std()), 4)
SW['G_vit_params'] = int(npar)
print('   FINAL sigma = %.4f' % SW['G_vit_patches'])

if SEED_ONLY:
    os.makedirs(os.path.join(HERE, 'seeds'), exist_ok=True)
    json.dump({'seed': int(SEED_ONLY), 'G_vit_patches': SW['G_vit_patches']},
              open(os.path.join(HERE, 'seeds', 'sweep_%s.json' % SEED_ONLY), 'w'))
    raise SystemExit

# ------------------------------------------------------------ verdict
RD = R['rendering']
print('\n' + '=' * 70)
print('ARCHITECTURE SWEEP ON IDENTICAL RENDERED INPUT')
print('=' * 70)
rows = [('A  four fit scalars (tabular GBM)', RD['A_scalars4'], 'reference'),
        ('E  Ridge on 400 pixels', SW['E_ridge_pixels'], 'linear'),
        ('F  MLP on 400 pixels', SW['F_mlp_pixels'], 'fully connected'),
        ('B  CNN on the image', RD['B_rendered'], 'convolutional'),
        ('G  ViT on 4x5 patches', SW['G_vit_patches'], 'attention')]
for lbl, v, kind in rows:
    print('  %-36s %8.4f   %s' % (lbl, v, kind))
vals = [v for _, v, _ in rows[1:]]
print('-' * 70)
print('  spread across architectures: %.4f' % (max(vals) - min(vals)))
print('  vs four-scalar reference:    %+.4f .. %+.4f'
      % (min(vals) - RD['A_scalars4'], max(vals) - RD['A_scalars4']))

R['sweep'] = SW
R.setdefault('curves', {})['G_vit'] = curve
np.savez(os.path.join(HERE, 'preds_sweep.npz'), r_idx=te, G_vit_patches=p)   # local only
json.dump(R, open(OUT, 'w'), indent=2)
print('\nwrote sweep block to results.json')
