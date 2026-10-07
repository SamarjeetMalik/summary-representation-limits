"""The known-truth benchmark with a binary target.

Same measurement X, summary T = A X and extra channel M as run_synthetic.py (same seed, so the
same samples). The target is the label

    z = 1[ h(u) + tau * xi > c ],    u = w'X,   h(u) = u + 0.5 sin(2u),   xi ~ N(0,1),

with c the median of h(u), so P(z = 1 | u) = Phi((h(u) - c) / tau).

Two risks are known for every representation Z, because u | Z ~ N(mu(Z), v_Z):

  * the Brier risk, which is squared loss on the label. Its Bayes value is E[q (1 - q)] with
    q = P(z = 1 | Z), and Propositions 1 and 2 apply to it as stated, with y replaced by z;
  * the 0-1 risk. Its Bayes value is E[min(q, 1 - q)]. It also cannot decrease under a
    deterministic map of Z, by the same conditioning argument applied to the Bayes classifier.

Reported: the true floors; the Gamma-test estimate of the Brier floor (the estimator is the
same, applied to a 0/1 target); gradient-boosted and MLP classifiers; and a CNN classifier on the
image rendered from T.

Extends results.json.  Run:  python analysis/run_synthetic_cls.py
"""
import json, math, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier as GBC
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results.json')
torch.manual_seed(0)
torch.set_num_threads(int(os.environ.get('TORCH_THREADS', max(1, (os.cpu_count() or 4) - 1))))

N, NTEST, P, KT, TAU, K = 170000, 34000, 32, 6, 0.30, 10
NROW, NCOL = 16, 25
rng = np.random.default_rng(0)

# ------------------------------------------------------------------ construction (as run_synthetic.py)
Q, _ = np.linalg.qr(rng.normal(size=(P, P)))
lam = 1.0 / np.arange(1, P + 1) ** 0.5
Sigma = (Q * lam) @ Q.T
A = rng.normal(size=(KT, P)) / np.sqrt(P)
w = rng.normal(size=P); w /= np.sqrt(w @ Sigma @ w)
m = w + 0.8 * rng.normal(size=P) / np.sqrt(P)
SM = 0.30


def h(u):
    return u + 0.5 * np.sin(2 * u)


X = rng.multivariate_normal(np.zeros(P), Sigma, size=N)
T = X @ A.T
u = X @ w
M = X @ m + SM * rng.normal(size=N)
gx, gw = np.polynomial.hermite_e.hermegauss(160); gw = gw / gw.sum()
CUT = 0.0                                        # h is odd and u is symmetric, so the median of h(u) is 0
z = (h(u) + TAU * np.random.default_rng(7).normal(size=N) > CUT).astype(int)
te = np.arange(N - NTEST, N); tr = np.arange(N - NTEST)


# ------------------------------------------------------------------ truth
def true_risks(C, noise_var=None):
    """(Bayes Brier risk, Bayes 0-1 risk) for Z = C X (+ independent noise); C = None means Z = X."""
    pu = lambda uu: norm.cdf((h(uu) - CUT) / TAU)
    if C is None:
        q = pu(gx)                               # u is known: q = P(z = 1 | u), u ~ N(0, 1)
        return float((q * (1 - q)) @ gw), float(np.minimum(q, 1 - q) @ gw)
    VZ = C @ Sigma @ C.T
    if noise_var is not None:
        VZ = VZ + np.diag(noise_var)
    c = C @ Sigma @ w
    v = max(1.0 - c @ np.linalg.solve(VZ, c), 0.0)
    mu = np.sqrt(max(1.0 - v, 0.0)) * gx
    q = pu(mu[:, None] + np.sqrt(v) * gx[None, :]) @ gw
    return float((q * (1 - q)) @ gw), float(np.minimum(q, 1 - q) @ gw)


TRUE = {'T': true_risks(A), 'T+M': true_risks(np.vstack([A, m]), np.r_[np.zeros(KT), SM ** 2]),
        'X': true_risks(None)}
FEATS = {'T': T, 'T+M': np.column_stack([T, M]), 'X': X}

# Monte Carlo check of the quadrature on the summary, independent of every estimator below
VT = A @ Sigma @ A.T; cT = A @ Sigma @ w
mu_mc = T @ np.linalg.solve(VT, cT); vT = 1.0 - cT @ np.linalg.solve(VT, cT)
q_mc = norm.cdf((h(mu_mc[:, None] + np.sqrt(vT) * rng.normal(size=(N, 200))) - CUT) / TAU).mean(1)
mc = float((q_mc * (1 - q_mc)).mean())
assert abs(mc - TRUE['T'][0]) / TRUE['T'][0] < 0.02, 'quadrature and Monte Carlo disagree'
assert abs(z.mean() - 0.5) < 0.01


def gamma_brier(F):
    sc = StandardScaler().fit(F)
    dist, idx = NearestNeighbors(n_neighbors=K + 1).fit(sc.transform(F)).kneighbors(sc.transform(F[te]))
    d2 = dist[:, 1:] ** 2; g2 = 0.5 * (z[te][:, None] - z[idx[:, 1:]]) ** 2
    return float(max(np.polyfit(d2.mean(0), g2.mean(0), 1)[1], 0))


def score(p):
    return float(((p - z[te]) ** 2).mean()), float(((p > 0.5).astype(int) != z[te]).mean())


rows = {}
for name, F in FEATS.items():
    gb = GBC(max_iter=600, learning_rate=.06, random_state=42).fit(F[tr], z[tr]).predict_proba(F[te])[:, 1]
    ml = make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(128, 64) if name == 'X' else (64, 64), max_iter=120, random_state=42,
        early_stopping=True)).fit(F[tr], z[tr]).predict_proba(F[te])[:, 1]
    g = gamma_brier(F)
    rows[name] = {'brier_true': round(TRUE[name][0], 4), 'err_true': round(TRUE[name][1], 4),
                  'brier_gamma': round(g, 4),
                  'gamma_err_pct': round(100 * (g - TRUE[name][0]) / TRUE[name][0], 1),
                  'gb': [round(v, 4) for v in score(gb)], 'mlp': [round(v, 4) for v in score(ml)]}
    print('%-4s Brier true %.4f gamma %.4f [%+.1f%%]  0-1 true %.4f | GB %s  MLP %s'
          % (name, TRUE[name][0], g, rows[name]['gamma_err_pct'], TRUE[name][1], rows[name]['gb'], rows[name]['mlp']))


# ------------------------------------------------------------------ a CNN classifier on a rendering of T
def render(Tm):
    s = Tm / T[tr].std(0)
    ii = np.arange(NROW)[None, :, None]; jj = np.arange(NCOL)[None, None, :]
    ci_ = 7.5 + 2.0 * np.tanh(s[:, 2])[:, None, None]; cj = 12.0 + 3.0 * np.tanh(s[:, 3])[:, None, None]
    wd = (3.0 + 1.0 * np.tanh(s[:, 1]))[:, None, None]
    img = s[:, 0][:, None, None] * np.exp(-((ii - ci_) ** 2 + (jj - cj) ** 2) / (2 * wd ** 2))
    img = img + 0.05 * s[:, 4][:, None, None] * (ii - 7.5) + 0.05 * s[:, 5][:, None, None] * (jj - 12.0)
    return img.astype(np.float32)


def t(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    return torch.frombuffer(a, dtype=torch.float32).reshape(a.shape)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


sub = np.random.default_rng(3).choice(tr, 60000, replace=False)
img_tr, img_te = render(T[sub]), render(T[te])
mi, si = img_tr.mean(), img_tr.std()
Xt, Xe = t(((img_tr - mi) / si)[:, None]), ((img_te - mi) / si)[:, None]
Yt = t(z[sub].astype(np.float32))
net = Net(); EP, BS = 20, 256
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=EP * math.ceil(len(Xt) / BS))
for ep in range(EP):
    net.train(); perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), BS):
        j = perm[i:i + BS]; opt.zero_grad()
        L = nn.functional.binary_cross_entropy_with_logits(net(Xt[j]), Yt[j]); L.backward(); opt.step(); sched.step()
net.eval(); out = []
with torch.no_grad():
    for i in range(0, len(Xe), 1024):
        out.extend(torch.sigmoid(net(t(Xe[i:i + 1024]))).tolist())
cnn = score(np.asarray(out))
gb_same = score(GBC(max_iter=600, learning_rate=.06, random_state=42).fit(T[sub], z[sub]).predict_proba(T[te])[:, 1])
rend = {'cnn_rendered': [round(v, 4) for v in cnn], 'gb_T_same_train': [round(v, 4) for v in gb_same]}
print('rendering:', rend)

R = json.load(open(OUT))
R['synthetic_cls'] = {'n': N, 'n_test': NTEST, 'quadrature_vs_mc_pct': round(100 * abs(mc - TRUE['T'][0]) / TRUE['T'][0], 2),
                      'rows': rows, 'rendering': rend}
json.dump(R, open(OUT, 'w'), indent=2)
print('extended %s' % OUT)
