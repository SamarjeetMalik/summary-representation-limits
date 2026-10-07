"""A known-truth check of the audit, outside air showers.

Every Bayes risk in this benchmark is known (to quadrature precision), so each step of the audit
can be compared with the truth instead of with another estimate.

Construction. A "measurement" X in R^32 is Gaussian with a decaying spectrum. The exported
"summary" is a fixed lossy linear map T = A X in R^6 (the stand-in for a fit). The target is

    y = h(u) + tau * (0.5 + V) * xi,     u = w'X,   h(u) = u + 0.5 sin(2u),   xi ~ N(0,1),

so y is a nonlinear function of one direction of X plus heteroscedastic noise. Two further
channels are observed: M = m'X + noise, which carries information about u that T lacks, and
V ~ U(0,1), which sets the noise scale. V changes the conditional distribution of y and leaves
its conditional mean alone: I(y; V | T) > 0 while the squared-loss Bayes risk is unchanged,
the situation described after Proposition 2.

Truth. For jointly Gaussian (u, Z), u | Z ~ N(mu(Z), v_Z) with v_Z constant, so

    R*(Z) = E[noise^2] + E_mu[ Var_{u ~ N(mu, v_Z)} h(u) ],    mu ~ N(0, Var(u) - v_Z),

a two-level Gauss-Hermite integral. R*(X) = E[noise^2].

Steps run: (1) Gamma-test floor against the true floor; (2) ladder with the null channel V;
(3) a network on an image rendered from T, and a model on the measurement X; (4) the probe.
Step 5 (an inherited selection) has no counterpart here.

Extends results.json.  Run:  python analysis/run_synthetic.py
"""
import json, math, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as GBR
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
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

# ------------------------------------------------------------------ construction
Q, _ = np.linalg.qr(rng.normal(size=(P, P)))
lam = 1.0 / np.arange(1, P + 1) ** 0.5
Sigma = (Q * lam) @ Q.T
A = rng.normal(size=(KT, P)) / np.sqrt(P)
w = rng.normal(size=P); w /= np.sqrt(w @ Sigma @ w)                 # Var(u) = 1
m = w + 0.8 * rng.normal(size=P) / np.sqrt(P)
SM = 0.30                                                           # noise on the M channel


def h(u):
    return u + 0.5 * np.sin(2 * u)


X = rng.multivariate_normal(np.zeros(P), Sigma, size=N)
T = X @ A.T
u = X @ w
M = X @ m + SM * rng.normal(size=N)
V = rng.uniform(size=N)
y = h(u) + TAU * (0.5 + V) * rng.normal(size=N)
te = np.arange(N - NTEST, N); tr = np.arange(N - NTEST)
ENOISE = TAU ** 2 * (0.25 + 0.5 + 1.0 / 3.0)                        # E[(0.5 + V)^2] = 13/12

# ------------------------------------------------------------------ truth
gx, gw = np.polynomial.hermite_e.hermegauss(120); gw = gw / gw.sum()


def true_risk(C, noise_var=None):
    """R*(Z) for Z = C X (+ independent noise with the given variances)."""
    VZ = C @ Sigma @ C.T
    if noise_var is not None:
        VZ = VZ + np.diag(noise_var)
    c = C @ Sigma @ w
    v = max(1.0 - c @ np.linalg.solve(VZ, c), 0.0)
    mu = np.sqrt(max(1.0 - v, 0.0)) * gx
    uu = mu[:, None] + np.sqrt(v) * gx[None, :]
    hv = h(uu)
    var = (hv ** 2) @ gw - (hv @ gw) ** 2
    return ENOISE + float(var @ gw)


TRUE = {'T%d' % k: true_risk(A[:k]) for k in range(1, KT + 1)}
TRUE['T+M'] = true_risk(np.vstack([A, m]), np.r_[np.zeros(KT), SM ** 2])
TRUE['T+M+V'] = TRUE['T+M']                                         # V leaves E[y | .] unchanged
TRUE['X'] = ENOISE
FEATS = {**{'T%d' % k: T[:, :k] for k in range(1, KT + 1)},
         'T+M': np.column_stack([T, M]), 'T+M+V': np.column_stack([T, M, V]), 'X': X}

# Monte Carlo check of the quadrature on the full summary (independent of every estimator below)
VT = A @ Sigma @ A.T; cT = A @ Sigma @ w
mu_mc = T @ np.linalg.solve(VT, cT); vT = 1.0 - cT @ np.linalg.solve(VT, cT)
draw = mu_mc[:, None] + np.sqrt(vT) * rng.normal(size=(N, 64))
mc = ENOISE + float(h(draw).var(axis=1, ddof=1).mean())
assert abs(mc - TRUE['T6']) / TRUE['T6'] < 0.01, 'quadrature and Monte Carlo disagree'


# ------------------------------------------------------------------ step 1: floor estimate
def gamma_floor(F):
    sc = StandardScaler().fit(F)
    Z, Zq = sc.transform(F), sc.transform(F[te])
    dist, idx = NearestNeighbors(n_neighbors=K + 1).fit(Z).kneighbors(Zq)
    d2 = dist[:, 1:] ** 2; g2 = 0.5 * (y[te][:, None] - y[idx[:, 1:]]) ** 2
    est = lambda rows: math.sqrt(max(np.polyfit(d2[rows].mean(0), g2[rows].mean(0), 1)[1], 0))
    bs = [est(rng.integers(0, NTEST, NTEST)) for _ in range(500)]
    return est(slice(None)), [float(v) for v in np.percentile(bs, [2.5, 97.5])]


def sig(p):
    return float(np.sqrt(((p - y[te]) ** 2).mean()))                # RMSE: the quantity R* bounds


gb = lambda: GBR(max_iter=600, learning_rate=.06, random_state=42)
mlp = lambda hid=(64, 64): make_pipeline(StandardScaler(), MLPRegressor(
    hidden_layer_sizes=hid, max_iter=120, random_state=42, early_stopping=True))
rows, PRED = {}, {}
for name, F in FEATS.items():
    fl, ci = gamma_floor(F)
    p_gb = gb().fit(F[tr], y[tr]).predict(F[te]); PRED[name] = p_gb
    r = {'true_floor': round(math.sqrt(TRUE[name]), 4), 'gamma': round(fl, 4),
         'gamma_ci': [round(v, 4) for v in ci], 'gb': round(sig(p_gb), 4),
         'gamma_err_pct': round(100 * (fl - math.sqrt(TRUE[name])) / math.sqrt(TRUE[name]), 1)}
    if name in ('T6', 'T+M', 'X'):
        r['mlp'] = round(sig(mlp((128, 64) if name == 'X' else (64, 64)).fit(F[tr], y[tr]).predict(F[te])), 4)
        r['ridge'] = round(sig(make_pipeline(StandardScaler(), Ridge(1.0)).fit(F[tr], y[tr]).predict(F[te])), 4)
    rows[name] = r
    print('%-6s true %.4f  gamma %.4f [%+.1f%%]  GB %.4f  %s'
          % (name, r['true_floor'], r['gamma'], r['gamma_err_pct'], r['gb'],
             'MLP %.4f ridge %.4f' % (r['mlp'], r['ridge']) if 'mlp' in r else ''))

# ------------------------------------------------------------------ step 2: paired change for the null channel
BOOT = rng.integers(0, NTEST, size=(1000, NTEST))
e_a, e_b = PRED['T+M'] - y[te], PRED['T+M+V'] - y[te]
dv = np.sqrt((e_b[BOOT] ** 2).mean(1)) - np.sqrt((e_a[BOOT] ** 2).mean(1))
null = {'delta': round(sig(PRED['T+M+V']) - sig(PRED['T+M']), 5),
        'ci': [round(float(v), 5) for v in np.percentile(dv, [2.5, 97.5])],
        # the channel is not uninformative: it predicts the size of the error
        'corr_V_abs_err': round(float(np.corrcoef(V[te], np.abs(e_a))[0, 1]), 3)}
print('null channel V: delta %+.5f  CI %s   corr(V, |error|) = %.3f' % (null['delta'], null['ci'], null['corr_V_abs_err']))


# ------------------------------------------------------------------ step 3: a rendering of T
def render(Tm):
    """A smooth image computed from the six summary numbers: no measurement enters it."""
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


sub = rng.choice(tr, 60000, replace=False)
img_tr, img_te = render(T[sub]), render(T[te])
mi, si = img_tr.mean(), img_tr.std()
Xt, Xe = t(((img_tr - mi) / si)[:, None]), ((img_te - mi) / si)[:, None]
ym, ys = y[sub].mean(), y[sub].std(); Yt = t((y[sub] - ym) / ys)
net = Net(); EP, BS = 20, 256
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=EP * math.ceil(len(Xt) / BS))
for ep in range(EP):
    net.train(); perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), BS):
        j = perm[i:i + BS]; opt.zero_grad()
        L = nn.functional.mse_loss(net(Xt[j]), Yt[j]); L.backward(); opt.step(); sched.step()
net.eval(); out = []
with torch.no_grad():
    for i in range(0, len(Xe), 1024):
        out.extend(net(t(Xe[i:i + 1024])).tolist())
p_cnn = np.asarray(out) * ys + ym
rend = {'true_floor_T': rows['T6']['true_floor'], 'true_floor_X': rows['X']['true_floor'],
        'gb_T_same_train': round(sig(gb().fit(T[sub], y[sub]).predict(T[te])), 4),
        'cnn_rendered': round(sig(p_cnn), 4),
        'ridge_pixels': round(sig(make_pipeline(StandardScaler(), Ridge(10.0)).fit(
            img_tr.reshape(len(sub), -1), y[sub]).predict(img_te.reshape(NTEST, -1))), 4),
        'mlp_X': rows['X']['mlp'], 'gb_X': rows['X']['gb']}
print('rendering:', rend)


# ------------------------------------------------------------------ step 4: probe
def probe(mout, F):
    pr = np.zeros(NTEST)
    for a, b in KFold(5, shuffle=True, random_state=0).split(F):
        pr[b] = GBR(max_iter=400, random_state=0).fit(F[a], mout[a]).predict(F[b])
    return round(float(1 - ((mout - pr) ** 2).mean() / mout.var()), 4)


Tt, TMt = T[te], np.column_stack([T[te], M[te]])
outs = {'GB on T': PRED['T6'], 'CNN on rendered T': p_cnn, 'GB on T and M': PRED['T+M'],
        'GB on T + noise': PRED['T6'] + rng.normal(0, 0.3 * PRED['T6'].std(), NTEST),
        'MLP on X': mlp((128, 64)).fit(X[tr], y[tr]).predict(X[te])}
pb = {k: {'T': probe(v, Tt), 'T_M': probe(v, TMt)} for k, v in outs.items()}
for k, v in pb.items():
    print('probe %-20s R2(T) %.4f  R2(T,M) %.4f' % (k, v['T'], v['T_M']))

R = json.load(open(OUT))
R['synthetic'] = {'n': N, 'n_test': NTEST, 'p': P, 'k_summary': KT, 'tau': TAU,
                  'quadrature_vs_mc_pct': round(100 * abs(mc - TRUE['T6']) / TRUE['T6'], 2),
                  'rows': rows, 'null_channel': null, 'rendering': rend, 'probe': pb}
json.dump(R, open(OUT, 'w'), indent=2)
print('extended %s' % OUT)
