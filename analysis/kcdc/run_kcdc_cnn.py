"""A learned encoder of the MEASURED detector image on KASCADE.

The GRAPES-3 part of the paper shows that a CNN on an image rendered from the fit cannot beat the
fit. KASCADE's 252 stations sit on a 16 x 16 grid at 13 m spacing (KCDC Simulation Manual,
Appendix A), so its per-station signals form a measured image. This script asks the counterpart
question: can a CNN on the measured image do better than the nine-feature summary, and better than
the engineered radial profiles of run_kcdc_floor.py?

  input   4 x 16 x 16: log10(1 + e/gamma deposit), log10(1 + muon deposit), arrival time (scaled,
          0 where silent), hit mask; stations off the regular grid are snapped to the nearest node
  models  (i)  CNN on the image alone
          (ii) residual CNN: gradient boosting on the nine features gives a base prediction (out of
               fold, by shower, on the training set; refit on all training events for the test
               set), and a CNN on the image + nine features learns a correction through a
               zero-initialised output layer. It starts exactly at the summary's resolution and can
               only improve on it if the image carries usable information (the valid-by-
               construction design of the authors' earlier draft).
  Optimisation history, kept for the record. Two earlier configurations (BatchNorm, one-cycle
  schedule peaking at 3e-3, 25 and 40 epochs) gave unstable evaluation-mode predictions: single
  seeds at 0.155-0.169, far worse than their own 3-seed average, and a residual model that ended
  above its own starting point. A diagnostic on random train/test subsets showed the network
  itself is fine when trained gently (constant 1e-3: 0.142 on test after 6 epochs). The final
  configuration therefore uses GroupNorm (no train/eval discrepancy) and AdamW at 1e-3 with cosine
  decay; the reported value is the last epoch, fixed in advance.
  protocol identical event selection and shower-disjoint split as run_kcdc_floor.py; final epoch
          reported, fixed in advance (no selection on the test set); 2 seeds; paired bootstrap
          differences to gradient boosting on the nine features and on the best engineered set.

Output: analysis/kcdc/kcdc_cnn_results.json.  Run (GPU):  python analysis/kcdc/run_kcdc_cnn.py
"""
import csv, json, math, os, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor as GBR
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'data', 'kcdc')
# ENCODER selects the network: 'small' is the CNN of the main experiment; 'resnet' and 'settx' are
# the larger and the differently structured encoders of the capacity check, each written to its
# own results file so the main experiment is never overwritten.
ENCODER = os.environ.get('ENCODER', 'small')
OUT = os.path.join(HERE, 'kcdc_cnn_results.json' if ENCODER == 'small' else 'kcdc_cnn_%s_results.json' % ENCODER)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS, BS, NB = int(os.environ.get('EPOCHS', '20')), 512, 1000
SEEDS = (42, 43, 44) if ENCODER == 'small' else (42, 43)
torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '8')))

F = dict(np.load(os.path.join(DATA, 'kcdc_station_features.npz')))
Dn = np.load(os.path.join(DATA, 'kcdc_station_dense.npz'))
assert np.array_equal(Dn['TrEP'], F['TrEP']), 'dense and feature files are not row-aligned'

# ---- selection and shower-disjoint split: identical to run_kcdc_floor.py
ze = F['Ze'] if np.nanmax(F['Ze']) > 3.2 else np.degrees(F['Ze'])
y_all = F['TrEP'] - 9.0
ok = np.ones(len(y_all), bool)
for k in ('Ne', 'Nmu', 'Age', 'Xc', 'Yc', 'TrNe', 'TrNm'):
    ok &= np.isfinite(F[k])
ok &= (F['Ne'] > 1) & (F['Nmu'] > 0) & (F['TrNe'] > 1) & (F['TrNm'] > 0) & (y_all >= 5) & (y_all <= 6)
ids = np.where(ok)[0]; y = y_all[ids].astype(np.float32)
_, shower = np.unique(np.column_stack([F['TrEP'][ids], F['TrPP'][ids].astype(np.float32)]), axis=0, return_inverse=True)
shower = shower.ravel()
ush = np.random.default_rng(0).permutation(np.unique(shower))
is_tr = np.isin(shower, ush[:int(.75 * len(ush))])
tr, te = np.where(is_tr)[0], np.where(~is_tr)[0]; yt = y[te]

col = lambda k: F[k][ids].astype(float)
Ne, Nmu, age, xc, yc = col('Ne'), col('Nmu'), col('Age'), col('Xc'), col('Yc')
zz = ze[ids].astype(float)
NINE = np.column_stack([Ne, Nmu, Nmu - Ne, age, zz, np.cos(np.radians(zz)), xc, yc, np.hypot(xc, yc)])
NINE_Z = StandardScaler().fit(NINE[tr]).transform(NINE).astype(np.float32)

# ---- station -> grid node
RC = np.zeros((252, 2), int)
for r in csv.DictReader(open(os.path.join(HERE, 'kascade_station_coordinates.csv'))):
    i = int(r['station_id']) - 1
    RC[i] = [int(np.clip(round((float(r['y_m']) + 97.5) / 13), 0, 15)), int(np.clip(round((float(r['x_m']) + 97.5) / 13), 0, 15))]
assert len({tuple(v) for v in RC}) == 252, 'two stations map to the same grid node'

e = np.log10(1 + np.maximum(Dn['dense_e'][ids].astype(np.float32), 0))
mu = np.log10(1 + np.maximum(Dn['dense_mu'][ids].astype(np.float32), 0))
tt = Dn['dense_t'][ids].astype(np.float32); hit = (tt > 0).astype(np.float32)
tmed = np.array([np.median(r[r > 0]) if (r > 0).any() else 0 for r in tt], np.float32)
tt = np.where(hit > 0, (tt - tmed[:, None]) / 50.0, 0).astype(np.float32)
IMG = np.zeros((len(ids), 4, 16, 16), np.float32)
for c_, arr in enumerate((e, mu, tt, hit)):
    IMG[:, c_, RC[:, 0], RC[:, 1]] = arr
print('KASCADE measured image: %d events, train %d / test %d, device %s' % (len(ids), len(tr), len(te), DEV), flush=True)


class Net(nn.Module):
    def __init__(self, use_nine):
        super().__init__()
        self.use_nine = use_nine
        self.f = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1), nn.GroupNorm(8, 96), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(96, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.h = nn.Sequential(nn.Linear(128 + (9 if use_nine else 0), 128), nn.ReLU(), nn.Linear(128, 1))
        if use_nine:                                   # residual mode: start exactly at the base
            nn.init.zeros_(self.h[-1].weight); nn.init.zeros_(self.h[-1].bias)

    def forward(self, x, z):
        h = self.f(x)
        return self.h(torch.cat([h, z], 1) if self.use_nine else h).squeeze(-1)


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.a = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.ReLU(),
                               nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout))
        self.s = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        return torch.relu(self.a(x) + self.s(x))


class ResNet(nn.Module):
    """A residual network about thirty times the size of Net."""
    def __init__(self, use_nine):
        super().__init__()
        self.use_nine = use_nine
        self.f = nn.Sequential(Block(4, 96), Block(96, 96), nn.MaxPool2d(2), Block(96, 192), Block(192, 192),
                               nn.MaxPool2d(2), Block(192, 384), Block(384, 384), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.h = nn.Sequential(nn.Linear(384 + (9 if use_nine else 0), 256), nn.ReLU(), nn.Linear(256, 1))
        if use_nine:
            nn.init.zeros_(self.h[-1].weight); nn.init.zeros_(self.h[-1].bias)

    def forward(self, x, z):
        h = self.f(x)
        return self.h(torch.cat([h, z], 1) if self.use_nine else h).squeeze(-1)


class SetTx(nn.Module):
    """A transformer over stations: one token per grid node, no convolution."""
    def __init__(self, use_nine, dim=128, depth=6, heads=8):
        super().__init__()
        self.use_nine = use_nine
        self.emb = nn.Linear(4, dim)
        self.pos = nn.Parameter(torch.zeros(1, 257, dim)); nn.init.trunc_normal_(self.pos, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth); self.norm = nn.LayerNorm(dim)
        self.h = nn.Sequential(nn.Linear(dim + (9 if use_nine else 0), 128), nn.ReLU(), nn.Linear(128, 1))
        if use_nine:
            nn.init.zeros_(self.h[-1].weight); nn.init.zeros_(self.h[-1].bias)

    def forward(self, x, z):
        t_ = self.emb(x.flatten(2).transpose(1, 2))
        t_ = torch.cat([self.cls.expand(len(t_), -1, -1), t_], 1) + self.pos
        h = self.norm(self.enc(t_)[:, 0])
        return self.h(torch.cat([h, z], 1) if self.use_nine else h).squeeze(-1)


NETS = {'small': Net, 'resnet': ResNet, 'settx': SetTx}
# The residual network was first trained with the CNN's recipe (AdamW 1e-3, cosine, no warm-up).
# It did not generalise: image-only 0.153-0.156 at the final epoch, worse than the small CNN, and
# the zero-initialised correction never left zero. A second attempt at 3e-4 with warm-up and
# gradient clipping was worse still at the final epoch (0.174). Both logs are kept
# (kcdc_cnn_resnet_lr1e-3.log, enc_resnet_v2.log). The larger encoders are therefore trained with
# early stopping on a validation set: a tenth of the TRAINING showers is held out, the weights
# with the best validation error are kept, and the test set plays no part in the choice.
LR = float(os.environ.get('LR', {'small': 1e-3, 'resnet': 3e-4, 'settx': 5e-4}[ENCODER]))
WARMUP = ENCODER == 'resnet'
EARLY = ENCODER != 'small'


def train(use_nine, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = NETS[ENCODER](use_nine).to(DEV)
    if seed == SEEDS[0] and not use_nine:
        print('  encoder %s: %d parameters' % (ENCODER, sum(p.numel() for p in net.parameters())), flush=True)
    target = (y - BASE) if use_nine else y                # residual mode learns a correction
    ym, ys = (0.0, float(target[tr].std())) if use_nine else (float(y[tr].mean()), float(y[tr].std()))
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    spe = math.ceil(len(tr) / BS)                           # a little long under early stopping: harmless
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * spe)
    if WARMUP:
        sched = torch.optim.lr_scheduler.SequentialLR(opt, [
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.05, total_iters=spe),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=(EPOCHS - 1) * spe)], milestones=[spe])
    Xi = torch.from_numpy(IMG); Z = torch.from_numpy(NINE_Z); Y = torch.from_numpy(((target - ym) / ys).astype(np.float32))
    lossf = nn.MSELoss(); t0 = time.time()
    fit = tr
    if EARLY:                                              # validation showers, drawn from training only
        vs = np.random.default_rng(100 + seed).random(shower.max() + 1) < 0.10
        fit, val = tr[~vs[shower[tr]]], tr[vs[shower[tr]]]
        best, best_state, best_ep = np.inf, None, -1

    def predict(rows):
        net.eval(); o = []
        with torch.no_grad():
            for i in range(0, len(rows), 4096):
                j = torch.from_numpy(rows[i:i + 4096])
                o.append(net(Xi[j].to(DEV), Z[j].to(DEV)).cpu().numpy())
        return np.concatenate(o)

    for ep in range(EPOCHS):
        net.train(); perm = torch.from_numpy(np.random.permutation(fit))
        for i in range(0, len(perm), BS):
            j = perm[i:i + BS]
            opt.zero_grad(); L = lossf(net(Xi[j].to(DEV), Z[j].to(DEV)), Y[j].to(DEV)); L.backward()
            if WARMUP:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
        if EARLY:
            v = float(((predict(val) - Y[val].numpy()) ** 2).mean())
            if v < best:
                best, best_ep = v, ep + 1
                best_state = {k: t_.detach().clone() for k, t_ in net.state_dict().items()}
    if EARLY:
        net.load_state_dict(best_state)
        print('    early stopping: best validation epoch %d of %d' % (best_ep, EPOCHS), flush=True)
    p = predict(te) * ys + ym + (BASE[te] if use_nine else 0)
    print('  CNN %s seed %d: sigma %.4f (%.0fs)' % ('residual on GB(nine)' if use_nine else 'image only', seed,
                                                   float((p - yt).std()), time.time() - t0), flush=True)
    return p - yt


# ---- references on the same split, and the out-of-fold base for the residual model
GB = lambda: GBR(max_iter=600, learning_rate=.06, random_state=42)
gb_full = GB().fit(NINE[tr], y[tr])
e_nine = gb_full.predict(NINE[te]) - yt
BASE = np.zeros(len(y), np.float32); BASE[te] = gb_full.predict(NINE[te])
fold = np.random.default_rng(3).integers(0, 5, size=shower.max() + 1)[shower]
for k in range(5):
    a_ = tr[fold[tr] != k]; b_ = tr[fold[tr] == k]
    BASE[b_] = GB().fit(NINE[a_], y[a_]).predict(NINE[b_])
print('  out-of-fold base on training events: sigma %.4f' % float((BASE[tr] - y[tr]).std()), flush=True)
ENG = np.column_stack([NINE, F['e_ann'][ids], F['mu_ann'][ids], F['e_nhit'][ids], F['mu_nhit'][ids],
                       F['e_logsum'][ids], F['mu_logsum'][ids], F['t_spread'][ids]])
e_eng = GB().fit(ENG[tr], y[tr]).predict(ENG[te]) - yt
print('  GB nine %.4f, GB engineered %.4f' % (e_nine.std(), e_eng.std()), flush=True)

BOOT = np.random.default_rng(1).integers(0, len(te), size=(NB, len(te)), dtype=np.int32)


def paired(a, b):
    dl = a[BOOT].std(1) - b[BOOT].std(1)
    return round(float(a.std() - b.std()), 5), [round(float(v), 5) for v in np.percentile(dl, [2.5, 97.5])]


R = {'encoder': ENCODER, 'n_params': int(sum(p.numel() for p in NETS[ENCODER](False).parameters())),
     'n_events': int(len(ids)), 'n_train': int(len(tr)), 'n_test': int(len(te)), 'epochs': EPOCHS,
     'gb_nine': round(float(e_nine.std()), 4), 'gb_engineered': round(float(e_eng.std()), 4), 'cnn': {}}
for use_nine in (False, True):
    key = 'residual on GB(nine)' if use_nine else 'image only'
    es = [train(use_nine, s) for s in SEEDS]
    e_ens = np.mean(es, axis=0)
    R['cnn'][key] = {'vs_nine_single': [dict(zip(('delta', 'ci'), paired(e_, e_nine))) for e_ in es],
                     'sigma_seeds': [round(float(v.std()), 4) for v in es], 'sigma_ensemble': round(float(e_ens.std()), 4),
                     'vs_nine': dict(zip(('delta', 'ci'), paired(e_ens, e_nine))),
                     'vs_engineered': dict(zip(('delta', 'ci'), paired(e_ens, e_eng)))}
    print('  %s: seeds %s, ensemble %.4f; vs nine %s; vs engineered %s'
          % (key, R['cnn'][key]['sigma_seeds'], e_ens.std(), R['cnn'][key]['vs_nine'], R['cnn'][key]['vs_engineered']),
          flush=True)
json.dump(R, open(OUT, 'w'), indent=2)
print('wrote %s' % OUT)
