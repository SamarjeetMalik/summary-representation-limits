"""Two additional figures.

fig4  a gallery of rendered "detector images", so the reader can see that they
      are a smooth four-parameter family rather than detector data;
fig5  training curves for the rendered-input models against the four-scalar
      ceiling, showing convergence ONTO the bound rather than through it.

Curve data is transcribed from the run logs of run_rendering.py / run_sweep.py
(epochs were sampled every 6-7). It is stored in results.json under
'curves' with that provenance recorded, so the figure is generated from the JSON
rather than from literals in this file.

Run:  python figures/make_figures_extra.py
"""
import json, os, sys, math, warnings
import numpy as np
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
ANA = os.path.join(os.path.dirname(HERE), 'analysis')
sys.path.insert(0, ANA)
import common as C

RES = os.path.join(ANA, 'results.json')
R = json.load(open(RES))

BLUE, ORANGE, GREEN, VERM = '#0072B2', '#E69F00', '#009E73', '#D55E00'
GREY = '#5A5A5A'
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 9,
    'axes.titlesize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .7, 'legend.frameon': False, 'figure.dpi': 150,
    'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': .02,
})

# ---- curve data, recorded once with provenance -----------------------------
# Training curves are written by run_rendering.py and run_sweep.py (every epoch).
# They used to be transcribed from stdout here; a --full rerun now refreshes them.
if 'curves' not in R or 'transcribed' in R['curves'].get('_provenance', ''):
    raise SystemExit('results.json lacks generated training curves: run '
                     'analysis/run_rendering.py and analysis/run_sweep.py first')

# ---------------------------------------------------------------- fig 4
NROW, NCOL, PITCH, R_M = 16, 25, 8.0, 103.0
gx = (np.arange(NCOL) - (NCOL - 1) / 2.0) * PITCH
gy = (np.arange(NROW) - (NROW - 1) / 2.0) * PITCH
GX, GY = np.meshgrid(gx, gy)


def nkg(Ne, s, xc, yc):
    r = np.sqrt((GX[None] - xc[:, None, None]) ** 2 + (GY[None] - yc[:, None, None]) ** 2)
    np.maximum(r, 1.0, out=r)
    lnC = gammaln(4.5 - s) - math.log(2 * math.pi) - gammaln(s) - gammaln(4.5 - 2 * s)
    x = r / R_M
    ln = (np.log(Ne)[:, None, None] - 2 * math.log(R_M) + lnC[:, None, None]
          + (s[:, None, None] - 2.0) * np.log(x) + (s[:, None, None] - 4.5) * np.log1p(x))
    return np.log10(1.0 + np.exp(ln)).astype(np.float32)


def fig_gallery():
    d = C.load_root()
    age = d['Age']
    # five events spanning the age range, which is what shapes the profile
    picks = [np.argmin(np.abs(age - q)) for q in (0.85, 1.00, 1.15, 1.30, 1.45)]
    picks = np.array(picks)
    im = nkg(d['NKGSize'][picks], age[picks], d['NKGX'][picks], d['NKGY'][picks])
    vmin, vmax = im.min(), im.max()
    fig, axes = plt.subplots(1, 5, figsize=(6.6, 1.5))
    for k, ax in enumerate(axes):
        ax.imshow(im[k], cmap='magma', vmin=vmin, vmax=vmax, aspect='equal',
                  extent=[gx[0], gx[-1], gy[0], gy[-1]], origin='lower')
        ax.set_title(r'$s=%.2f$' % age[picks[k]], fontsize=8.5, pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    # explanation lives in the LaTeX caption, not burned into the figure
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(os.path.join(HERE, 'fig4_gallery.pdf'))
    plt.close(fig)
    print('  fig4_gallery.pdf')


# ---------------------------------------------------------------- fig 5
def fig_curves():
    K = R['curves']
    A = R['rendering']['A_scalars4']
    D = R['rendering']['D_nine']
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ax.axhline(A, color=GREY, ls='--', lw=1.1, zorder=1)
    ax.text(29.4, A - 0.0007, 'row A: four fit scalars', ha='right', va='top',
            fontsize=7.8, color=GREY)
    ax.axhline(D, color=GREEN, ls=':', lw=1.0, zorder=1)
    ax.text(5.2, D + 0.0006, 'row D: nine summary features', ha='left', va='bottom',
            fontsize=7.8, color=GREEN)
    for key, col, mk, lbl in (('B_cnn', BLUE, 'o', 'CNN, rendered only'),
                              ('G_vit', ORANGE, '^', 'ViT, rendered only'),
                              ('C_cnn_muon', VERM, 's', 'CNN, rendered $+$ real muons')):
        ax.plot(K[key]['epoch'], K[key]['sigma'], '-', color=col, marker=mk,
                ms=4, lw=1.3, label=lbl, zorder=3)
    ax.set_xlabel('training epoch')
    ax.set_ylabel(r'test $\sigma\,(\log_{10}E)$')
    ax.set_xlim(0.5, 30.5)
    ax.legend(loc='center right', handlelength=1.7)
    fig.savefig(os.path.join(HERE, 'fig5_curves.pdf'))
    plt.close(fig)
    print('  fig5_curves.pdf')


if __name__ == '__main__':
    print('building extra figures...')
    fig_gallery()
    fig_curves()
    print('done')
