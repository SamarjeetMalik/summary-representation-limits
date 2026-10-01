"""Three figures for the R/PS 2026 Research Note.

Reads analysis/results.json for Fig 1 and Fig 3; recomputes the (cheap) scalar
regression for Fig 2, which needs point-level data rather than summaries.

Run:  python figures/make_figures.py
"""
import json, os, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'analysis'))
import common as C

RES = json.load(open(os.path.join(os.path.dirname(HERE), 'analysis', 'results.json')))

# Okabe-Ito, colourblind-safe
BLUE, ORANGE, GREEN, VERM = '#0072B2', '#E69F00', '#009E73', '#D55E00'
PURPLE, SKY, GREY = '#CC79A7', '#56B4E9', '#5A5A5A'

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9,
    'axes.labelsize': 9.5, 'axes.titlesize': 9.5,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .7, 'xtick.major.width': .7, 'ytick.major.width': .7,
    'legend.frameon': False, 'figure.dpi': 150,
    'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': .02,
})


def bin_centres():
    return [np.mean([float(x) for x in b['bin'].split('-')])
            for b in RES['age_curves']['GradBoost']]


# ---------------------------------------------------------------- Figure 1
# The archived 'NKG' column is a two-coefficient size-age calibration, not an analytic
# reconstruction (analysis/run_nkg_baseline.py); label it for what it is.
LEGEND = {'NKG': 'size-age calib.', 'RandomForest': 'random forest', 'GradBoost': 'grad. boosting'}


def fig1():
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    x = bin_centres()
    style = [('Ridge', SKY, '-', 'o', 1.1), ('RandomForest', GREEN, '-', 's', 1.1),
             ('GradBoost', BLUE, '-', '^', 1.1), ('MLP', PURPLE, '-', 'D', 1.1),
             ('NKG', VERM, '-', 'v', 1.9), ('CNN/ViT', GREY, '--', 'x', 1.3)]
    for name, col, ls, mk, lw in style:
        if name not in RES['age_curves']:
            continue
        ys = [b['sigma'] for b in RES['age_curves'][name]]
        keep = [(a, b) for a, b in zip(x, ys) if b is not None]
        ax.plot([a for a, _ in keep], [b for _, b in keep], ls, color=col, lw=lw,
                marker=mk, ms=3.4, mew=.9, label=LEGEND.get(name, name), zorder=3 if name == 'NKG' else 2)
    ax.axvspan(0.90, 1.15, color=ORANGE, alpha=.07, lw=0, zorder=0)
    ns = [b_['n'] for b_ in RES['age_curves']['GradBoost']]
    for xx, nn in zip(x, ns):                                   # events per age bin (test set)
        ax.text(xx, 0.02, '%s' % format(nn, ','), transform=ax.get_xaxis_transform(), ha='center',
                fontsize=6, color=GREY)
    ax.text(0.005, 0.07, 'events per bin:', transform=ax.transAxes, fontsize=6, color=GREY)
    ax.text(1.025, ax.get_ylim()[1], 'standard\nquality cut', ha='center', va='top',
            fontsize=7.5, color='#8A5410', linespacing=1.2)
    ax.set_xlabel('shower age $s$')
    ax.set_ylabel(r'$\sigma\,(\log_{10} E)$')
    ax.xaxis.set_minor_locator(MultipleLocator(.05))
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2), handlelength=1.6,
              columnspacing=1.1)   # below the axes: inside, it collided with the cut label
    fig.savefig(os.path.join(HERE, 'fig1_estimators.pdf'))
    plt.close(fig)
    print('  fig1_estimators.pdf')


# ---------------------------------------------------------------- Figure 2
def fig2():
    from sklearn.ensemble import HistGradientBoostingRegressor as GBR
    from sklearn.model_selection import train_test_split

    d = C.load_root(); c = C.load_csv()
    r_idx, c_idx = C.match(d, c)
    S4 = np.column_stack([d['log_Ne'][r_idx], d['Age'][r_idx],
                          d['NKGX'][r_idx], d['NKGY'][r_idx]])
    targets = [('size-age\ncalibration', c['nkg'][c_idx], VERM),
               ('CNN/ViT on\nrendered images', c['cnnvit'][c_idx], BLUE),
               ('true primary\nenergy', d['y'][r_idx], GREEN)]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.95), sharex=True, sharey=True)
    for ax, (lbl, tgt, col) in zip(axes, targets):
        a, b, ta, tb = train_test_split(S4, tgt, test_size=.25, random_state=0)
        pred = GBR(max_iter=400, random_state=0).fit(a, ta).predict(b)
        r2 = C.r2(pred, tb)
        ax.hexbin(pred, tb, gridsize=52, cmap='Greys', bins='log',
                  mincnt=1, linewidths=0, zorder=1)
        lo, hi = 4.9, 6.3
        ax.plot([lo, hi], [lo, hi], color=col, lw=1.2, ls='--', zorder=3)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.set_title(lbl.replace(chr(10),' '), fontsize=7.6)
        ax.text(.045, .955, r'$R^2 = %.3f$' % r2, transform=ax.transAxes,
                ha='left', va='top', fontsize=8.0, color=col, weight='bold')
        ax.text(.045, .86, r'resid sd $= %.4f$' % (tb - pred).std(),
                transform=ax.transAxes, ha='left', va='top', fontsize=6.8, color='#444')
    axes[0].set_ylabel('estimator output')
    # The x-axis description lives in the LaTeX caption: at this compressed
    # height a supxlabel collides with the tick labels.
    fig.savefig(os.path.join(HERE, 'fig2_degeneracy.pdf'))
    plt.close(fig)
    print('  fig2_degeneracy.pdf')


# ---------------------------------------------------------------- Figure 3
def fig3():
    """Age against energy over the whole slice and inside one 0.1-wide size window, and the
    age-energy correlation against window width (run_mechanism.py)."""
    d = C.load_root(); y = d['y']; lne, s = d['log_Ne'], d['Age']
    W = RES['mechanism_width']; mech = RES['mechanism']
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4), gridspec_kw={'width_ratios': [1, 1, 1.05]})
    win = mech[-1]; a_, b_ = [float(v) for v in win['window'].split('-')]
    for ax, m, title in ((axs[0], np.ones(len(y), bool), r'whole slice, $\log_{10}N_e\in[4.8,5.2]$'),
                         (axs[1], (lne >= a_) & (lne < b_), r'$\log_{10}N_e\in[%.2f,%.2f)$' % (a_, b_))):
        ax.hexbin(y[m], s[m], gridsize=45, cmap='Blues', bins='log', mincnt=1, linewidths=0)
        r_ = np.corrcoef(s[m], y[m])[0, 1]
        ax.set_title(title, fontsize=7.5); ax.text(.04, .93, 'corr = %.3f' % r_, transform=ax.transAxes, fontsize=7.5, va='top')
        ax.set_xlabel(r'true $\log_{10}(E/\mathrm{GeV})$'); ax.set_ylim(0.6, 1.5)
    axs[0].set_ylabel('shower age $s$')
    ax = axs[2]
    w = [r['width'] for r in W]; cm = [r['corr_mean'] for r in W]
    lo = [r['corr_min'] for r in W]; hi = [r['corr_max'] for r in W]
    ax.fill_between(w, lo, hi, color=BLUE, alpha=.15, lw=0, label='range over window positions')
    ax.plot(w, cm, 'o-', color=BLUE, ms=4, lw=1.3, label='mean over positions')
    ax.set_xscale('log'); ax.set_xlabel(r'window width in $\log_{10}N_e$'); ax.set_ylabel(r'corr$(s,\ \log_{10}E)$')
    ax.set_title('correlation vs window width', fontsize=7.5); ax.legend(fontsize=6.3, loc='upper center', bbox_to_anchor=(.5, -.32), ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'fig3_mechanism.pdf'))
    plt.close(fig)
    print('  fig3_mechanism.pdf')


if __name__ == '__main__':
    print('building figures...')
    fig1(); fig3(); fig2()
    print('done')
