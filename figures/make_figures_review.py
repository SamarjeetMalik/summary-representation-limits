"""Figures added in response to the pre-submission review (all from results.json).

  fig6_ladder.pdf     information ladder: sigma per nested feature set, paired CIs, floors
  fig7_probe_map.pdf  degeneracy-probe map: R2(phi) vs R2(phi, M), GRAPES-3 and KASCADE
  fig8_regimes.pdf    sigma / regime floor for every estimator in five sub-regimes
  fig9_learning.pdf   learning curves (GB, MLP on nine features; CNN rows B, C) and floors
  fig10_energy.pdf    sigma and bias against true energy, bootstrap bands
  fig11_coverage.pdf  coverage calibration: raw quantile regression vs CQR
  fig0_pipeline.pdf   schematic of the summary pipeline and the rendering branch

Same palette and style as make_figures.py / make_figures_extra.py (Okabe-Ito).
Run:  python figures/make_figures_review.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ANA = os.path.join(os.path.dirname(HERE), 'analysis')
R = json.load(open(os.path.join(ANA, 'results.json')))
KP = os.path.join(ANA, 'kcdc', 'kcdc_results.json')
K = json.load(open(KP)) if os.path.exists(KP) else None

BLUE, ORANGE, GREEN, VERM = '#0072B2', '#E69F00', '#009E73', '#D55E00'
PURPLE, SKY, GREY = '#CC79A7', '#56B4E9', '#5A5A5A'
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 9,
    'axes.titlesize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .7, 'legend.frameon': False, 'figure.dpi': 150,
    'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': .02,
})
SIG = r'$\sigma\,(\log_{10}E)$'


def save(fig, name):
    fig.savefig(os.path.join(HERE, name)); plt.close(fig); print('  ' + name)


def floor_band(ax, val, lo, hi, label, col=GREY, xr=None, ha='right'):
    ax.axhspan(lo, hi, color=col, alpha=.15, lw=0, zorder=0)
    ax.axhline(val, color=col, ls='--', lw=.9, zorder=1)
    if label:
        x = xr if xr is not None else ax.get_xlim()[1]
        ax.text(x, val, label, ha=ha, va='bottom', fontsize=7.5, color=col)


def fig_ladder():
    L = R['ladder']['rows']; B = R['bayes']
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    x = np.arange(len(L))
    s = [r['sigma'] for r in L]
    ax.plot(x, s, '-', color=BLUE, lw=1.3, zorder=3)
    for i, r in enumerate(L):
        ci = r.get('ci')
        if ci:
            ax.plot([i, i], ci, color=BLUE, lw=2.5, alpha=.5, zorder=2)
        ax.plot(i, r['sigma'], 'o', color=BLUE, ms=4.5, zorder=4)
        if 'delta' in r:
            sigd = r['delta_ci'][0] > 0 or r['delta_ci'][1] < 0
            txt = ('%+.4f' % r['delta']) if abs(r['delta']) >= 1e-3 else ('%+.5f' % r['delta'])
            ax.text(i, r['sigma'] + 0.0012, txt + ('' if sigd else '\n(n.s.)'),
                    ha='center', va='bottom', fontsize=6.6, color=BLUE if sigd else GREY, linespacing=1.0)
    P = R['ladder']['permutation']
    ax.plot(len(L) - 1 + .25, P['sigma_perm'], 'x', color=VERM, ms=6, mew=1.3, zorder=4)
    ax.annotate('modules permuted\nwithin each event', xy=(len(L) - 1 + .25, P['sigma_perm']),
                xytext=(len(L) - 2.2, 0.110), fontsize=6.5, color=VERM,
                arrowprops=dict(arrowstyle='-', color=VERM, lw=.6))
    ax.set_xlim(-.4, len(L) - .1)
    floor_band(ax, B['four']['rms_floor'], *B['four']['ci'], 'floor, four parameters', xr=len(L) - .3)
    floor_band(ax, B['nine']['rms_floor'], *B['nine']['ci'], 'floor, nine features', col=GREEN, xr=-.3, ha='left')
    short = ['four fit\nparameters', '+ zenith\nangle', '+ core\ndistance', '+ total\nmuons',
             '+ muon ratio\n(nine features)', '+ 16 module\ncounts']
    ax.set_xticks(x); ax.set_xticklabels(short[:len(L)], fontsize=7)
    ax.set_ylabel(SIG + ', gradient boosting')
    save(fig, 'fig6_ladder.pdf')


def fig_probe_map():
    P = R['probe_calibration']; rows = P['rows']
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.plot([0.55, 1.002], [0.55, 1.002], color=GREY, lw=.7, ls=':')
    pts = [('gb_phi', 'GB on $\\phi$', BLUE, 'o'), ('linear_Ne_s', 'size-age calib.', BLUE, 's'),
           ('cnn_rendered', 'CNN, rendered', BLUE, '^'), ('vit_rendered', 'ViT, rendered', BLUE, 'v'),
           ('cnn_rendered_muon', 'CNN, rendered + muons', GREEN, '^'), ('gb_nine', 'GB, nine features', GREEN, 'o'),
           ('true_energy', 'true energy', GREY, '*')]
    for k, lab, col, mk in pts:
        ax.plot(rows[k]['phi'], rows[k]['phi_M'], mk, color=col, ms=5.5, label=lab, zorder=3)
    for i, (tau, r) in enumerate(P['noise'].items()):
        ax.plot(r['phi'], r['phi_M'], 'D', color=ORANGE, ms=4.5, label='$\\phi$-model + noise' if i == 0 else None)
    for i, (lam, r) in enumerate(P['mix'].items()):
        ax.plot(r['phi'], r['phi_M'], 'P', color=GREEN, ms=5, mfc='none', label='$\\phi$-model + muon mix' if i == 0 else None)
    a = rows['archived']
    ax.plot(a['phi'], a['phi_M'], 'X', color=VERM, ms=9, mec='k', mew=.5, label='archived CNN/ViT', zorder=5)
    if K:
        kp = K['probe']
        kk = [kp[k] for k in kp if 'N(0' in k] + [kp[k] for k in kp if 'u(M)' in k]
        ax.plot([v['phi'] for v in kk], [v['phi_M'] for v in kk], '.', color=GREY, ms=4, alpha=.8,
                label='KASCADE controls')
    ax.set_xlabel(r'$R^2_{\mathrm{deg}}(\phi)$, fitted parameters only')
    ax.set_ylabel(r'$R^2_{\mathrm{deg}}(\phi, M)$, + muons and zenith')
    ax.set_xlim(0.55, 1.005); ax.set_ylim(0.55, 1.005)
    ax.text(0.57, 0.985, 'information outside $\\phi$:\nleft of the diagonal, at the top', fontsize=6.8, color=GREEN, va='top')
    ax.text(0.78, 0.68, 'noise only:\non the diagonal', fontsize=6.8, color=ORANGE)
    ax.legend(loc='center left', fontsize=6.6, handletextpad=.3, bbox_to_anchor=(1.02, 0.5))
    save(fig, 'fig7_probe_map.pdf')


def fig_regimes():
    G = R['regimes']['rows']; names = list(G); EX = R.get('group_checks', {}).get('regime_excess')
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    models = [('Ridge', SKY, 'o'), ('RandomForest', GREEN, 's'), ('MLP', PURPLE, 'D')]
    for j, (m, col, mk) in enumerate(models):
        ratio = [100 * (G[n]['sigma'][m] / G[n]['floor'] - 1) for n in names]
        ax.plot(np.arange(len(names)) + (j - 1) * .1 + .18, ratio, mk, color=col, ms=4, alpha=.8,
                label={'RandomForest': 'random forest'}.get(m, m))
    if EX:
        v = [EX[n]['excess_pct'] for n in names]
        lo = [EX[n]['excess_pct'] - EX[n]['ci'][0] for n in names]; hi = [EX[n]['ci'][1] - EX[n]['excess_pct'] for n in names]
        ax.errorbar(np.arange(len(names)) - .12, v, yerr=[lo, hi], fmt='^', color=BLUE, ms=5.5, capsize=2.5,
                    lw=1.2, label='grad. boosting (95% CI)')
    ax.axhline(0, color=GREY, lw=.8, ls='--')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace('logNe ', r'$\log_{10}N_e$ ').replace('zenith ', 'zenith\n') for n in names], fontsize=7.5)
    ax.set_ylabel('$\\sigma$ above regime floor estimate (%)')
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.18), fontsize=7)
    save(fig, 'fig8_regimes.pdf')


def fig_learning():
    D = R['diagnostics']['learning_curves']; Cc = R.get('cnn_curves'); B = R['bayes']
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    for m, col, mk, lab in (('GradBoost', BLUE, '^', 'grad. boosting, nine features'),
                            ('MLP', PURPLE, 'D', 'MLP, nine features')):
        ax.errorbar(D[m]['n'], D[m]['sigma'], yerr=D[m]['sd'], color=col, marker=mk, ms=4, lw=1.2,
                    capsize=1.5, label=lab)
    if Cc:
        ax.plot(Cc['n'], Cc['B'], 'o-', color=ORANGE, ms=4, lw=1.2, label='CNN, rendered (B)')
        ax.plot(Cc['n'], Cc['C'], 's-', color=VERM, ms=4, lw=1.2, label='CNN, rendered + muons (C)')
    ax.set_xscale('log'); ax.set_xlim(800, 1.8e5); ax.set_ylim(0.098, 0.134)
    floor_band(ax, B['four']['rms_floor'], *B['four']['ci'], 'floor, four parameters', xr=850, ha='left')
    floor_band(ax, B['nine']['rms_floor'], *B['nine']['ci'], 'floor, nine features', col=GREEN, xr=850, ha='left')
    mlp0 = D['MLP']['sigma'][0]
    if mlp0 > 0.134:
        ax.annotate('MLP at %d events: %.3f' % (D['MLP']['n'][0], mlp0), xy=(D['MLP']['n'][0], 0.1335),
                    xytext=(1.6e3, 0.1325), fontsize=6.4, color=PURPLE,
                    arrowprops=dict(arrowstyle='->', color=PURPLE, lw=.6))
    ax.set_xlabel('training events'); ax.set_ylabel('test ' + SIG)
    ax.legend(fontsize=6.8, loc='upper right')
    save(fig, 'fig9_learning.pdf')


def fig_energy():
    D = R['diagnostics']; xc = D['energy_centres']; V = D['vs_energy']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    sty = [('Ridge', SKY, 'o'), ('RandomForest', GREEN, 's'), ('GradBoost', BLUE, '^'), ('MLP', PURPLE, 'D'),
           ('size-age calibration', VERM, 'v'), ('archived CNN/ViT', GREY, 'x')]
    for k, col, mk in sty:
        rows = V[k]
        for ax, key in ((a1, 'sigma'), (a2, 'bias')):
            v = [r[key] for r in rows]; lo = [r[key + '_ci'][0] for r in rows]; hi = [r[key + '_ci'][1] for r in rows]
            ax.fill_between(xc, lo, hi, color=col, alpha=.15, lw=0)
            ax.plot(xc, v, '-', marker=mk, color=col, ms=3.2, lw=1.1,
                    label={'RandomForest': 'random forest', 'GradBoost': 'grad. boosting'}.get(k, k) if ax is a1 else None)
    a2.axhline(0, color=GREY, lw=.7)
    a1.set_ylabel(SIG); a2.set_ylabel('mean bias'); a1.set_xlabel('true $\\log_{10}(E/\\mathrm{GeV})$')
    a2.set_xlabel('true $\\log_{10}(E/\\mathrm{GeV})$')
    fig.legend(ncol=3, loc='upper center', bbox_to_anchor=(.5, 1.1), fontsize=7)
    fig.tight_layout()
    save(fig, 'fig10_energy.pdf')


def fig_coverage():
    Cv = R['diagnostics']['coverage_curve']
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    nom = [100 * v for v in Cv['nominal']]
    ax.plot([45, 100], [45, 100], color=GREY, lw=.8, ls='--', label='ideal')
    ax.plot(nom, Cv['raw'], 'o-', color=ORANGE, ms=4, lw=1.2, label='quantile regression')
    ax.plot(nom, Cv['cqr'], 's-', color=BLUE, ms=4, lw=1.2, label='conformalised (CQR)')
    ax.set_xlabel('nominal coverage (%)'); ax.set_ylabel('empirical coverage (%)')
    ax.set_xlim(45, 98); ax.set_ylim(45, 98); ax.legend(loc='upper left', fontsize=7)
    ins = ax.inset_axes([.58, .1, .38, .32])
    ins.plot(nom, np.array(Cv['raw']) - nom, 'o-', color=ORANGE, ms=2.5, lw=1)
    ins.plot(nom, np.array(Cv['cqr']) - nom, 's-', color=BLUE, ms=2.5, lw=1)
    ins.axhline(0, color=GREY, lw=.6, ls='--'); ins.set_title('empirical $-$ nominal', fontsize=6.5)
    ins.tick_params(labelsize=6)
    save(fig, 'fig11_coverage.pdf')


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.0, 2.5)); ax.axis('off'); ax.set_xlim(0, 14); ax.set_ylim(-1.0, 4.2)

    def box(x, y, w, h, text, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05,rounding_size=0.12',
                                    fc=col, ec=col, lw=.6, alpha=.2))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=7)

    def arrow(x0, y0, x1, y1, col=GREY):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', color=col, lw=1))

    W, H = 2.9, 1.15
    xs = [0.0, 3.6, 7.2, 10.8]
    top, bot = 2.75, 0.35
    box(xs[0], top, W, H, 'about 400 scintillator\ndensities and times', GREY)
    box(xs[1], top, W, H, 'fitted parameters\n$\\phi=(\\log N_e,s,X_c,Y_c)$', BLUE)
    box(xs[2], top, W, H, 'nine-feature summary\n$\\phi$ + zenith + muons', BLUE)
    box(xs[3], top, W, H, 'learned estimator:\nplateau at the floor', GREEN)
    box(xs[0], bot, W, H, '16 muon-module\ntrack counts $M$', GREY)
    box(xs[1], bot, W, H, 'rendered image $g(\\phi)$\n(no detector data)', ORANGE)
    box(xs[2], bot, W, H, 'CNN or ViT on\n$g(\\phi)$, with or without $M$', ORANGE)
    box(xs[3], bot, W, H, 'bounded by the\n$\\phi$ (or $\\phi,M$) floor', VERM)
    for k in range(3):
        arrow(xs[k] + W, top + H / 2, xs[k + 1], top + H / 2)
        arrow(xs[k] + W, bot + H / 2, xs[k + 1], bot + H / 2) if k > 0 else None
    ax.text((xs[0] + W + xs[1]) / 2, top + H + 0.12, 'NKG fit\n(lossy)', ha='center', va='bottom', fontsize=6.6, color=GREY)
    arrow(xs[1] + W / 2, top, xs[1] + W / 2, bot + H, ORANGE)
    ax.text(xs[1] + W / 2 + 0.12, (top + bot + H) / 2, 'deterministic\nrendering', ha='left', va='center',
            fontsize=6.6, color=ORANGE)
    ax.annotate('', xy=(xs[2] + 0.5, bot), xytext=(xs[0] + W - 0.5, bot),
                arrowprops=dict(arrowstyle='->', color=GREY, lw=.8, connectionstyle='arc3,rad=0.25'))
    ax.text((xs[1] + xs[1] + W) / 2, bot - 0.85, 'measured muon channel', ha='center', fontsize=6.4, color=GREY)
    ax.text(xs[3] + W / 2, (top + bot + H) / 2, 'Proposition 1', ha='center', va='center', fontsize=7,
            color=VERM, style='italic')
    save(fig, 'fig0_pipeline.pdf')


if __name__ == '__main__':
    for f in (fig_pipeline, fig_ladder, fig_probe_map, fig_regimes, fig_learning, fig_energy, fig_coverage):
        try:
            f()
        except KeyError as e:
            print('  %s skipped: results.json lacks %s' % (f.__name__, e))
