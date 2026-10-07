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
                    ha='center', va='bottom', fontsize=7.4, color=BLUE if sigd else GREY, linespacing=1.0)
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
    ax.set_xticks(x); ax.set_xticklabels(short[:len(L)], fontsize=8)
    ax.set_ylabel(SIG + ', gradient boosting')
    save(fig, 'fig6_ladder.pdf')


def fig_probe_map(P=None, out='fig7_probe_map.pdf'):
    P = P or R['probe_calibration']; rows = P['rows']
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
    # the three readings of the map (the protocol's step 4)
    kw = dict(fontsize=6.4, color='0.35', style='italic')
    ax.annotate('function of $\\phi$', xy=(0.997, 0.997), xytext=(0.965, 0.87), ha='center', **kw,
                arrowprops=dict(arrowstyle='-', color='0.6', lw=.5))
    ax.text(0.70, 0.975, 'uses a measured channel outside $\\phi$', ha='center', va='top', **kw)
    ax.text(0.80, 0.765, 'noise-like, or an unlisted channel', ha='left', va='center', rotation=38, **kw)
    ax.legend(loc='center left', fontsize=6.6, handletextpad=.3, bbox_to_anchor=(1.02, 0.5))
    save(fig, out)


def fig_regimes(RG=None, EX=None, out='fig8_regimes.pdf'):
    G = (RG or R['regimes'])['rows']; names = list(G)
    EX = EX or R.get('group_checks', {}).get('regime_excess')
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
        for i_, vv in enumerate(v):
            ax.text(i_ - .12, EX[names[i_]]['ci'][0] - 0.25, '%+.1f%%' % vv, ha='center', va='top', fontsize=7, color=BLUE)
        ax.set_ylim(min(EX[n]['ci'][0] for n in names) - 1.0, None)
    ax.axhline(0, color='k', lw=1.1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace('logNe ', r'$\log_{10}N_e$ ').replace('zenith ', 'zenith\n') for n in names], fontsize=7.5)
    ax.set_ylabel('$\\sigma$ above regime floor estimate (%)')
    ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.18), fontsize=7)
    save(fig, out)


def fig_learning():
    D = R['diagnostics']['learning_curves']; Cc = R.get('cnn_curves'); B = R['bayes']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.7), sharex=True)
    for m, col, mk, lab in (('GradBoost', BLUE, '^', 'gradient boosting'), ('MLP', PURPLE, 'D', 'MLP')):
        a1.errorbar(D[m]['n'], D[m]['sigma'], yerr=D[m]['sd'], color=col, marker=mk, ms=4, lw=1.2, capsize=1.5, label=lab)
    floor_band(a1, B['nine']['rms_floor'], *B['nine']['ci'], 'floor estimate, nine features', col=GREEN, xr=900, ha='left')
    a1.set_ylim(0.098, 0.124); a1.set_xscale('log')
    mlp0 = D['MLP']['sigma'][0]
    if mlp0 > 0.124:
        a1.text(1.3e3, 0.1228, '(MLP at %d events: %.3f, off scale)' % (D['MLP']['n'][0], mlp0), fontsize=6.2,
                color=PURPLE, va='top')
    a1.set_title('summary representation (nine features)', fontsize=8); a1.set_xlabel('training events')
    a1.set_ylabel('test ' + SIG); a1.legend(fontsize=7, loc='center right')
    if Cc:
        a2.plot(Cc['n'], Cc['B'], 'o-', color=ORANGE, ms=4, lw=1.2, label='CNN, rendered image (B)')
        a2.plot(Cc['n'], Cc['C'], 's-', color=VERM, ms=4, lw=1.2, label='CNN, rendered + muons (C)')
    floor_band(a2, B['four']['rms_floor'], *B['four']['ci'], 'floor estimate, four parameters', xr=900, ha='left')
    a2.set_xscale('log'); a2.set_xlim(800, 1.8e5); a2.set_ylim(0.100, 0.128)
    a2.set_title('rendered representation', fontsize=8); a2.set_xlabel('training events')
    a2.legend(fontsize=7, loc='lower right')
    fig.tight_layout()
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
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.6, 4.0), sharex=True, gridspec_kw={'height_ratios': [2.2, 1]})
    nom = [100 * v for v in Cv['nominal']]
    a1.plot([45, 100], [45, 100], color=GREY, lw=.8, ls='--', label='ideal')
    a1.plot(nom, Cv['raw'], 'o-', color=ORANGE, ms=4, lw=1.2, label='quantile regression')
    a1.plot(nom, Cv['cqr'], 's-', color=BLUE, ms=4, lw=1.2, label='conformalised (CQR)')
    a1.set_ylabel('empirical coverage (%)'); a1.set_xlim(45, 98); a1.set_ylim(45, 98); a1.legend(loc='upper left', fontsize=7)
    a2.axhline(0, color='k', lw=.9)
    a2.plot(nom, np.array(Cv['raw']) - nom, 'o-', color=ORANGE, ms=4, lw=1.2)
    a2.plot(nom, np.array(Cv['cqr']) - nom, 's-', color=BLUE, ms=4, lw=1.2)
    a2.set_ylabel('empirical $-$\nnominal (pp)'); a2.set_xlabel('nominal coverage (%)')
    fig.tight_layout()
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


def fig_audit_flow():
    """The audit as a flow: each step's question, tool and control, and how its outcomes are read."""
    fig, ax = plt.subplots(figsize=(7.0, 7.3)); ax.axis('off'); ax.set_xlim(-0.15, 14.15); ax.set_ylim(-1.2, 14.4)
    XL, WL, XR, WR, H, GAP = 0.0, 6.7, 7.4, 6.6, 1.82, 0.36

    def box(x, y, w, h, col, alpha=.16, ls='-'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.12',
                                    fc=col, ec=col, lw=.8, alpha=alpha, ls=ls))

    def arrow(x0, y0, x1, y1, col=GREY, lw=.9):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', color=col, lw=lw))

    top = 13.5
    box(XL, top, WL, 0.8, GREY)
    ax.text(XL + WL / 2, top + 0.4, 'models trained on a representation $T$ stop improving',
            ha='center', va='center', fontsize=7.0)
    box(XR, top, WR, 0.8, GREY, alpha=.0)
    ax.text(XR + WR / 2, top + 0.4, 'how each outcome is read', ha='center', va='center', fontsize=7.4,
            style='italic', color=GREY)

    steps = [
        ('1  Floor', 'What can this representation support?',
         'model-free estimate $\\hat\\sigma_{\\mathrm{B}}(T)$ of the floor, beside\nestimators of unrelated inductive bias',
         'control: known noise on the real feature matrix', BLUE,
         [('$\\sigma$ within the spread of $\\hat\\sigma_{\\mathrm{B}}$: representation-limited', GREEN),
          ('$\\sigma$ well above it, or estimators disagree:\nlearner-limited; change learner, data, optimisation', PURPLE)]),
        ('2  Ladder', 'Which channels carry the information?',
         'nested feature sets on fixed events,\npaired bootstrap of each change',
         'control: within-event permutation of a structured channel', BLUE,
         [('error falls: the channel helps this learner; a Bayes-\nrisk gain is shown only below the old estimated floor', GREEN),
          ('no change: no detectable gain in the conditional\nmean for this learner (Proposition 2)', GREY)]),
        ('3  Rendering check', 'Is an input presented as richer really measured?',
         'channel-by-channel record: measured, or\ncomputed from $T$',
         'control: outcome predicted from step 1 before training', VERM,
         [('computed from $T$: cannot lower the squared-loss\nBayes risk (Proposition 1)', VERM),
          ('measured: it may; test it (positive control)', GREEN)]),
        ('4  Degeneracy probe', 'Did a trained model use anything beyond $T$?',
         'out-of-fold regression of the output $m$ on $T$\nand on $(T,M)$',
         'control: outputs of known construction; two training runs', BLUE,
         [('near 1 on $T$: output is a function of $T$', GREY),
          ('low on $T$, near 1 on $(T,M)$: model used $M$', GREEN),
          ('low on both: noise-like, or an unlisted channel', GREY)]),
        ('5  Selection', 'Does an inherited selection $S$ still help here?',
         'score with and without $S$, retrain within $S$,\nreport what $S$ was meant to protect',
         'control: stratify by what drives selection and error', BLUE,
         [('worse with $S$: counterproductive in this regime', GREY),
          ('better with $S$: keep it', GREY)]),
        ('Validation  Second system', 'Does the diagnosis hold elsewhere?',
         'repeat on an independent system', '', BLUE,
         [('report what replicates and what does not', GREY)]),
    ]
    y = top - GAP - H
    for i, (name, q, tool, ctrl, col, outs) in enumerate(steps):
        box(XL, y, WL, H, col)
        ax.text(XL + 0.18, y + H - 0.1, name, ha='left', va='top', fontsize=7.6, fontweight='bold')
        ax.text(XL + 0.18, y + H - 0.47, q, ha='left', va='top', fontsize=6.9, style='italic')
        ax.text(XL + 0.18, y + H - 0.84, tool, ha='left', va='top', fontsize=6.6, linespacing=1.15)
        ax.text(XL + 0.18, y + 0.09, ctrl, ha='left', va='bottom', fontsize=6.2, color='0.3')
        arrow(XL + WL / 2, y + H + GAP - 0.02, XL + WL / 2, y + H + 0.04)
        n = len(outs)
        for j, (txt, c) in enumerate(outs):
            hh = (H - 0.1 * (n - 1)) / n
            yy = y + H - (j + 1) * hh - 0.1 * j
            box(XR, yy, WR, hh, c, alpha=.12)
            ax.text(XR + 0.16, yy + hh / 2, txt, ha='left', va='center', fontsize=6.4, linespacing=1.1)
            arrow(XL + WL + 0.06, y + H / 2, XR - 0.06, yy + hh / 2, col='0.55', lw=.6)
        y -= H + GAP
    # legend
    yl = -1.1
    for x0, c, lab in ((0.0, VERM, 'guarantee'), (2.2, BLUE, 'estimate, read with its control'),
                       (7.7, GREEN, 'positive finding'), (10.6, PURPLE, 'learner limit'), (13.0, GREY, 'other')):
        box(x0, yl, 0.42, 0.34, c, alpha=.3)
        ax.text(x0 + 0.58, yl + 0.17, lab, ha='left', va='center', fontsize=6.4)
    save(fig, 'fig13_audit_flow.pdf')


def fig_benchmark():
    """The known-truth benchmark: the floor estimate against the truth, and learners against the floor."""
    S = R['synthetic']['rows']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={'width_ratios': [1, 1.15], 'wspace': .32})
    keys = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T+M', 'T+M+V', 'X']
    dims = [1, 2, 3, 4, 5, 6, 7, 8, 32]
    err = [S[k]['gamma_err_pct'] for k in keys]
    a1.axhspan(-2, 2, color=GREEN, alpha=.12, lw=0)
    a1.axhline(0, color='k', lw=.8)
    a1.plot(range(len(keys)), err, 'o-', color=BLUE, ms=4.5, lw=1)
    a1.set_xticks(range(len(keys))); a1.set_xticklabels([str(d) for d in dims])
    a1.set_xlabel('dimension of the representation'); a1.set_ylabel('floor estimate $-$ truth (%)')
    a1.annotate('%+.0f%%' % err[-1], xy=(len(keys) - 1, err[-1]), xytext=(len(keys) - 2.6, err[-1] - 1.5), fontsize=7.5, color=BLUE)
    a1.text(0.1, 2.6, '$\\pm2\\%$', fontsize=7, color=GREEN)
    a1.set_title('step 1 against the truth', fontsize=8.5)
    reps = [('T6', 'summary $T$'), ('T+M', '$T$ and $M$'), ('X', 'measurement $X$')]
    x = np.arange(len(reps)); w = .26
    a2.bar(x - w, [S[k]['true_floor'] for k, _ in reps], w, color='0.25', label='true floor')
    a2.bar(x, [S[k]['gb'] for k, _ in reps], w, color=BLUE, label='gradient boosting')
    a2.bar(x + w, [S[k]['mlp'] for k, _ in reps], w, color=PURPLE, label='MLP')
    a2.set_xticks(x); a2.set_xticklabels([l for _, l in reps], fontsize=7.5)
    a2.set_ylabel('root-mean-square error'); a2.legend(fontsize=7, loc='upper right')
    a2.annotate('learner-limited', xy=(2, S['X']['gb']), xytext=(1.25, S['X']['gb'] + 0.22), fontsize=7.5, color=BLUE,
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=.7))
    a2.set_title('learners against the true floor', fontsize=8.5)
    save(fig, 'fig14_benchmark.pdf')


def fig_kcdc_audit():
    KF = json.load(open(os.path.join(ANA, 'kcdc', 'kcdc_floor_results.json')))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 3.0), gridspec_kw={'width_ratios': [1, 1.25]})
    L = KF['ladder']; x = np.arange(len(L))
    a1.plot(x, [r['sigma'] for r in L], 'o-', color=BLUE, ms=4, lw=1.2)
    a1.axhline(KF['floor']['nine'], color=GREEN, ls='--', lw=.9)
    a1.text(-.2, KF['floor']['nine'], 'floor estimate, nine features', fontsize=6.6, color=GREEN, va='bottom')
    a1.set_xticks(x); a1.set_xticklabels(['four fit\nparams', '+ zenith', '+ core\ndist.', '+ muon\nsize', '+ muon\nratio'],
                                         fontsize=6.6)
    a1.set_ylabel(SIG); a1.set_title('KASCADE information ladder', fontsize=8)
    it = KF['intermediate']; labs = ['nine features'] + ['+ ' + k for k in it]
    vals = [KF['oracle']['sigma_nine']] + [v['sigma'] for v in it.values()]
    cols = [BLUE] + [ORANGE] * len(it)
    CP = os.path.join(ANA, 'kcdc', 'kcdc_cnn_results.json')
    if os.path.exists(CP):
        CN = json.load(open(CP))['cnn']
        labs += ['CNN, measured image only', '+ CNN on measured image']
        vals += [np.mean(CN['image only']['sigma_seeds']), np.mean(CN['residual on GB(nine)']['sigma_seeds'])]
        cols += [GREEN, GREEN]
    labs += ['MC-truth size oracle']; vals += [KF['oracle']['sigma_true_sizes']]; cols += [GREY]
    yy = np.arange(len(labs))[::-1]
    a2.barh(yy, vals, color=cols, alpha=.75, height=.6)
    for yi, v in zip(yy, vals):
        a2.text(v + .001, yi, '%.4f' % v, va='center', fontsize=6.4)
    a2.set_yticks(yy); a2.set_yticklabels([l.replace('+ profiles + counts, totals, timing', '+ profiles, counts,\ntotals and timing')
                                           for l in labs], fontsize=6.6)
    a2.set_xlim(min(vals) * 0.9, max(vals) * 1.06); a2.set_xlabel(SIG)
    a2.set_title('what the per-station signals add', fontsize=8)
    fig.tight_layout()
    save(fig, 'fig12_kcdc_audit.pdf')


if __name__ == '__main__':
    for f in (fig_pipeline, fig_audit_flow, fig_benchmark, fig_ladder, fig_probe_map, fig_regimes, fig_learning, fig_energy, fig_coverage,
              fig_kcdc_audit):
        try:
            f()
        except (KeyError, FileNotFoundError) as e:
            print('  %s skipped: results.json lacks %s' % (f.__name__, e))
    CS = R.get('crossfit_secondary')                      # the same two figures under the shower cross-fit
    if CS:
        fig_probe_map(CS['probe_calibration'], 'fig7_probe_map_cf.pdf')
        fig_regimes(CS['regimes'], CS['regimes']['regime_excess'], 'fig8_regimes_cf.pdf')
