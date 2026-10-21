"""Figures for the physics-journal manuscript (epjc.tex).

fig_e1_decades.pdf: by decade of true energy and for three hadronic-interaction models,
  left  the excess of gradient boosting over the Gamma-test floor estimate (shower-bootstrap
        interval), right the share of error variance lost to observation and reconstruction
        (oracle with true particle numbers).
fig_e2_measured.pdf (only once analysis/kcdc/kcdc_measured_results.json exists): mean log Nmu
  at fixed log Ne of the measured data on each model's proton-to-iron scale.
Reads analysis/kcdc/*.json only.
"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
K = os.path.join(os.path.dirname(HERE), 'analysis', 'kcdc')
GENS = [('QGSJET-II-04', '', 'o', '#1b6ca8'), ('EPOS-LHC', '_epos', 's', '#d1495b'), ('SIBYLL-2.3d', '_sibyll', '^', '#3a7d44')]
DEC = ['14-15', '15-16', '16-17', '17-18']
plt.rcParams.update({'font.size': 9, 'font.family': 'sans-serif', 'axes.linewidth': 0.6})

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
for i, (g, s, mk, c) in enumerate(GENS):
    R = json.load(open(os.path.join(K, 'kcdc_energy_results%s.json' % s)))
    x = [14.5 + j + (i - 1) * 0.09 for j in range(len(DEC))]
    ex = [R[d]['gb_excess_pct'] for d in DEC]
    lo = [R[d]['gb_excess_pct'] - R[d]['gb_excess_ci'][0] for d in DEC]
    hi = [R[d]['gb_excess_ci'][1] - R[d]['gb_excess_pct'] for d in DEC]
    n3 = 3                                    # top decade is data-limited and off this scale
    ax[0].errorbar(x[:n3], ex[:n3], yerr=[lo[:n3], hi[:n3]], fmt=mk, color=c, ms=4, lw=0.8, capsize=2, label=g)
    ax[0].annotate('', xy=(x[3], 5.6), xytext=(x[3], 4.2), arrowprops=dict(arrowstyle='->', color=c, lw=0.9))
    ax[1].plot([14.5 + j for j in range(len(DEC))], [R[d]['oracle_share_of_variance_pct'] for d in DEC],
               marker=mk, color=c, ms=4, lw=0.8, label=g)
ax[0].axhline(0, color='0.5', lw=0.6, ls='--')
ax[0].set_ylim(-3, 6)
ax[0].text(17.5, 3.4, 'off scale\n(data-limited)', ha='center', fontsize=6.5, color='0.3')
ax[0].set_xlabel(r'$\log_{10}(E/\mathrm{eV})$, true')
ax[0].set_ylabel('boosting above floor estimate (%)')
ax[0].set_xticks([14, 15, 16, 17, 18])
ax[0].legend(frameon=False, fontsize=7, loc='upper left')
ax[1].set_xlabel(r'$\log_{10}(E/\mathrm{eV})$, true')
ax[1].set_ylabel('variance lost to reconstruction (%)')
ax[1].set_xticks([14, 15, 16, 17, 18]); ax[1].set_ylim(0, 100)
for a in ax:
    a.set_xlim(14, 18)
    a.axvspan(17, 18, color='0.92', zorder=0)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig_e1_decades.pdf'))
print('  figures/fig_e1_decades.pdf')

mp = os.path.join(K, 'kcdc_measured_results.json')
if os.path.exists(mp):
    MR = json.load(open(mp))
    fig, a = plt.subplots(figsize=(3.5, 2.7))
    for i, (g, s, mk, c) in enumerate(GENS):
        rows = MR['muon_scale']['models'][g]
        a.plot([0.5 * sum(r['ne']) for r in rows], [r['z_data'] for r in rows], marker=mk, color=c, ms=4, lw=0.8, label=g)
    a.axhline(0, color='0.5', lw=0.6, ls='--'); a.axhline(1, color='0.5', lw=0.6, ls='--')
    a.set_xlabel(r'$\log_{10} N_e$'); a.set_ylabel(r'$\langle\log_{10}N_\mu\rangle$: proton $\to$ iron')
    a.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_e2_measured.pdf'))
    print('  figures/fig_e2_measured.pdf')

# =========================================================================== further figures
import numpy as np
from matplotlib.colors import Normalize
FI = os.path.join(K, 'kcdc_figure_inputs.json')
COL = {g: c for g, _, _, c in GENS}; MK = {g: m for g, _, m, _ in GENS}

# ---- fig_e0_events: two showers on the array; fig_e0_lateral: mean lateral profiles
if os.path.exists(FI):
    J = json.load(open(FI)); ev = J['events']
    sx, sy = np.array(ev['station_x']), np.array(ev['station_y'])
    fig, axs = plt.subplots(2, 2, figsize=(5.6, 5.6), gridspec_kw=dict(hspace=0.62, wspace=0.55))
    for r_, nm in enumerate(('proton', 'iron')):
        E_ = ev[nm]
        for c_, (key, lab, cmap) in enumerate((('e', r'e/$\gamma$ deposit', 'viridis'), ('mu', 'muon deposit', 'magma'))):
            a = axs[r_, c_]
            v = np.log10(1 + np.maximum(np.array(E_[key]), 0)); hit = v > 0
            a.scatter(sx[~hit], sy[~hit], s=6, c='0.85', marker='s', lw=0)
            sc = a.scatter(sx[hit], sy[hit], s=13, c=v[hit], cmap=cmap, marker='s', lw=0,
                           norm=Normalize(0, np.percentile(v[hit], 99) if hit.any() else 1))
            a.plot(*E_['core'], marker='+', color='red', ms=10, mew=1.5)
            a.set_aspect('equal'); a.set_xticks([-100, 0, 100]); a.set_yticks([-100, 0, 100]); a.tick_params(labelsize=7)
            a.set_xlabel('x (m)', fontsize=8); a.set_ylabel('y (m)' if c_ == 0 else '', fontsize=8)
            cb = fig.colorbar(sc, ax=a, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=6)
            cb.set_label(r'$\lg(1+$deposit$)$', fontsize=6.5)
            if c_ == 0:
                a.set_title(r'%s: %s' % (nm, lab), fontsize=8)
            else:
                a.set_title(lab, fontsize=8)
        axs[r_, 0].text(0.0, -0.36, r'%s, $\lg(E/\mathrm{eV})=%.2f$, $\theta=%.0f^\circ$; reconstructed $\lg N_e=%.2f$, $\lg N_\mu=%.2f$ (true %.2f, %.2f)'
                        % (nm, E_['lgE_eV'], E_['zenith'], E_['Ne'], E_['Nmu'], E_['TrNe'], E_['TrNm']),
                        transform=axs[r_, 0].transAxes, fontsize=6.5)
    fig.savefig(os.path.join(HERE, 'fig_e0_events.pdf'), bbox_inches='tight')
    print('  figures/fig_e0_events.pdf')

    L = J['lateral']; rb = np.array(L['r_edges']); rm = 0.5 * (rb[1:] + rb[:-1])
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for a, (key, lab) in zip(axs, (('e', r'e/$\gamma$'), ('mu', 'muon'))):
        for g in L['models']:
            for nm, ls in (('proton', '-'), ('iron', '--')):
                a.plot(rm, L['models'][g][nm][key], ls=ls, marker=MK[g], ms=2.5, lw=0.8, color=COL[g],
                       label='%s, %s' % (g, nm))
        a.set_xlabel('distance to reconstructed core (m)'); a.set_ylabel(r'mean $\lg(1+$%s deposit$)$' % lab)
    axs[0].legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_e0_lateral.pdf')); print('  figures/fig_e0_lateral.pdf')

# ---- fig_e3_resolution: estimators against energy, and bias by primary
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
xs = np.array([14.5, 15.5, 16.5])
for g, s, mk, c in GENS:
    R = json.load(open(os.path.join(K, 'kcdc_energy_results%s.json' % s)))
    dd = DEC[:3]
    ax[0].plot(xs, [R[d]['floor'] for d in dd], color=c, lw=3, alpha=0.25)
    ax[0].plot(xs, [R[d]['sigma']['GradBoost'] for d in dd], marker=mk, color=c, lw=0.9, ms=4)
    ax[0].plot(xs, [R[d]['kcdc_estimate']['calibrated_sigma'] for d in dd], marker=mk, color=c, lw=0.9, ms=4, ls='--', mfc='white')
    ax[0].plot(xs, [R[d]['oracle'] for d in dd], marker=mk, color=c, lw=0.9, ms=4, ls=':')
from matplotlib.lines import Line2D
ax[0].legend(handles=[Line2D([], [], color='0.3', lw=3, alpha=0.3, label='floor estimate'),
                      Line2D([], [], color='0.3', lw=0.9, marker='o', ms=4, label='boosting, nine features'),
                      Line2D([], [], color='0.3', lw=0.9, ls='--', marker='o', mfc='white', ms=4, label='KCDC estimate, calibrated'),
                      Line2D([], [], color='0.3', lw=0.9, ls=':', marker='o', ms=4, label='oracle (true sizes)')] +
             [Line2D([], [], color=c, marker=mk, lw=0, label=g) for g, _, mk, c in GENS], fontsize=6, frameon=False)
ax[0].set_xlabel(r'$\lg(E/\mathrm{eV})$, true'); ax[0].set_ylabel(r'$\sigma$ of $\lg E$'); ax[0].set_xticks([14, 15, 16, 17]); ax[0].set_xlim(14, 17)
lnA = {14: 0.0, 402: np.log(4), 1206: np.log(12), 2814: np.log(28), 5626: np.log(56)}
for g, s, mk, c in GENS:
    R = json.load(open(os.path.join(K, 'kcdc_energy_results%s.json' % s)))['15-16']
    gb = R['by_primary']['GradBoost']['mean_bias']; kc = R['kcdc_estimate']['by_primary_calibrated']['mean_bias']
    ks = sorted(gb, key=lambda k: int(k))
    ax[1].plot([lnA[int(k)] for k in ks], [gb[k] for k in ks], marker=mk, color=c, lw=0.9, ms=4)
    ax[1].plot([lnA[int(k)] for k in ks], [kc[k] for k in ks], marker=mk, color=c, lw=0.9, ms=4, ls='--', mfc='white')
ax[1].axhline(0, color='0.5', lw=0.6)
ax[1].set_xticks([lnA[k] for k in (14, 402, 1206, 2814, 5626)]); ax[1].set_xticklabels(['p', 'He', 'C', 'Si', 'Fe'])
ax[1].set_ylabel(r'mean bias in $\lg E$'); ax[1].set_xlabel(r'primary ($15\leq\lg(E/\mathrm{eV})<16$)')
ax[1].legend(handles=[Line2D([], [], color='0.3', lw=0.9, marker='o', ms=4, label='boosting'),
                      Line2D([], [], color='0.3', lw=0.9, ls='--', marker='o', mfc='white', ms=4, label='KCDC, calibrated')], fontsize=6, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_e3_resolution.pdf')); print('  figures/fig_e3_resolution.pdf')

# ---- fig_e4_stations: share of the variance gap to the oracle closed, by representation
def vsh(g9, o, x):
    return 100 * (g9 ** 2 - x ** 2) / (g9 ** 2 - o ** 2)
fig, ax = plt.subplots(figsize=(7.0, 2.6))
labs, xpos, k = [], [], 0
reps = [('station features', 'eng', '#9bb7d4'), ('CNN on image', 'cnn', '#4a7fb5'), ('graph network', 'gnn', '#173f6b')]
for wsuf, wl in (('', r'$\lg(E/\mathrm{GeV})\in[5,6]$'), ('_e67', r'$[6,7]$')):
    for g, s, mk, c in GENS:
        fp = os.path.join(K, 'kcdc_floor_results%s%s.json' % (s, wsuf)); cp = os.path.join(K, 'kcdc_cnn_results%s%s.json' % (s, wsuf))
        if not (os.path.exists(fp) and os.path.exists(cp)):
            continue
        Fr, Cr = json.load(open(fp)), json.load(open(cp)); g9, o = Fr['oracle']['sigma_nine'], Fr['oracle']['sigma_true_sizes']
        vals = {'eng': vsh(g9, o, Cr['gb_engineered']), 'cnn': vsh(g9, o, Cr['cnn']['residual on GB(nine)']['sigma_ensemble'])}
        gp = os.path.join(K, 'kcdc_cnn_gnn_results%s%s.json' % (s, wsuf))
        if os.path.exists(gp):
            vals['gnn'] = vsh(g9, o, json.load(open(gp))['cnn']['residual on GB(nine)']['sigma_ensemble'])
        for j, (lab_, key, cc) in enumerate(reps):
            if key in vals:
                ax.bar(k + (j - 1) * 0.26, vals[key], 0.25, color=cc, label=lab_ if k == 0 else None)
        labs.append('%s\n%s' % (g.split('-')[0], wl if g == GENS[0][0] else '')); xpos.append(k); k += 1
    k += 0.5
ax.set_xticks(xpos); ax.set_xticklabels(labs, fontsize=7)
ax.set_ylabel('variance gap to oracle closed (%)'); ax.legend(fontsize=7, frameon=False, ncol=3, loc='upper right')
ax.set_ylim(0, 15)
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_e4_stations.pdf')); print('  figures/fig_e4_stations.pdf')

# ---- fig_e5_measured_hist: log10 Nmu in data against simulation per decade
if os.path.exists(FI):
    Mh = json.load(open(FI))['measured']; b = np.array(Mh['bins']); ctr = 0.5 * (b[1:] + b[:-1])
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
    for a, (d, row) in zip(ax, Mh['decades'].items()):
        a.step(ctr, row['data'], where='mid', color='k', lw=1.1, label='measured')
        for nm, cc in (('proton', '#1b6ca8'), ('mixture', '0.45'), ('iron', '#c0392b')):
            a.plot(ctr, row['EPOS-LHC'][nm], color=cc, lw=0.9, label='EPOS-LHC, %s' % nm)
            a.plot(ctr, row['QGSJET-II-04'][nm], color=cc, lw=0.6, ls=':')
        a.set_yscale('log'); a.set_ylim(1e-4, 0.2)
        a.set_xlabel(r'$\lg N_\mu$'); a.set_title(r'$\lg(E_{\rm rec}/\mathrm{eV})\in[%s,%s)$' % tuple(d.split('-')), fontsize=8)
        nz = [i for i, v in enumerate(row['data']) if v > 1e-4]
        a.set_xlim(ctr[nz[0]] - 0.1, ctr[nz[-1]] + 0.1)
    ax[0].set_ylabel('fraction per bin'); ax[0].legend(fontsize=5.5, frameon=False, loc='lower center')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_e5_measured_hist.pdf')); print('  figures/fig_e5_measured_hist.pdf')

# ---- fig_e6_floorsens: excess over the floor estimate, reference setting and all settings
fig, a = plt.subplots(figsize=(7.0, 2.4))
k, ticks, tl = 0, [], []
for key, ds, lab_ in (('st', '', r'$[5,6]$ GeV'), ('14-15', '_d14-15', '14-15'), ('15-16', '_d15-16', '15-16'), ('16-17', '_d16-17', '16-17')):
    for g, s, mk, c in GENS:
        fp = os.path.join(K, 'kcdc_floor_sens%s%s.json' % (s, ds))
        if not os.path.exists(fp):
            continue
        S = json.load(open(fp)); lo_, hi_ = S['gb_excess_range_pct']
        a.plot([k, k], [lo_, hi_], color=c, lw=6, alpha=0.25, solid_capstyle='butt')
        a.errorbar(k, S['gb_excess_pct'], yerr=[[S['gb_excess_pct'] - S['gb_excess_ci_shower'][0]], [S['gb_excess_ci_shower'][1] - S['gb_excess_pct']]],
                   fmt=mk, color=c, ms=4, lw=0.9, capsize=2)
        ticks.append(k); tl.append(lab_ if g == GENS[1][0] else ''); k += 1
    k += 0.7
a.axhline(0, color='0.5', lw=0.6, ls='--'); a.set_xticks(ticks); a.set_xticklabels(tl, fontsize=7)
a.set_ylabel('boosting above floor est. (%)'); a.set_xlabel(r'sample ($\lg(E/\mathrm{eV})$ decades unless marked)')
a.legend(handles=[Line2D([], [], color=c, marker=mk, lw=0.9, label=g) for g, _, mk, c in GENS] +
         [Line2D([], [], color='0.5', lw=6, alpha=0.3, label='range over estimator settings')], fontsize=6, frameon=False, ncol=4, loc='lower left')
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_e6_floorsens.pdf')); print('  figures/fig_e6_floorsens.pdf')
