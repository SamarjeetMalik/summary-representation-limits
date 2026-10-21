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
