"""Tables and macros for the journal-version sections: cut sweep, per-species,
conformal calibration. Run after run_extra.py.
"""
import json, math, os


def sci(v):
    """LaTeX scientific notation, e.g. 2e-05 -> 2\\times10^{-5} (never a raw '2e-05')."""
    e = int(math.floor(math.log10(v)))
    return r'%d\times10^{%d}' % (round(v / 10 ** e), e)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TAB = os.path.join(ROOT, 'tables')
R = json.load(open(os.path.join(HERE, 'results.json')))


def w(name, body):
    open(os.path.join(TAB, name), 'w', encoding='utf-8').write(body)
    print('  tables/%s' % name)


# ---------------------------------------------------------- cut sweep
CS = R['cut_sweep']
rows = []
for k in ('no cut', '0.90-1.15 (standard)', '0.90-1.25', '0.90-1.40', 's >= 1.15 only'):
    v = CS[k]
    lbl = (k.replace('0.90-1.15 (standard)', r'$0.90 \le s \le 1.15$ (standard)')
            .replace('0.90-1.25', r'$0.90 \le s \le 1.25$')
            .replace('0.90-1.40', r'$0.90 \le s \le 1.40$')
            .replace('s >= 1.15 only', r'$s \ge 1.15$ only')
            .replace('no cut', 'no selection'))
    emph = r'\bf ' if k == '0.90-1.15 (standard)' else ''
    rows.append(r'%s%s & %s$%.1f\%%$ & %s$%.4f$ & $[%.4f, %.4f]$ \\'
                % (emph, lbl, emph, v['kept'], emph, v['sigma'], v['ci_events'][0], v['ci_events'][1]))

w('tab_cutsweep.tex', r"""\begin{table}[htbp]
\centering
\caption{Energy resolution under different shower-age selections. Gradient boosting,
evaluated on the matched population. Intervals are 95\%% bootstrap over events
(%d resamples); across-seed variation is below $10^{-4}$ throughout and is reported
separately in the text. The standard selection discards half the sample and returns a
\emph{worse} resolution than applying no selection at all.}
\label{tab:cutsweep}
\begin{tabular}{lccc}
\toprule
Selection & Events kept & $\sigma(\lgE)$ & 95\%% CI \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (2000, '\n'.join(rows)))

# ---------------------------------------------------------- per species
srows = []
for s in R['species']:
    if s['skipped']:
        srows.append(r'%s & $%s$ & \multicolumn{4}{c}{\emph{too few young showers ($n=%d$)}} \\'
                     % (s['name'].capitalize(), '{:,}'.format(s['n']).replace(',', '{,}'), s['n_low']))
    else:
        srows.append(r'%s & $%s$ & $%.4f$ & $%.4f$ & $%+.1f\%%$ & $%.1f\%%$ \\'
                     % (s['name'].capitalize(), '{:,}'.format(s['n']).replace(',', '{,}'),
                        s['sigma_young'], s['sigma_old'], s['delta_pct'], s['frac_old']))
P = R['pooled_split']
srows.append(r'\midrule')
srows.append(r'\emph{pooled} & --- & $%.4f$ & $%.4f$ & $\mathbf{%+.1f\%%}$ & --- \\'
             % (P['sigma_young'], P['sigma_old'], P['delta_pct']))

w('tab_species.tex', r"""\begin{table}[htbp]
\centering
\caption{Resolution by primary species, split at $s = 1.15$. The effect holds for every
species, but the within-species improvement is roughly half the pooled figure: heavier
primaries are both better reconstructed \emph{and} far more likely to be old, so pooling
inflates the apparent gap. The pooled row is reported as a mixture effect, not as the
size of the underlying phenomenon.}
\label{tab:species}
\begin{tabular}{lccccc}
\toprule
Primary & $n$ & $\sigma$ ($s<1.15$) & $\sigma$ ($s\ge1.15$) & $\Delta$ & fraction old \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % '\n'.join(srows))

# ---------------------------------------------------------- calibration
CAL = R['calibration']
crows = []
for b90, b68 in zip(CAL['90']['bins'], CAL['68']['bins']):
    crows.append(r'$%s$ & $%s$ & $%.1f\%%$ & $%.1f\%%$ & $%.1f\%%$ & $%.1f\%%$ \\'
                 % (b90['bin'].replace('-', '$--$'),
                    '{:,}'.format(b90['n']).replace(',', '{,}'),
                    b68['raw'], b68['cqr'], b90['raw'], b90['cqr']))

w('tab_calibration.tex', r"""\begin{table}[htbp]
\centering
\caption{Interval coverage per age stratum, before and after conformalisation.
Raw quantile regression under-covers uniformly; conformalised quantile regression (CQR)
restores nominal coverage, and does so \emph{without} degrading in the old-shower strata
that the standard selection discards. Nominal levels are 68\%% and 90\%%.}
\label{tab:calibration}
\begin{tabular}{lccccc}
\toprule
& & \multicolumn{2}{c}{68\%% interval} & \multicolumn{2}{c}{90\%% interval} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
Age bin & $n$ & raw & CQR & raw & CQR \\
\midrule
%s
\midrule
\emph{overall} & --- & $%.1f\%%$ & $%.1f\%%$ & $%.1f\%%$ & $%.1f\%%$ \\
\bottomrule
\end{tabular}
\end{table}
""" % ('\n'.join(crows), CAL['68']['overall_raw'], CAL['68']['overall_cqr'],
       CAL['90']['overall_raw'], CAL['90']['overall_cqr']))

# ---------------------------------------------------------- macros
m = {
    'cutStdKept': '%.1f' % CS['0.90-1.15 (standard)']['kept'],
    'cutStdSigma': '%.4f' % CS['0.90-1.15 (standard)']['sigma'],
    'cutNoneSigma': '%.4f' % CS['no cut']['sigma'],
    'cutWideKept': '%.1f' % CS['0.90-1.40']['kept'],
    'cutWideSigma': '%.4f' % CS['0.90-1.40']['sigma'],
    'cutOldSigma': '%.4f' % CS['s >= 1.15 only']['sigma'],
    'pooledDelta': '%.1f' % abs(P['delta_pct']),
    'withinLo': '%.1f' % abs(R['within_species_range'][0]),
    'withinHi': '%.1f' % abs(R['within_species_range'][1]),
    'covRawNinety': '%.1f' % CAL['90']['overall_raw'],
    'covCqrNinety': '%.1f' % CAL['90']['overall_cqr'],
    'covRawSixtyEight': '%.1f' % CAL['68']['overall_raw'],
    'covCqrSixtyEight': '%.1f' % CAL['68']['overall_cqr'],
    'conformalQ': '%.4f' % CAL['90']['Q'],
    'cutPenalty': '%.4f' % R['cut_penalty']['delta'],
    'cutPenaltyLo': '%.4f' % R['cut_penalty']['ci'][0],
    'cutPenaltyHi': '%.4f' % R['cut_penalty']['ci'][1],
    'seedSd': sci(max(v['seed_sd'] for v in CS.values())),
}
existing = open(os.path.join(TAB, 'values.tex'), encoding='utf-8').read()
extra = '\n'.join(r'\newcommand{\%s}{%s}' % (k, v) for k, v in m.items()) + '\n'
open(os.path.join(TAB, 'values.tex'), 'w', encoding='utf-8').write(existing + extra)
print('  +%d journal macros appended to tables/values.tex' % len(m))
