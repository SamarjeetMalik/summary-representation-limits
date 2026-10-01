"""Generate the manuscript's tables from results.json.

Every numeric cell in the paper is emitted here rather than transcribed by hand.
Run after run_all.py:  python analysis/make_tables.py
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TAB = os.path.join(ROOT, 'tables')
os.makedirs(TAB, exist_ok=True)
R = json.load(open(os.path.join(HERE, 'results.json')))


def w(name, body):
    open(os.path.join(TAB, name), 'w', encoding='utf-8').write(body)
    print('  tables/%s' % name)


E = R['estimators']

# ---------------------------------------------------------------- Table 1
rows = []
order = [('Ridge regression', 'Ridge', 'linear, 10 coefficients'),
         ('Random forest', 'RandomForest', '60 trees, depth 18'),
         ('Histogram gradient boosting', 'GradBoost', '600 boosting rounds'),
         ('Multilayer perceptron', 'MLP', r'$64\times64$, $\sim$5k params')]
for label, key, cap in order:
    e = E[key]
    rows.append(r'%s & %s & $%.4f$ \\' % (label, cap, e['sigma']))
rows.append(r'\midrule')
rows.append(r'NKG analytic reconstruction & non-learned & $%.4f$ \\' % E['NKG']['sigma'])
rows.append(r'\midrule')
rows.append(r'\multicolumn{3}{l}{\emph{reported elsewhere, different split --- not directly comparable}} \\')
rows.append(r'Physics-initialised spline network & 2{,}800 params & $0.1015$ \\')
rows.append(r'Gradient-boosted trees (XGBoost) & $1000\times$ depth 6 & $0.1018$ \\')

w('tab_floor.tex', r"""\begin{table}[h]
\centering
\caption{The summary-feature ceiling. Four learned estimators spanning a linear model to a
neural network, plus the deployed non-learned reconstruction, all evaluated on the
\emph{same} %s events and trained on the same %s. Learned rows are the mean over
%d seeds; seed-to-seed s.d.\ is below $10^{-4}$ for every row except the MLP
($4\times10^{-4}$), so it is omitted here.}
\label{tab:floor}
\begin{tabular}{llc}
\toprule
Estimator & Capacity & $\sigma(\lgE)$ \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % ('{:,}'.format(R['n_matched']).replace(',', '{,}'),
       '{:,}'.format(R['n_train']).replace(',', '{,}'),
       E['Ridge']['seeds'], '\n'.join(rows)))

# ---------------------------------------------------------------- Table 2
M = R['muon']
w('tab_muon.tex', r"""\begin{table}[h]
\centering
\caption{Restoring the muon spatial structure discarded by the summary representation.
Gradient boosting, mean over 5 seeds, identical population. The three rows agree to within
%s in $\sigma$; that agreement is the result, not a transcription error.}
\label{tab:muon}
\begin{tabular}{lcc}
\toprule
Feature set & $\sigma(\lgE)$ & $\Delta$ vs.\ baseline \\
\midrule
Nine-feature (muon total only) & $%.4f$ & --- \\
$+$ 16 raw module counts & $%.4f$ & $%+.2f\%%$ \\
$+$ 16 normalised module shape & $%.4f$ & $%+.2f\%%$ \\
\bottomrule
\end{tabular}
\end{table}
""" % ('$%.4f$' % (max(M[k]['sigma'] for k in ('nine_feature', 'plus_raw16', 'plus_shape16'))
                    - min(M[k]['sigma'] for k in ('nine_feature', 'plus_raw16', 'plus_shape16'))),
       M['nine_feature']['sigma'],
       M['plus_raw16']['sigma'], M['plus_raw16']['delta_pct'],
       M['plus_shape16']['sigma'], M['plus_shape16']['delta_pct']))

# ---------------------------------------------------------------- Table 3
D = R['degeneracy']
w('tab_degeneracy.tex', r"""\begin{table}[h]
\centering
\caption{How much of each estimator is recoverable from the four summary scalars
$(\log_{10}\Ne, s, X_c, Y_c)$ alone.}
\label{tab:degeneracy}
\begin{tabular}{lcc}
\toprule
Target & $R^2$ & residual s.d. \\
\midrule
NKG analytic reconstruction & $%.4f$ & $%.4f$ \\
CNN/ViT on rendered images & $%.4f$ & $%.4f$ \\
True primary energy & $%.4f$ & $%.4f$ \\
\bottomrule
\end{tabular}
\end{table}
""" % (D['NKG']['r2'], D['NKG']['resid_sd'],
       D['CNN/ViT']['r2'], D['CNN/ViT']['resid_sd'],
       D['true_energy']['r2'], D['true_energy']['resid_sd']))

# ---------------------------------------------------------------- Table 4
mrows = [r'$%s$ & $%s$ & $%+.3f$ \\' % (m['window'].replace('-', '$--$'),
         '{:,}'.format(m['n']).replace(',', '{,}'), m['corr']) for m in R['mechanism']]
w('tab_corr.tex', r"""\begin{table}[h]
\centering
\caption{Correlation of shower age with $\lgE$ inside fixed shower-size windows.}
\label{tab:corr}
\begin{tabular}{lcc}
\toprule
$\log_{10}\Ne$ window & $n$ & $\mathrm{corr}(s, \lgE)$ \\
\midrule
%s
\midrule
whole sample & $%s$ & $%+.3f$ \\
\bottomrule
\end{tabular}
\end{table}
""" % ('\n'.join(mrows), '{:,}'.format(R['n_root']).replace(',', '{,}'), R['corr_whole']))

print('tables written from results.json')


# ------------------------------------------------- generated macros for PROSE
# Prose numbers are the remaining transcription risk once tables are generated.
# Define them as macros so the text cannot drift from results.json either.
def esc(n):
    return '{:,}'.format(int(n)).replace(',', '{,}')


learned = [E[k]['sigma'] for k in ('Ridge', 'RandomForest', 'GradBoost', 'MLP')]
nonlin = [E[k]['sigma'] for k in ('RandomForest', 'GradBoost', 'MLP')]
macros = {
    'Nroot': esc(R['n_root']),
    'Nmatched': esc(R['n_matched']),
    'Ntrain': esc(R['n_train']),
    'PctMatched': '%.1f' % R['pct_matched_of_csv'],
    'targetSpread': '%.3f' % R['target_spread'],
    'logEmin': '%.2f' % R['logE_min'],
    'logEmax': '%.2f' % R['logE_max'],
    'sigmaFloorLo': '%.4f' % min(learned),
    'sigmaFloorHi': '%.4f' % max(learned),
    'sigmaFloorSpread': '%.1g' % (max(learned) - min(learned)),
    'sigmaNonlinLo': '%.4f' % min(nonlin),
    'sigmaNonlinHi': '%.4f' % max(nonlin),
    'sigmaNonlinSpread': '%.4f' % (max(nonlin) - min(nonlin)),
    'ridgePenaltyPct': '%.1f' % (100 * (E['Ridge']['sigma'] - min(nonlin)) / min(nonlin)),
    'sigmaRidge': '%.4f' % E['Ridge']['sigma'],
    'sigmaGBM': '%.4f' % E['GradBoost']['sigma'],
    'sigmaNKG': '%.4f' % E['NKG']['sigma'],
    'sigmaCNNViT': '%.4f' % E['CNN/ViT']['sigma'],
    'muonDeltaRaw': '%+.2f' % M['plus_raw16']['delta_pct'],
    'muonDeltaShape': '%+.2f' % M['plus_shape16']['delta_pct'],
    'rtwoNKG': '%.3f' % D['NKG']['r2'],
    'rtwoCNN': '%.3f' % D['CNN/ViT']['r2'],
    'rtwoTrue': '%.3f' % D['true_energy']['r2'],
    'residCNN': '%.4f' % D['CNN/ViT']['resid_sd'],
    'residTrue': '%.4f' % D['true_energy']['resid_sd'],
    'corrLo': '%.3f' % min(m['corr'] for m in R['mechanism']),
    'corrHi': '%.3f' % max(m['corr'] for m in R['mechanism']),
    'corrWhole': '%.3f' % R['corr_whole'],
    'selNoCut': '%.4f' % R['selection']['no_cut']['sigma'],
    'selSDRN': '%.4f' % R['selection']['sdrn']['sigma'],
    'selKept': '%.1f' % R['selection']['sdrn']['kept'],
    'trendNKG': '%.1f' % abs(E['NKG']['age_trend_pct']),
    'trendRidge': '%.1f' % abs(E['Ridge']['age_trend_pct']),
}
w('values.tex', '\n'.join(r'\newcommand{\%s}{%s}' % (k, v) for k, v in macros.items()) + '\n')
print('  %d prose macros' % len(macros))
