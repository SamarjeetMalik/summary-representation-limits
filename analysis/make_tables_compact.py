"""Compact table variants for a short version.

Same numbers, same source (results.json), tighter footprint. Run after run_all.py,
run_extra.py and run_eventci.py.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(os.path.dirname(HERE), 'tables')
R = json.load(open(os.path.join(HERE, 'results.json')))
E, M, D, CS = R['estimators'], R['muon'], R['degeneracy'], R['cut_sweep']


def w(name, body):
    open(os.path.join(TAB, name), 'w', encoding='utf-8').write(body)
    print('  tables/%s' % name)


# --- ceiling + muon ablation merged into one small table -------------------
rows = []
for lbl, key in (('Ridge (10 coef.)', 'Ridge'), ('Random forest', 'RandomForest'),
                 ('Gradient boosting', 'GradBoost'), ('MLP $64{\\times}64$', 'MLP')):
    rows.append(r'%s & $%.4f$ \\' % (lbl, E[key]['sigma']))
rows.append(r'\midrule')
rows.append(r'NKG (non-learned) & $%.4f$ \\' % E['NKG']['sigma'])
rows.append(r'\bf CNN/ViT, rendered & \bf $%.4f$ \\' % E['CNN/ViT']['sigma'])

mrows = [r'nine features & $%.4f$ & --- \\' % M['nine_feature']['sigma'],
         r'$+$ 16 counts & $%.4f$ & $%+.2f\%%$ \\'
         % (M['plus_raw16']['sigma'], M['plus_raw16']['delta_pct']),
         r'$+$ 16 shape & $%.4f$ & $%+.2f\%%$ \\'
         % (M['plus_shape16']['sigma'], M['plus_shape16']['delta_pct'])]

w('tab_compact_floor.tex', r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{4pt}
\begin{minipage}[t]{.49\textwidth}\centering
\caption{Every estimator on the \emph{same} %s events. Seed-to-seed s.d.\ is
below $10^{-4}$ for every learned row (MLP $4\times10^{-4}$), so it is omitted; quoted
uncertainties elsewhere are bootstrap over events.}
\label{tab:floor}
\begin{tabular}{lc}
\toprule
Estimator & $\sigma(\lgE)$ \\
\midrule
%s
\bottomrule
\end{tabular}
\end{minipage}\hfill
\begin{minipage}[t]{.49\textwidth}\centering
\caption{Restoring the discarded muon spatial pattern. The three rows are
identical to four decimal places; that agreement is the result, not a transcription
error.}
\label{tab:muon}
\begin{tabular}{lcc}
\toprule
Features & $\sigma(\lgE)$ & $\Delta$ \\
\midrule
%s
\bottomrule
\end{tabular}
\end{minipage}
\end{table}
""" % ('{:,}'.format(R['n_matched']).replace(',', '{,}'), '\n'.join(rows), '\n'.join(mrows)))

# --- degeneracy + cut sweep merged ----------------------------------------
drows = [r'NKG analytic & $%.3f$ & $%.4f$ \\' % (D['NKG']['r2'], D['NKG']['resid_sd']),
         r'CNN/ViT (rendered) & $%.3f$ & $%.4f$ \\' % (D['CNN/ViT']['r2'], D['CNN/ViT']['resid_sd']),
         r'true energy & $%.3f$ & $%.4f$ \\' % (D['true_energy']['r2'], D['true_energy']['resid_sd'])]

crows = []
for k, lbl in (('no cut', 'none'), ('0.90-1.15 (standard)', r'$0.90\!-\!1.15$ (std.)'),
               ('0.90-1.40', r'$0.90\!-\!1.40$'), ('s >= 1.15 only', r'$s\ge1.15$')):
    v = CS[k]
    em = r'\bf ' if 'standard' in k else ''
    crows.append(r'%s%s & %s$%.1f\%%$ & %s$%.4f$ \\' % (em, lbl, em, v['kept'], em, v['sigma']))

w('tab_compact_diag.tex', r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{4pt}
\caption{Shower-age selection sweep; $\sigma$ is $\sigma(\lgE)$. Gradient boosting,
%d seeds, on the matched population; bootstrap intervals over events are in the text.}
\label{tab:cutsweep}
\begin{tabular}{lcc}
\toprule
Selection & events kept & $\sigma$ \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (CS['no cut']['seeds'], '\n'.join(crows)))

print('compact tables written')
