"""Table and macros for the rendering experiment and architecture sweep.

Run after run_rendering.py and run_sweep.py, and after make_tables.py /
make_tables_extra.py (which write values.tex; this appends to it).
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(os.path.dirname(HERE), 'tables')
R = json.load(open(os.path.join(HERE, 'results.json')))
RD, SW = R['rendering'], R['sweep']

rrows = [
    (r'A\ \ four scalars', 0.1234, RD['A_scalars4'], ''),
    (r'B\ \ CNN, rendered', 0.1234, RD['B_rendered'], r'\bf '),
    (r'C\ \ CNN, rendered $+$ muons', 0.1063, RD['C_rendered_muon'], r'\bf '),
    (r'D\ \ nine features', 0.1019, RD['D_nine'], ''),
]
rbody = '\n'.join(r'%s%s & %s$%.4f$ & %s$%.4f$ \\' % (em, lbl, em, pred, em, obs)
                  for lbl, pred, obs, em in rrows)

srows = [('Ridge, pixels', 'linear', SW['E_ridge_pixels']),
         ('MLP, pixels', 'fully conn.', SW['F_mlp_pixels']),
         ('CNN, image', 'convolutional', RD['B_rendered']),
         (r'ViT, patches', 'attention', SW['G_vit_patches'])]
sbody = '\n'.join(r'%s & %s & $%.4f$ \\' % (a, b, c) for a, b, c in srows)

body = r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{4pt}
\begin{minipage}[t]{.50\textwidth}\centering
\caption{Reconstructing the failure. Predictions were fixed before running.}
\label{tab:render}
\begin{tabular}{lcc}
\toprule
Input & predicted & observed \\
\midrule
%s
\bottomrule
\end{tabular}
\end{minipage}\hfill
\begin{minipage}[t]{.45\textwidth}\centering
\caption{Architectures on the \emph{same} rendered images; reference $%.4f$.}
\label{tab:sweep}
\begin{tabular}{llc}
\toprule
Model & class & $\sigma$ \\
\midrule
%s
\bottomrule
\end{tabular}
\end{minipage}
\end{table}
""" % (rbody, RD['A_scalars4'], sbody)

open(os.path.join(TAB, 'tab_compact_render.tex'), 'w', encoding='utf-8').write(body)
print('  tables/tab_compact_render.tex')

m = {
    'rendA': '%.4f' % RD['A_scalars4'],
    'rendB': '%.4f' % RD['B_rendered'],
    'rendC': '%.4f' % RD['C_rendered_muon'],
    'rendD': '%.4f' % RD['D_nine'],
    'rendBA': '%+.4f' % (RD['B_rendered'] - RD['A_scalars4']),
    'rendNtrain': '{:,}'.format(RD['n_train']).replace(',', '{,}'),
    'sweepE': '%.4f' % SW['E_ridge_pixels'],
    'sweepF': '%.4f' % SW['F_mlp_pixels'],
    'sweepG': '%.4f' % SW['G_vit_patches'],
    'vitParams': '{:,}'.format(SW['G_vit_params']).replace(',', '{,}'),
    'cnnParams': '{:,}'.format(RD['B_rendered_params']).replace(',', '{,}'),
    'muonGain': '%.4f' % (RD['B_rendered'] - RD['C_rendered_muon']),
}
p = os.path.join(TAB, 'values.tex')
ex = open(p, encoding='utf-8').read()
open(p, 'w', encoding='utf-8').write(
    ex + '\n'.join(r'\newcommand{\%s}{%s}' % (k, v) for k, v in m.items()) + '\n')
print('  +%d rendering macros' % len(m))
