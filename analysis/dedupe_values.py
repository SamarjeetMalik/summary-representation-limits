"""Keep only the last definition of each macro in tables/values.tex.

make_tables_extra.py and make_tables_render.py APPEND to values.tex, so running
either twice without first rerunning make_tables.py (which writes it fresh)
leaves duplicate \\newcommand lines and LaTeX fails with 'already defined'.

Run this after the generators, or simply run them in order:
    make_tables.py -> make_tables_extra.py -> make_tables_render.py
"""
import os
import re

TAB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tables')
path = os.path.join(TAB, 'values.tex')

lines = open(path, encoding='utf-8').read().split('\n')
pat = re.compile(r'\\newcommand\{\\([A-Za-z]+)\}')

seen = {}
other = []
n_defs = 0
for ln in lines:
    m = pat.match(ln)
    if m:
        n_defs += 1
        seen[m.group(1)] = ln          # later definition wins
    elif ln.strip():
        other.append(ln)

body = other + [seen[k] for k in seen]
open(path, 'w', encoding='utf-8').write('\n'.join(body) + '\n')
print('values.tex: %d unique macros, removed %d duplicate definition(s)'
      % (len(seen), n_defs - len(seen)))
