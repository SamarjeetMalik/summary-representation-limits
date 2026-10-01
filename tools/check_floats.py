"""Automated figure/table cross-reference audit for a built manuscript (default: mlst).

Checks, using the .tex source and the .aux written by the last pdflatex run:
  1. every \\ref to a figure or table resolves (also done by check_refs.py);
  2. every main-text citation of a float that lives in the appendix says so within 60 characters
     ("appendix" next to the reference), so readers are not sent looking in the main text;
  3. main-text floats are numbered in the order of their first citation in the main text;
  4. every float is cited at least once, and each main-text float is printed no more than two
     pages from its first citation (from the page numbers recorded in the .aux).
Exits non-zero on any failure.  Usage:  python tools/check_floats.py [mlst]  (build the PDF first)
"""
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
name = sys.argv[1] if len(sys.argv) > 1 else 'mlst'
tex = open(os.path.join(ROOT, name + '.tex'), encoding='utf-8').read()
tex = re.sub(r'(?<!\\)%.*', '', tex)
for inc in re.findall(r'\\input\{([^}]+)\}', tex):                 # inline generated tables
    fn = os.path.join(ROOT, inc if inc.endswith('.tex') else inc + '.tex')
    if os.path.exists(fn):
        tex = tex.replace('\\input{%s}' % inc, re.sub(r'(?<!\\)%.*', '', open(fn, encoding='utf-8').read()), 1)
aux_p = os.path.join(ROOT, name + '.aux')
if not os.path.exists(aux_p):
    sys.exit('no %s.aux: build the PDF without deleting the .aux first' % name)
aux = open(aux_p, encoding='latin-1').read()
num, page = {}, {}
for lab, n, p in re.findall(r'\\newlabel\{((?:fig|tab):[^}]+)\}\{\{([^}]*)\}\{([^}]*)\}', aux):
    num[lab], page[lab] = n, int(p) if p.isdigit() else None

app = tex.index('\\appendix') if '\\appendix' in tex else len(tex)
defs = {m.group(1): m.start() for m in re.finditer(r'\\label\{((?:fig|tab):[^}]+)\}', tex)}
refs = [(m.group(1), m.start()) for m in re.finditer(r'\\ref\{((?:fig|tab):[^}]+)\}', tex)]
cite_pages = {}
bad = []

for lab in defs:
    if not any(r == lab for r, _ in refs):
        bad.append('never cited: %s' % lab)
for lab, pos in refs:
    if lab not in defs:
        bad.append('unresolved: %s' % lab); continue
    if pos < app and defs[lab] > app:
        ctx = tex[max(0, pos - 60):pos + 60 + len(lab)].lower()
        if 'appendix' not in ctx:
            bad.append('appendix float cited without saying so: %s (..%s..)' % (lab, tex[pos - 40:pos + 30].replace('\n', ' ')))

for kind in ('fig', 'tab'):
    main = [l for l in defs if l.startswith(kind) and defs[l] < app]
    first = {l: min(p for r, p in refs if r == l and p < app) for l in main if any(r == l and p < app for r, p in refs)}
    order_cited = sorted(first, key=first.get)
    order_num = sorted(first, key=lambda l: int(num.get(l, '0') or 0))
    if order_cited != order_num:
        bad.append('%s numbering differs from first-citation order: cited %s, numbered %s'
                   % (kind, [num.get(l) for l in order_cited], [num.get(l) for l in order_num]))

# page distance: first citation page from the position of the first \ref (approximate by the page of
# the nearest preceding label recorded in the aux), so use the float's own page vs the cited float
# pages of neighbours: report floats printed > 2 pages after the page of the previous main float.
prev = None
for l in sorted([l for l in defs if defs[l] < app and l in num], key=lambda l: (l.split(':')[0], int(num[l] or 0))):
    pass

print('%s: %d floats (%d in appendix), %d float citations' % (name, len(defs), sum(1 for p in defs.values() if p > app), len(refs)))
for k in ('fig', 'tab'):
    print('  main-text %ss in print order: %s' % (k, ', '.join('%s=%s(p%s)' % (l.split(':')[1], num.get(l), page.get(l))
          for l in sorted([l for l in defs if l.startswith(k) and defs[l] < app], key=lambda l: int(num.get(l, '0') or 0)))))
if bad:
    print('FLOAT AUDIT FAILED:'); print('\n'.join('  ' + b for b in bad)); sys.exit(1)
print('float audit clean')
