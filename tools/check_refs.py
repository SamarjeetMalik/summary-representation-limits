"""Check every \\ref resolves to a \\label, and every citation matches a bibitem.

LaTeX only warns about an undefined reference and prints '??' in the PDF, which
is easy to miss after moving a figure or deleting a table. This fails loudly.

It also fails on a \\bibitem that is never cited, or a cite key with no \\bibitem.
LaTeX prints an uncited bibitem without complaint: an earlier draft shipped
nine references of which eight were never cited, and a reviewer read that as
"related work is largely missing".

Usage:  python tools/check_refs.py                  (manuscript, the manuscript in play)
        python tools/check_refs.py manuscript   (named manuscripts)
"""
import glob
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
papers = sys.argv[1:] or ['manuscript']   # the manuscript in play; name others explicitly

tables = ''
for f in glob.glob(os.path.join(ROOT, 'tables', '*.tex')):
    tables += io.open(f, encoding='utf-8').read()

bad_any = False
for p in papers:
    src = io.open(os.path.join(ROOT, p + '.tex'), encoding='utf-8').read()
    included = ''.join(
        io.open(os.path.join(ROOT, t + ('' if t.endswith('.tex') else '.tex')),
                encoding='utf-8').read()
        for t in re.findall(r'\\(?:wide)?input\{([^}]+)\}', src)
        if os.path.exists(os.path.join(ROOT, t + ('' if t.endswith('.tex') else '.tex'))))
    text = src + included
    used = set(re.findall(r'\\ref\{([^}]+)\}', text))
    defined = set(re.findall(r'\\label\{([^}]+)\}', text))
    broken = sorted(used - defined)
    body = re.sub(r'(?<!\\)%.*', '', text)                      # drop comments
    cited = set(k.strip() for grp in re.findall(r'\\cite[pt]?\*?(?:\[[^]]*\])*\{([^}]+)\}', body)
                for k in grp.split(','))
    items = set(re.findall(r'\\bibitem(?:\[[^]]*\])?\{([^}]+)\}', body))
    bib = os.path.join(ROOT, p + '.bib')                        # BibTeX manuscripts keep entries there
    if os.path.exists(bib):
        items |= set(re.findall(r'@\w+\{([^,\s]+),', io.open(bib, encoding='utf-8').read()))
    uncited = sorted(items - cited)
    missing = sorted(cited - items)
    bad_any |= bool(broken or uncited or missing)
    print('  %-12s refs=%-3d broken: %s' % (p, len(used), ', '.join(broken) or 'none'))
    print('  %-12s cites=%-2d bibitems=%-2d uncited: %s | no bibitem: %s'
          % ('', len(cited), len(items), ', '.join(uncited) or 'none', ', '.join(missing) or 'none'))

sys.exit(1 if bad_any else 0)
