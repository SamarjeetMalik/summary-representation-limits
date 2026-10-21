"""Is each station-level sample (2,400 blocks of 200 events) a fair sample of its release?

Compares the primary fractions and the median true energy of kcdc_station_features[_gen].npz with
the full converted release kcdc_kascade_<gen>.npz. Output: analysis/kcdc/kcdc_sample_check.json.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'data', 'kcdc')
R = {}
for tag, suf in (('qgs4', ''), ('epos', '_epos'), ('sibyll', '_sibyll')):
    fp, sp = os.path.join(D, 'kcdc_kascade_%s.npz' % tag), os.path.join(D, 'kcdc_station_features%s.npz' % suf)
    if not (os.path.exists(fp) and os.path.exists(sp)):
        continue
    f, s = np.load(fp), np.load(sp)
    pr = np.unique(f['TrPP'])
    ff = np.array([(f['TrPP'] == p).mean() for p in pr]); sf = np.array([(s['TrPP'] == p).mean() for p in pr])
    R[tag] = {'n_sample': int(len(s['TrPP'])), 'max_primary_fraction_diff': round(float(np.abs(ff - sf).max()), 4),
              'median_lgE_diff': round(float(np.median(s['TrEP']) - np.median(f['TrEP'])), 4)}
    print(tag, R[tag])
json.dump(R, open(os.path.join(HERE, 'kcdc_sample_check.json'), 'w'), indent=2)
