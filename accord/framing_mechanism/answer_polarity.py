#!/usr/bin/env python3
"""Post hoc descriptive analysis of answer polarity in the local labellers.

Not pre-registered. Reads only the stored per-pair label caches from the prompt
sensitivity runs; those caches carry a synthetic pair key and a parsed label and
no clinical text, so every output here is an aggregate safe to print.

Two questions:
  1. When the same model is asked "is there a contradiction?" and "are these
     consistent?" on the identical pairs, does it answer coherently? A labeller
     reading evidence should. A labeller answering the prompt should not.
  2. Within one model family under one prompt, does the flag rate move with
     parameter count?
"""
import json, os, glob
from math import sqrt, log10, comb

CACHE = os.path.expanduser('~/mimic_local_only/accord_prompt_sensitivity_cache')
OUT = 'results/accord/framing_mechanism/answer_polarity.json'

CONTRA = {'P_asserting', 'P_neutral'}
CONSIS = {'P_reversed', 'P_reversed_rerun', 'P_rev_agree', 'P_rev_compatible', 'P_rev_follow'}
STEM = {'_granite33_8b_nothink': ('Granite 3.3 8B', 8.0),
        '_llama3_8b_nothink':    ('Llama 3 8B', 8.0),
        '_qwen3_8b_nothink':     ('Qwen 3 8B', 8.0),
        '_qwen3_14b_nothink':    ('Qwen 3 14B', 14.0),
        '_qwen3_4b_nothink':     ('Qwen 3 4B', 4.0),
        '_qwen3_17b_nothink':    ('Qwen 3 1.7B', 1.7),
        '_qwen3_06b_nothink':    ('Qwen 3 0.6B', 0.6)}


def load(name):
    out = {}
    with open(os.path.join(CACHE, name)) as fh:
        for line in fh:
            r = json.loads(line)
            out[r['pair_id']] = r['flag_label']
    return out


def wilson(k, n, z=1.959963985):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


res = {'note': 'post hoc descriptive; not pre-registered', 'cells': [], 'coherence': [], 'ladder': []}

# ---- per cell raw yes rate -------------------------------------------------
cells = {}
for path in sorted(glob.glob(os.path.join(CACHE, '*.jsonl'))):
    base = os.path.basename(path)[:-6]
    stem, prompt = base.split('__', 1)
    if stem not in STEM or prompt not in (CONTRA | CONSIS):
        continue
    model, params = STEM[stem]
    lab = load(base + '.jsonl')
    ok = {k: v for k, v in lab.items() if v is not None}
    unparsed = len(lab) - len(ok)
    if not ok:
        res['cells'].append({'model': model, 'prompt': prompt, 'n_total': len(lab),
                             'n_parsed': 0, 'n_unparsed': unparsed, 'status': 'all unparsed'})
        continue
    flag = sum(ok.values())
    direction = 'contradiction' if prompt in CONTRA else 'consistency'
    raw_yes = flag if direction == 'contradiction' else len(ok) - flag
    res['cells'].append({'model': model, 'params_b': params, 'prompt': prompt,
                         'asks': direction, 'n_parsed': len(ok), 'n_unparsed': unparsed,
                         'contradiction_flag_rate': round(flag / len(ok), 4),
                         'raw_yes_count': raw_yes,
                         'raw_yes_rate': round(raw_yes / len(ok), 4),
                         'raw_yes_wilson95': wilson(raw_yes, len(ok))})
    cells[(model, prompt)] = ok

# ---- coherence between the two directions on identical pairs --------------
for model in ['Llama 3 8B', 'Granite 3.3 8B', 'Qwen 3 8B']:
    rev = cells.get((model, 'P_reversed'))
    if rev is None:
        continue
    for fwd_prompt in ['P_neutral', 'P_asserting']:
        fwd = cells.get((model, fwd_prompt))
        if fwd is None:
            continue
        ids = sorted(set(fwd) & set(rev))
        n = len(ids)
        # normalised label 1 = contradiction indicated.
        # raw yes to "contradiction?" is fwd==1; raw yes to "consistent?" is rev==0.
        yy = sum(1 for i in ids if fwd[i] == 1 and rev[i] == 0)   # yes to both questions
        nn = sum(1 for i in ids if fwd[i] == 0 and rev[i] == 1)   # no to both questions
        a = sum(1 for i in ids if fwd[i] == 1 and rev[i] == 1)
        d = sum(1 for i in ids if fwd[i] == 0 and rev[i] == 0)
        b, c = yy, nn
        po = (a + d) / n
        pe = ((a + b) * (a + c) + (c + d) * (b + d)) / n ** 2
        kappa = (po - pe) / (1 - pe) if pe != 1 else None
        res['coherence'].append({
            'model': model, 'forward_prompt': fwd_prompt, 'reverse_prompt': 'P_reversed', 'n': n,
            'yes_to_both_questions': yy, 'yes_to_both_rate': round(yy / n, 4),
            'yes_to_both_wilson95': wilson(yy, n),
            'no_to_both_questions': nn, 'no_to_both_rate': round(nn / n, 4),
            'no_to_both_wilson95': wilson(nn, n),
            'logically_coherent': n - yy - nn, 'coherent_rate': round((n - yy - nn) / n, 4),
            'kappa_between_framings': None if kappa is None else round(kappa, 4),
            'mcnemar_exact_p': round(mcnemar_exact(b, c), 6)})

# ---- scale ladder, one family, one prompt --------------------------------
lad = [c for c in res['cells']
       if c['model'].startswith('Qwen 3') and c.get('prompt') == 'P_asserting']
lad.sort(key=lambda c: c.get('params_b', 0))
usable = [c for c in lad if c.get('n_parsed')]
res['ladder'] = {'family': 'Qwen 3', 'prompt': 'P_asserting', 'points': lad,
                 'excluded': [c['model'] for c in lad if not c.get('n_parsed')]}
if len(usable) >= 3:
    xs = [log10(c['params_b']) for c in usable]
    ys = [c['raw_yes_rate'] for c in usable]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    res['ladder']['slope_per_log10_params'] = round(sxy / sxx, 4)
    res['ladder']['pearson_r'] = round(sxy / sqrt(sxx * syy), 4)
    res['ladder']['monotone_decreasing'] = all(
        usable[i]['raw_yes_rate'] >= usable[i + 1]['raw_yes_rate'] for i in range(len(usable) - 1))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, 'w'), indent=1)
print('wrote', OUT)
