"""Utility: strip all non-ASCII chars from a Python source file to make it cp1252-safe."""
import sys

MAPPING = {
    '\u2192': '->',
    '\u2190': '<-',
    '\u2194': '<->',
    '\u2500': '-',
    '\u2501': '=',
    '\u2502': '|',
    '\u250c': '+',
    '\u2510': '+',
    '\u2514': '+',
    '\u2518': '+',
    '\u251c': '+',
    '\u2524': '+',
    '\u252c': '+',
    '\u2534': '+',
    '\u253c': '+',
    '\u2550': '=',
    '\u2551': '|',
    '\u2554': '+',
    '\u2557': '+',
    '\u255a': '+',
    '\u255d': '+',
    '\u2560': '+',
    '\u2563': '+',
    '\u2566': '+',
    '\u2569': '+',
    '\u256c': '+',
    '\u2713': '[OK]',
    '\u2714': '[OK]',
    '\u2717': '[X]',
    '\u2718': '[X]',
    '\u0394': 'Delta',
    '\u03b1': 'alpha',
    '\u03b2': 'beta',
    '\u03b3': 'gamma',
    '\u03bc': 'mu',
    '\u03c3': 'sigma',
    '\u03c0': 'pi',
    '\u25b6': '>',
    '\u25c0': '<',
    '\u2260': '!=',
    '\u2264': '<=',
    '\u2265': '>=',
    '\u2018': "'",
    '\u2019': "'",
    '\u201c': '"',
    '\u201d': '"',
    '\u2026': '...',
    '\u2013': '-',
    '\u2014': '--',
    '\u00b2': '2',
    '\u00b3': '3',
    '\u00b0': ' deg',
    '\u2207': 'nabla',
}

def scrub(path):
    with open(path, encoding='utf-8') as f:
        src = f.read()
    result = []
    for ch in src:
        if ord(ch) > 127:
            result.append(MAPPING.get(ch, '?'))
        else:
            result.append(ch)
    cleaned = ''.join(result)
    try:
        cleaned.encode('cp1252')
    except UnicodeEncodeError as e:
        # Find remaining bad chars
        for i, ch in enumerate(cleaned):
            try:
                ch.encode('cp1252')
            except UnicodeEncodeError:
                print(f'  Still bad at pos {i}: U+{ord(ch):04X} = {repr(ch)}')
        sys.exit(1)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(cleaned)
    print(f'[OK] Scrubbed: {path}')

for target in ['evaluate_rollout_v2.py', 'st_gnn_v2_model.py', 'train_st_gnn_v2.py']:
    scrub(target)
