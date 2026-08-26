#!/usr/bin/env python3
"""
mlir_gvn.py -- offline attribution of emitter redundancy in a raw StableHLO dump.

Runs the SAME transformation XLA's CSE runs (global value numbering: an op whose
opcode, operands and attributes already appeared is replaced by the earlier
result), but as a single hash-table pass over the module text, and REPORTS what
it removed instead of doing it silently.

The point: `@code_hlo optimize=false` op count minus the post-GVN count is
exactly the redundancy the EMITTER manufactured and XLA then has to rediscover
pairwise. Per-opcode, that says which construct is responsible.

  usage: mlir_gvn.py FILE.mlir [--top N] [--dups N] [--slices]
"""
import re, sys, collections

OPLINE = re.compile(r'^\s*(%[^=]*?)\s*=\s*(.*)$')
TOK = re.compile(r'%[A-Za-z_][A-Za-z0-9_.$]*|%\d+')
LOC = re.compile(r'\s+loc\([^\n]*\)\s*$')
# opcode: first bare `dialect.op` token after the `=`
OPCODE = re.compile(r'^"?([a-zA-Z_][\w]*\.[\w.]+)"?')


def lhs_names(lhs):
    """`%0`, `%0:2`, `%a, %b` -> list of base names (result groups flattened)."""
    out = []
    for part in lhs.split(','):
        part = part.strip()
        m = re.match(r'(%[A-Za-z0-9_.$]+)(?::(\d+))?', part)
        if not m:
            continue
        base, n = m.group(1), m.group(2)
        if n:
            out.extend(['%s#%d' % (base, i) for i in range(int(n))])
            out.append(base)
        else:
            out.append(base)
    return out


def analyze(path):
    canon = {}            # ssa name -> canonical ssa name
    table = {}            # (depth-scoped) key -> (lhs names, depth)
    scope_keys = collections.defaultdict(list)
    depth = 0
    total = collections.Counter()
    dup = collections.Counter()
    dup_key_hits = collections.Counter()
    slice_windows = collections.Counter()

    with open(path) as fh:
        for line in fh:
            stripped = line.rstrip('\n')
            m = OPLINE.match(stripped)
            if m:
                lhs, rhs = m.group(1), m.group(2)
                rhs = LOC.sub('', rhs).strip()
                oc = OPCODE.match(rhs)
                opcode = oc.group(1) if oc else '<?>'
                total[opcode] += 1
                # substitute operands by their canonical representative
                key = TOK.sub(lambda t: canon.get(t.group(0), t.group(0)), rhs)
                names = lhs_names(lhs)
                if opcode == 'stablehlo.slice':
                    slice_windows[key] += 1
                hit = table.get(key)
                if hit is not None:
                    dup[opcode] += 1
                    dup_key_hits[key] += 1
                    for a, b in zip(names, hit):
                        canon[a] = b
                else:
                    table[key] = names
                    scope_keys[depth].append(key)
                    for a in names:
                        canon.setdefault(a, a)
            # crude region tracking: a value defined inside a region must not be
            # CSE-visible after it closes.
            opens = stripped.count('{') - stripped.count('}')
            if opens > 0:
                depth += opens
            elif opens < 0:
                for _ in range(-opens):
                    for k in scope_keys.pop(depth, []):
                        table.pop(k, None)
                    depth = max(0, depth - 1)
    return total, dup, dup_key_hits, slice_windows


def main():
    path = sys.argv[1]
    top = 18
    ndups = 0
    show_slices = False
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == '--top':
            top = int(args[i + 1]); i += 2
        elif args[i] == '--dups':
            ndups = int(args[i + 1]); i += 2
        elif args[i] == '--slices':
            show_slices = True; i += 1
        else:
            i += 1

    total, dup, dup_keys, slices = analyze(path)
    T, Dd = sum(total.values()), sum(dup.values())
    print('=' * 78)
    print(path)
    print('  ops in dump            %8d' % T)
    print('  removable by pure GVN  %8d   (%.1f%%)' % (Dd, 100.0 * Dd / max(T, 1)))
    print('  residual after GVN     %8d   -> emitter redundancy factor %.2fx'
          % (T - Dd, T / max(T - Dd, 1)))
    print('-' * 78)
    print('  %-38s %8s %8s %8s %6s' % ('opcode', 'emitted', 'unique', 'dupes', '%dup'))
    for oc, n in total.most_common(top):
        d = dup[oc]
        print('  %-38s %8d %8d %8d %5.1f%%' % (oc, n, n - d, d, 100.0 * d / max(n, 1)))
    if ndups:
        print('-' * 78)
        print('  most-repeated single expressions (times emitted beyond the first):')
        for k, n in dup_keys.most_common(ndups):
            print('   %6d x  %s' % (n, k[:150]))
    if show_slices:
        print('-' * 78)
        nsl = sum(slices.values())
        print('  stablehlo.slice: %d emitted over %d distinct (operand, window) pairs'
              % (nsl, len(slices)))
        for k, n in slices.most_common(12):
            print('   %6d x  %s' % (n, k[:150]))


if __name__ == '__main__':
    main()
