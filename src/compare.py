"""Collect results/*/summary.json into markdown tables for the report.

python -m src.compare [--split dev]
"""
import argparse
import json

from . import ROOT


def fmt(r):
    if not r or r.get('N') in (None, 0):
        return '0/0 (n/a)'
    return f"{r['n']}/{r['N']} ({r['rate']:.0%})"


def money(x):
    return 'n/a' if x is None else f'${x:.4f}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split', choices=('dev', 'test'))
    a = p.parse_args()
    rows = []
    for f in sorted((ROOT / 'results').glob('*/summary.json')):
        s = json.loads(f.read_text(encoding='utf-8'))
        if a.split and s.get('split') != a.split:
            continue
        rows.append((f.parent.name, s))
    print('### Retrieval (query gốc)\n')
    print('| Run | Method | top-k / budget | Hit@1 | Evidence Hit@3 | MRR | Context Evidence Rate | median ms | cost |')
    print('|---|---|---|---|---|---|---|---|---|')
    for name, s in rows:
        if s.get('kind') == 'retrieval':
            print(f"| {name} | {s['strategy']} | {s['top_k']} / {s['budget_chars']} ch | {fmt(s['hit1'])} | "
                  f"{fmt(s['evidence_hit3'])} | {s['mrr']:.3f} | {fmt(s['context_evidence_rate'])} | "
                  f"{s['median_ms']:.0f} | {money(s['cost_usd'])} |")
    print('\n### Agent (số tự động; điểm grounded/citation lấy từ agent_metrics)\n')
    print('| Run | Status | Tool use | Searches | Context evidence (auto) | Answered (answerable) | '
          'Abstained (unanswerable) | median s | cost |')
    print('|---|---|---|---|---|---|---|---|---|')
    for name, s in rows:
        if s.get('kind') == 'agent':
            print(f"| {name} | {s['status']} | {fmt(s['tool_use'])} | {s['searches_distribution']} | "
                  f"{fmt(s['context_evidence_rate_auto'])} | {fmt(s['answered_on_answerable'])} | "
                  f"{fmt(s['abstained_on_unanswerable'])} | {s['median_elapsed_s']:.1f} | {money(s['cost_usd'])} |")


if __name__ == '__main__':
    main()
