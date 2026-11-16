"""Retrieval-only benchmark for one config and split (original query, no rewriting).

python -m src.run_retrieval_eval --config configs/dense.json --split dev
Reports Hit@1, Evidence Hit@3, MRR@k, Context Evidence Rate, latency and cost on answerable questions.
"""
import argparse
import json
import statistics
import time

from baseline import pack_context, validate  # starter/baseline.py

from . import ROOT
from .config import Usage, load_config, resolve
from .evaluation import answerable, covered, first_hit_rank, ratio
from .pipeline import Components, load_chunks


def run(cfg, split, out_dir, components=None):
    comp = components or Components(cfg)
    questions = json.loads(resolve(cfg['questions_path']).read_text(encoding='utf-8'))
    selected = [q for q in questions if q.get('split') == split and answerable(q)]
    if not selected:
        raise ValueError(f'không có câu có đáp án trong split {split}')
    retriever = comp.retriever()
    validate(load_chunks(cfg, comp.store), selected)
    top_k, budget = cfg['top_k'], cfg['budget']
    depth = max(3, top_k)
    traces = []
    for q in selected:
        before = comp.usage.snapshot()
        t0 = time.perf_counter()
        results = retriever.search(q['question'], filters=q.get('filter'), top_k=depth)
        context, ids = pack_context(results[:top_k], budget=budget)
        ms = (time.perf_counter() - t0) * 1000
        used = Usage.diff(comp.usage.snapshot(), before)
        rank = first_hit_rank(results, q)
        in_context = [r for r in results if r['id'] in ids]
        traces.append({
            'query_id': q['id'], 'split': split, 'config': cfg['name'], 'question': q['question'],
            'filter': q.get('filter'), 'strategy': retriever.name, 'top_k': top_k, 'budget_chars': budget,
            'gold': q['gold'], 'gold_mode': q.get('gold_mode', 'any'),
            'results': [{**r, 'hit': covered([r], {**q, 'gold_mode': 'any'})} for r in results],
            'context': context, 'context_ids': ids, 'context_chars': len(context),
            'hit1': rank == 1, 'hit3': rank is not None and rank <= 3, 'first_hit_rank': rank,
            'context_hit': covered(in_context, q), 'elapsed_ms': ms, 'usage': used, 'cost_usd': comp.usage.cost(used)})
    N = len(traces)
    costs = [t['cost_usd'] for t in traces]
    summary = {
        'kind': 'retrieval', 'config': cfg['name'], 'method': cfg['method'], 'strategy': retriever.name,
        'split': split, 'top_k': top_k, 'candidates': cfg.get('candidates'), 'budget_chars': budget,
        'hit1': ratio(sum(t['hit1'] for t in traces), N),
        'evidence_hit3': ratio(sum(t['hit3'] for t in traces), N),
        'mrr': sum(1 / t['first_hit_rank'] for t in traces if t['first_hit_rank']) / N,
        'context_evidence_rate': ratio(sum(t['context_hit'] for t in traces), N),
        'avg_context_chars': statistics.mean(t['context_chars'] for t in traces),
        'median_ms': statistics.median(t['elapsed_ms'] for t in traces),
        'cost_usd': None if None in costs else sum(costs), 'usage': comp.usage.snapshot(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in (('traces.json', traces), ('summary.json', summary), ('config.json', cfg)):
        (out_dir / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--split', required=True, choices=('dev', 'test'))
    p.add_argument('--output', help='mặc định results/retrieval-<split>-<config>')
    a = p.parse_args()
    cfg = load_config(a.config)
    out = resolve(a.output) if a.output else ROOT / 'results' / f"retrieval-{a.split}-{cfg['name']}"
    print(json.dumps(run(cfg, a.split, out), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
