"""Run the real agent (OpenAI controller + configured retriever) on one split.

python -m src.run_agent_bench --config configs/dense.json --split dev
Writes traces.json (agent_metrics.py format + details), cases.json, summary.json (automatic numbers only)
and reviews.todo.json: a grading sheet YOU fill in by reading the cited chunks, then save as reviews.json.
"""
import argparse
import json
import statistics
import time
from collections import Counter

from agentic import run_agent  # starter/agentic.py

from . import ROOT
from .config import Usage, load_config, resolve
from .controller import OpenAIController
from .evaluation import answerable, covered, ratio
from .pipeline import Components


class LoggingRetriever:
    """Wraps a retriever to time each call; errors propagate (never replaced by fake results)."""

    def __init__(self, inner):
        self.inner, self.name, self.log = inner, inner.name, []

    def search(self, query, *, filters=None, top_k=3):
        t0 = time.perf_counter()
        rows = self.inner.search(query, filters=filters, top_k=top_k)
        self.log.append({'tool': self.name, 'query': query, 'filter': filters, 'top_k': top_k,
                         'n_results': len(rows), 'latency_ms': (time.perf_counter() - t0) * 1000})
        return rows


def run_one(q, cfg, comp, controller, retriever):
    controller.reset()
    retriever.log = []
    before = comp.usage.snapshot()
    t0 = time.perf_counter()
    trace = {'query_id': q['id'], 'split': q['split'], 'config': cfg['name'], 'question': q['question'],
             'filter': q.get('filter'), 'answerable': answerable(q)}
    try:
        # Only the question text and the user-provided filter reach the agent; no gold.
        res = run_agent(q['question'], controller, retriever, filters=q.get('filter'),
                        max_searches=cfg['max_searches'], top_k=cfg['top_k'], budget=cfg['budget'])
        trace.update(status=res['status'], answer=res['answer'], citations=res['citations'], events=res['events'],
                     error=None)
    except Exception as e:  # provider / vector store / parsing error: report, do not hide
        trace.update(status='error', answer=None, citations=[], events=[], error=f'{type(e).__name__}: {e}')
    searches = [e for e in trace['events'] if e.get('event') == 'search']
    trace['final_context'] = searches[-1]['context'] if searches else ''
    trace['final_context_ids'] = searches[-1]['context_ids'] if searches else []
    trace['n_searches'] = len(searches)
    trace['controller_log'] = controller.log
    trace['search_log'] = retriever.log
    if controller.log and controller.log[-1]['tool'] == 'abstain':
        trace['abstain_reason'] = controller.log[-1]['args'].get('reason')
    trace['elapsed_s'] = time.perf_counter() - t0
    trace['usage'] = Usage.diff(comp.usage.snapshot(), before)
    trace['cost_usd'] = comp.usage.cost(trace['usage'])
    return trace


def review_row(q, t, chunk_by_id):
    row = {'query_id': t['query_id'], 'answerable': t['answerable'], 'status': t['status'],
           'question': q['question'], 'expected': q.get('expected_answer') or q.get('absence_reason'),
           'answer': t['answer'], 'citations': t['citations'],
           'cited_chunks': [chunk_by_id.get(c, {'id': c}) for c in t['citations']],
           'citation_support': [None] * len(t['citations']), 'reason': ''}
    if t['answerable']:
        row['answer_grade'] = 0 if t['status'] == 'abstain' else None
    else:
        row['abstention_correct'] = None
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--split', required=True, choices=('dev', 'test'))
    p.add_argument('--ids', nargs='*', help='chỉ chạy các query ID này (debug)')
    p.add_argument('--output', help='mặc định results/agent-<split>-<config>')
    a = p.parse_args()
    cfg = load_config(a.config)
    out = resolve(a.output) if a.output else ROOT / 'results' / f"agent-{a.split}-{cfg['name']}"
    questions = json.loads(resolve(cfg['questions_path']).read_text(encoding='utf-8'))
    selected = [q for q in questions if q.get('split') == a.split and (not a.ids or q['id'] in a.ids)]
    if not selected:
        raise SystemExit('không có câu hỏi phù hợp')
    comp = Components(cfg)
    retriever = LoggingRetriever(comp.retriever())
    controller = OpenAIController(comp.client, cfg['agent_model'], comp.usage,
                                  max_searches=cfg['max_searches'], reasoning_effort=cfg['reasoning_effort'],
                                  prompt=cfg.get('prompt', 'base'))
    traces = []
    for q in selected:
        t = run_one(q, cfg, comp, controller, retriever)
        if t['answerable'] and t['status'] != 'error':  # diagnostic only, computed after the run
            ctx = [r for e in t['events'] if e.get('event') == 'search' for r in e['results']
                   if r['id'] in t['final_context_ids']]
            t['context_evidence_auto'] = covered(list({r['id']: r for r in ctx}.values()), q)
        traces.append(t)
        print(f"{t['query_id']}: {t['status']} searches={t['n_searches']} "
              f"cost={t['cost_usd'] if t['cost_usd'] is None else round(t['cost_usd'], 5)} {t['error'] or ''}")
    ok = [t for t in traces if t['status'] != 'error']
    ans_ok = [t for t in ok if t['answerable']]
    costs = [t['cost_usd'] for t in traces]
    summary = {
        'kind': 'agent', 'config': cfg['name'], 'method': cfg['method'], 'split': a.split,
        'agent_model': cfg['agent_model'], 'top_k': cfg['top_k'], 'budget_chars': cfg['budget'],
        'max_searches': cfg['max_searches'], 'n': len(traces),
        'status': dict(Counter(t['status'] for t in traces)),
        'errors': [{'query_id': t['query_id'], 'error': t['error']} for t in traces if t['status'] == 'error'],
        'tool_use': ratio(sum(t['n_searches'] > 0 for t in ok), len(ok)),
        'searches_distribution': dict(sorted(Counter(t['n_searches'] for t in ok).items())),
        'context_evidence_rate_auto': ratio(sum(t['context_evidence_auto'] for t in ans_ok), len(ans_ok)),
        'answered_on_answerable': ratio(sum(t['status'] == 'answered' for t in ans_ok), len(ans_ok)),
        'abstained_on_unanswerable': ratio(sum(t['status'] == 'abstain' for t in ok if not t['answerable']),
                                           sum(not t['answerable'] for t in ok)),
        'median_elapsed_s': statistics.median(t['elapsed_s'] for t in traces),
        'cost_usd': None if None in costs else sum(costs), 'usage': comp.usage.snapshot(),
        'note': 'Số tự động; grounded answer/citation/abstention phải lấy từ reviews.json + starter/agent_metrics.py',
    }
    chunk_by_id = {r['id']: {k: r[k] for k in ('id', 'source', 'page', 'text')}
                   for t in traces for e in t['events'] if e.get('event') == 'search' for r in e['results']}
    q_by_id = {q['id']: q for q in selected}
    cases = [{'id': q['id'], 'split': q['split'], 'answerable': answerable(q)} for q in selected]
    reviews = [review_row(q_by_id[t['query_id']], t, chunk_by_id) for t in ok]
    out.mkdir(parents=True, exist_ok=True)
    for name, obj in (('traces.json', traces), ('cases.json', cases), ('summary.json', summary),
                      ('reviews.todo.json', reviews), ('config.json', cfg)):
        (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'usage'}, ensure_ascii=False, indent=2))
    if (out / 'reviews.json').exists():
        print('Lưu ý: reviews.json đã tồn tại và KHÔNG bị ghi đè; đối chiếu lại với trace mới.')


if __name__ == '__main__':
    main()
