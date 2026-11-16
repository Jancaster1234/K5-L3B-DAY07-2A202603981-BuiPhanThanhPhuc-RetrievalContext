"""Merge Day 6 questions + new Day 7 questions into data/benchmark.json (Day 7 schema).

Day 6 dev and Day 6 test questions all become Day 7 **dev**: Day 6 test was already inspected
(post-hoc metrics), so it cannot serve as an independent Day 7 test. Day 7 test = new questions only.

python -m src.build_benchmark --day6 <Day6 repo>/benchmark/questions.json --pages <Day6 repo>/starter/data/pages.jsonl
"""
import argparse
import json
from collections import Counter

from .config import resolve


def from_day6(q):
    return {'id': q['id'], 'split': 'dev', 'origin': f"day6-{q['split']}", 'type': q.get('type'),
            'question': q['question'], 'expected_answer': q['expected_answer'], 'answerable': True,
            'gold': [{'source_id': g['source_id'], 'page': g['page'], 'evidence': g['evidence_quote']} for g in q['gold']],
            'filter': q.get('filter')}


def check_against_pages(questions, pages):
    """Every gold evidence must be verbatim on its gold page (page-level check; chunk-level is in validate())."""
    text = {(p['document_id'], p['page']): p['text'] for p in pages}
    problems = []
    for q in questions:
        for g in q.get('gold', []):
            if g['evidence'] not in text.get((g['source_id'], g['page']), ''):
                problems.append(f"{q['id']}: evidence không có trên {g['source_id']} p{g['page']}")
    return problems


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--day6', required=True)
    p.add_argument('--new', default='data/day7_new_questions.json')
    p.add_argument('--pages', help='Day 6 pages.jsonl để kiểm tra evidence nguyên văn')
    p.add_argument('--output', default='data/benchmark.json')
    a = p.parse_args()
    day6 = json.loads(resolve(a.day6).read_text(encoding='utf-8'))['questions']
    new = json.loads(resolve(a.new).read_text(encoding='utf-8'))['questions']
    questions = [from_day6(q) for q in day6]
    for q in new:
        questions.append({'origin': 'day7-new', 'answerable': True, **q})
    ids = [q['id'] for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError('question ID trùng')
    if a.pages:
        pages = [json.loads(line) for line in resolve(a.pages).read_text(encoding='utf-8').splitlines() if line.strip()]
        problems = check_against_pages(questions, pages)
        if problems:
            raise SystemExit('\n'.join(problems))
    resolve(a.output).write_text(json.dumps(questions, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(Counter(f"{q['split']}/{'answerable' if q['answerable'] else 'unanswerable'}"
                             for q in questions), indent=2))


if __name__ == '__main__':
    main()
