"""Gold-based scoring helpers. Used only AFTER retrieval/agent runs, never inside them."""
from baseline import hit  # starter/baseline.py: same source_id, same page, evidence substring of chunk text


def gold_mode(q):
    """'any' (default): gold items are alternatives. 'all': every gold item is required (multi-evidence)."""
    return q.get('gold_mode', 'any')


def covered(chunks, q):
    gold = q['gold']
    if gold_mode(q) == 'all':
        return all(any(hit(c, [g]) for c in chunks) for g in gold)
    return any(hit(c, gold) for c in chunks)


def first_hit_rank(results, q):
    """1-based rank where the evidence requirement is first satisfied, else None."""
    for k in range(1, len(results) + 1):
        if covered(results[:k], q):
            return k
    return None


def answerable(q):
    return q.get('answerable', True) is not False


def ratio(n, N):
    return {'n': n, 'N': N, 'rate': (n / N) if N else None}
