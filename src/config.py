"""Experiment configs, OpenAI client and token/cost accounting."""
import json
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from . import ROOT

# USD per 1M tokens (input, output). Verify at https://developers.openai.com/api/docs/pricing
# before reporting; override per config with {"prices": {"model": [in, out]}}.
PRICES = {
    'gpt-5-mini': (0.25, 2.00),
    'text-embedding-3-small': (0.02, 0.0),
    'text-embedding-3-large': (0.13, 0.0),
    'BAAI/bge-small-en-v1.5': (0.0, 0.0),  # local CPU model (Day 6)
}

DEFAULTS = {
    'name': 'unnamed',
    'method': 'dense',            # dense | bm25 | tfidf | hybrid | hybrid_rerank | dense_rerank
    'top_k': 3,                   # chunks returned per search call (and sent to context)
    'candidates': 20,             # per-retriever candidates for fusion / rerank
    'rrf_k': 60,
    'budget': 800,                # context budget in characters (starter pack_context)
    'max_searches': 4,
    'agent_model': 'gpt-5-mini',
    'rerank_model': 'gpt-5-mini',
    'reasoning_effort': 'low',
    'bm25': {'k1': 1.5, 'b': 0.75, 'bigrams': False},
}


def _merge(base, override):
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_raw(path):
    cfg = json.loads(path.read_text(encoding='utf-8'))
    if 'extends' in cfg:
        cfg = _merge(_load_raw(path.parent / cfg.pop('extends')), cfg)
    return cfg


def load_config(path):
    """Load an experiment config; `extends` names a base config relative to it."""
    path = resolve(path)
    raw = _load_raw(path)
    raw.setdefault('name', path.stem)
    return _merge(DEFAULTS, raw)


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def env(name):
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    return os.environ.get(name)


def openai_client():
    from openai import OpenAI
    env('OPENAI_API_KEY')
    if not os.environ.get('OPENAI_API_KEY'):
        raise RuntimeError('OPENAI_API_KEY chưa được đặt trong .env hoặc biến môi trường')
    return OpenAI()


class Usage:
    """Accumulates tokens per model; cost uses PRICES (None if a model has no price)."""

    def __init__(self, prices=None):
        self.prices = {**PRICES, **{k: tuple(v) for k, v in (prices or {}).items()}}
        self.rows = defaultdict(lambda: {'calls': 0, 'input_tokens': 0, 'output_tokens': 0})

    def add(self, model, input_tokens, output_tokens=0):
        row = self.rows[model]
        row['calls'] += 1
        row['input_tokens'] += int(input_tokens or 0)
        row['output_tokens'] += int(output_tokens or 0)

    def snapshot(self):
        return {m: dict(r) for m, r in self.rows.items()}

    def cost(self, rows=None):
        rows = self.snapshot() if rows is None else rows
        total = 0.0
        for model, r in rows.items():
            if model not in self.prices:
                return None
            pin, pout = self.prices[model]
            total += (r['input_tokens'] * pin + r['output_tokens'] * pout) / 1e6
        return total

    @staticmethod
    def diff(after, before):
        out = {}
        for model, r in after.items():
            b = before.get(model, {'calls': 0, 'input_tokens': 0, 'output_tokens': 0})
            d = {k: r[k] - b[k] for k in r}
            if any(d.values()):
                out[model] = d
        return out
