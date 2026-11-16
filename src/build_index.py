"""Build a NumpyStore index from Day 7 chunks.

Two modes:
  * --embeddings <file.npy|.json>: reuse vectors exported from Day 6 (same row order as chunks). No API cost.
  * otherwise embed chunks with OpenAI (records tokens and cost in meta.json).

python -m src.build_index --chunks data/chunks.json --output index --model text-embedding-3-small
"""
import argparse
import json
import time

import numpy as np

from .config import Usage, openai_client, resolve
from .embeddings import OpenAIEmbedder
from .stores import NumpyStore

REQUIRED = ('id', 'text', 'source_id', 'source', 'page', 'metadata')


def check_chunks(chunks):
    if not isinstance(chunks, list) or not chunks:
        raise ValueError('chunks phải là JSON array không rỗng')
    ids = set()
    for c in chunks:
        missing = [k for k in REQUIRED if c.get(k) is None]
        if missing:
            raise ValueError(f"chunk {c.get('id')} thiếu {missing}")
        if c['id'] in ids:
            raise ValueError(f"chunk ID trùng: {c['id']}")
        ids.add(c['id'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--chunks', required=True)
    p.add_argument('--output', default='index')
    p.add_argument('--model', default='text-embedding-3-small')
    p.add_argument('--dimensions', type=int)
    p.add_argument('--embeddings', help='vector Day 6 đã có (.npy hoặc JSON array), cùng thứ tự với chunks')
    a = p.parse_args()
    chunks = json.loads(resolve(a.chunks).read_text(encoding='utf-8'))
    check_chunks(chunks)
    usage = Usage()
    t0 = time.perf_counter()
    if a.embeddings:
        src = resolve(a.embeddings)
        matrix = np.load(src) if src.suffix == '.npy' else np.array(json.loads(src.read_text(encoding='utf-8')))
        origin = f'reused from {a.embeddings}'
    else:
        embedder = OpenAIEmbedder(openai_client(), a.model, usage, dimensions=a.dimensions,
                                  cache_dir=resolve('cache'))
        matrix = np.array(embedder.embed([c['text'] for c in chunks]))
        origin = 'embedded with OpenAI API'
    meta = {'embedding_model': a.model, 'dimensions': int(matrix.shape[1]), 'n_chunks': len(chunks),
            'origin': origin, 'build_seconds': time.perf_counter() - t0,
            'usage': usage.snapshot(), 'cost_usd': usage.cost(),
            'n_sources': len({c['source_id'] for c in chunks}),
            'n_pages': len({(c['source_id'], c['page']) for c in chunks})}
    NumpyStore(chunks, matrix, meta).save(resolve(a.output))
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
