"""Vector store adapters. Each returns (chunk_dict, cosine_score) pairs in Day 7 chunk schema."""
import json
from pathlib import Path

import numpy as np


def matches(chunk, filters):
    """Equality filter on chunk metadata (or top-level fields such as source_id/page)."""
    if not filters:
        return True
    meta = chunk.get('metadata') or {}
    return all(meta.get(k, chunk.get(k)) == v for k, v in filters.items())


class NumpyStore:
    """Exact cosine search over a saved matrix: index_dir/{chunks.json, embeddings.npy, meta.json}."""

    def __init__(self, chunks, matrix, meta=None):
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim != 2 or len(chunks) != matrix.shape[0]:
            raise ValueError('số chunk và số vector không khớp')
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.matrix = matrix / np.where(norms == 0, 1, norms)
        self.chunks, self.meta = chunks, meta or {}

    @property
    def dim(self):
        return self.matrix.shape[1]

    @classmethod
    def load(cls, index_dir):
        d = Path(index_dir)
        chunks = json.loads((d / 'chunks.json').read_text(encoding='utf-8'))
        meta = json.loads((d / 'meta.json').read_text(encoding='utf-8')) if (d / 'meta.json').exists() else {}
        return cls(chunks, np.load(d / 'embeddings.npy'), meta)

    def save(self, index_dir):
        d = Path(index_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / 'chunks.json').write_text(json.dumps(self.chunks, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        (d / 'meta.json').write_text(json.dumps(self.meta, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        np.save(d / 'embeddings.npy', self.matrix)

    def query(self, vector, top_k, filters=None):
        q = np.asarray(vector, dtype=np.float32)
        if q.shape != (self.dim,):
            raise ValueError(f'query embedding dim {q.shape} khác index dim {self.dim}')
        q = q / (np.linalg.norm(q) or 1)
        scores = self.matrix @ q
        order = np.argsort(-scores, kind='stable')
        out = []
        for i in order:
            if matches(self.chunks[i], filters):
                out.append((self.chunks[i], float(scores[i])))
                if len(out) == top_k:
                    break
        return out


class PgVectorStore:
    """Read-only adapter for the Day 6 PostgreSQL + pgvector table `chunks` (one `strategy` partition).

    Day 6 schema: (strategy, chunk_id) PK, document_id, source, page, metadata jsonb, text, model_name,
    embedding vector(d). Search = exact cosine `<=>`, score = 1 - distance (same as Day 6).
    Filters are applied in SQL on metadata jsonb with parameters (no string interpolation of values).
    """

    def __init__(self, database_url, strategy):
        import psycopg  # optional dependency
        if not database_url:
            raise ValueError('DATABASE_URL rỗng')
        self.psycopg, self.url, self.strategy = psycopg, database_url, strategy
        with self._connect() as con:
            row = con.execute("""SELECT count(*), min(model_name), max(model_name),
                                        max(vector_dims(embedding)) FROM chunks WHERE strategy=%s""",
                              (strategy,)).fetchone()
        if not row[0]:
            raise ValueError(f'không có chunk nào cho strategy {strategy!r}')
        if row[1] != row[2]:
            raise ValueError(f'strategy {strategy} trộn nhiều embedding model: {row[1]}, {row[2]}')
        self.n_chunks, self.meta = row[0], {'embedding_model': row[1], 'dimensions': row[3],
                                            'store': 'pgvector', 'strategy': strategy}

    @property
    def dim(self):
        return self.meta['dimensions']

    def _connect(self):
        return self.psycopg.connect(self.url, connect_timeout=10)

    @staticmethod
    def _chunk(r):
        return {'id': r[0], 'source_id': r[1], 'source': r[2], 'page': r[3], 'metadata': r[4], 'text': r[5]}

    def all_chunks(self):
        with self._connect() as con:
            rows = con.execute("""SELECT chunk_id, document_id, source, page, metadata, text FROM chunks
                                  WHERE strategy=%s ORDER BY document_id, page, chunk_id""",
                               (self.strategy,)).fetchall()
        return [self._chunk(r) for r in rows]

    @property
    def chunks(self):
        if not hasattr(self, '_chunks'):
            self._chunks = self.all_chunks()
        return self._chunks

    def query(self, vector, top_k, filters=None):
        if len(vector) != self.dim:
            raise ValueError(f'query embedding dim {len(vector)} khác index dim {self.dim}')
        where, params = ['strategy = %s'], [self.strategy]
        for key, value in (filters or {}).items():
            where.append('metadata ->> %s = %s')
            params += [key, str(value)]
        qv = '[' + ','.join(f'{x:.7f}' for x in vector) + ']'
        sql = f"""SELECT chunk_id, document_id, source, page, metadata, text, 1 - (embedding <=> %s::vector)
                  FROM chunks WHERE {' AND '.join(where)}
                  ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s"""
        with self._connect() as con:
            rows = con.execute(sql, [qv, *params, qv, top_k]).fetchall()
        return [(self._chunk(r), float(r[6])) for r in rows]


class ChromaStore:
    """Persistent Chroma collection from Day 6. `fields` maps Day 7 keys to Chroma metadata keys."""

    def __init__(self, path, collection, fields=None):
        import chromadb  # optional dependency, only needed if Day 6 used Chroma
        self.col = chromadb.PersistentClient(path=str(path)).get_collection(collection)
        self.fields = {'source_id': 'source_id', 'source': 'source', 'page': 'page', **(fields or {})}

    @property
    def dim(self):
        got = self.col.peek(1)
        emb = got.get('embeddings')
        if emb is None or len(emb) == 0:
            raise ValueError('collection rỗng')
        return len(emb[0])

    def _chunk(self, cid, text, meta):
        meta = dict(meta or {})
        out = {'id': cid, 'text': text}
        for key, src in self.fields.items():
            out[key] = meta.pop(src, None)
        out['metadata'] = meta
        return out

    def all_chunks(self):
        got = self.col.get(include=['documents', 'metadatas'])
        return [self._chunk(i, t, m) for i, t, m in zip(got['ids'], got['documents'], got['metadatas'])]

    def query(self, vector, top_k, filters=None):
        where = None
        if filters:
            where = filters if len(filters) == 1 else {'$and': [{k: v} for k, v in filters.items()]}
        got = self.col.query(query_embeddings=[list(vector)], n_results=top_k, where=where,
                             include=['documents', 'metadatas', 'distances'])
        rows = zip(got['ids'][0], got['documents'][0], got['metadatas'][0], got['distances'][0])
        # Assumes cosine space (distance = 1 - cosine); record the space in the report.
        return [(self._chunk(i, t, m), 1.0 - float(d)) for i, t, m, d in rows]
