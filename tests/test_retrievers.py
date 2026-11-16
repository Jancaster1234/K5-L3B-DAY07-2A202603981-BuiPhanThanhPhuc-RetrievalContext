"""Offline checks of retrieval code with fake embeddings/LLM; these do not measure real quality."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Usage, load_config
from src.evaluation import covered, first_hit_rank
from src.retrievers import BM25Retriever, DenseRetriever, LLMRerankRetriever, RRFRetriever, TfidfRetriever
from src.stores import NumpyStore


def chunk(cid, text, page=1, **meta):
    return {'id': cid, 'text': text, 'source_id': 'doc', 'source': 'doc.pdf', 'page': page, 'metadata': meta}


CHUNKS = [chunk('a', 'Hoàn tiền được xử lý trong 7 ngày làm việc.', 1, year=2025),
          chunk('b', 'Giao hàng nội thành mất 2 ngày.', 2, year=2025),
          chunk('c', 'Hoàn tiền năm 2024 mất 14 ngày.', 3, year=2024)]


class FakeEmbedder:
    VOCAB = ['hoàn', 'giao', '2024']

    def embed(self, texts):
        return [[float(w in t.casefold()) + 0.01 for w in self.VOCAB] for t in texts]


class Fixed:
    def __init__(self, name, ids):
        self.name, self.ids = name, ids

    def search(self, query, *, filters=None, top_k=3):
        return [{**chunk(i, i), 'score': 1.0, 'rank': r + 1, 'strategy': self.name} for r, i in enumerate(self.ids[:top_k])]


def fake_llm(content, prompt_tokens=100, completion_tokens=20):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
                               choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls


class RetrieverTests(unittest.TestCase):
    def test_bm25_ranks_and_filters(self):
        r = BM25Retriever(CHUNKS)
        rows = r.search('hoàn tiền bao lâu', top_k=3)
        self.assertEqual({rows[0]['id'], rows[1]['id']}, {'a', 'c'})
        self.assertEqual([x['rank'] for x in rows], [1, 2])
        self.assertEqual([x['id'] for x in r.search('hoàn tiền', filters={'year': 2024})], ['c'])
        self.assertEqual(r.search('không khớp gì'), [])
        self.assertTrue(all({'id', 'text', 'source', 'page', 'score', 'rank', 'strategy'} <= set(x) for x in rows))

    def test_bm25_bigrams_prefer_phrase(self):
        chunks = [chunk('x', 'tiền hoàn lại và phí giao'), chunk('y', 'hoàn tiền')]
        self.assertEqual(BM25Retriever(chunks, bigrams=True).search('hoàn tiền')[0]['id'], 'y')

    def test_tfidf_wrapper_filters(self):
        self.assertEqual([x['id'] for x in TfidfRetriever(CHUNKS).search('hoàn tiền', filters={'year': 2024})], ['c'])

    def test_dense_numpy_store_and_dim_check(self):
        emb = FakeEmbedder()
        store = NumpyStore(CHUNKS, emb.embed([c['text'] for c in CHUNKS]))
        rows = DenseRetriever(store, emb).search('giao hàng', top_k=1)
        self.assertEqual(rows[0]['id'], 'b')
        self.assertEqual(DenseRetriever(store, emb).search('hoàn tiền', filters={'year': 2024}, top_k=3)[0]['id'], 'c')
        with self.assertRaises(ValueError):
            store.query([1.0, 0.0], 1)

    def test_numpy_store_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            NumpyStore(CHUNKS, [[1, 0], [0, 1], [1, 1]], {'embedding_model': 'm'}).save(d)
            s = NumpyStore.load(d)
            self.assertEqual((s.dim, s.meta['embedding_model'], s.chunks[2]['id']), (2, 'm', 'c'))

    def test_rrf_fuses_by_rank(self):
        rrf = RRFRetriever([Fixed('d', ['a', 'b', 'c']), Fixed('l', ['b', 'c', 'a'])], k=60)
        rows = rrf.search('q', top_k=3)
        self.assertEqual(rows[0]['id'], 'b')
        self.assertEqual(rows[0]['component_ranks'], {'d': 2, 'l': 1})

    def test_llm_rerank_and_bad_json(self):
        client, calls = fake_llm(json.dumps({'scores': [{'id': 'a', 'score': 0}, {'id': 'b', 'score': 3}, {'id': 'c', 'score': 2}]}))
        usage = Usage()
        rr = LLMRerankRetriever(Fixed('d', ['a', 'b', 'c']), client, 'gpt-5-mini', usage, candidates=3)
        rows = rr.search('q', top_k=2)
        self.assertEqual([r['id'] for r in rows], ['b', 'c'])
        self.assertEqual(rows[0]['base_rank'], 2)
        self.assertEqual(usage.snapshot()['gpt-5-mini']['input_tokens'], 100)
        self.assertIn('untrusted', calls[0]['messages'][0]['content'])
        bad, _ = fake_llm('not json')
        with self.assertRaises(ValueError):
            LLMRerankRetriever(Fixed('d', ['a']), bad, 'm', Usage()).search('q')


class EvalAndConfigTests(unittest.TestCase):
    def test_gold_modes(self):
        q = {'gold': [{'source_id': 'doc', 'page': 1, 'evidence': '7 ngày'},
                      {'source_id': 'doc', 'page': 3, 'evidence': '14 ngày'}]}
        self.assertTrue(covered([CHUNKS[0]], q))
        self.assertFalse(covered([CHUNKS[0]], {**q, 'gold_mode': 'all'}))
        self.assertEqual(first_hit_rank([CHUNKS[1], CHUNKS[0], CHUNKS[2]], {**q, 'gold_mode': 'all'}), 3)
        self.assertIsNone(first_hit_rank([CHUNKS[1]], q))

    def test_usage_cost(self):
        u = Usage()
        u.add('gpt-5-mini', 1_000_000, 1_000_000)
        self.assertAlmostEqual(u.cost(), 2.25)
        u.add('unknown-model', 1)
        self.assertIsNone(u.cost())

    def test_config_extends_and_name(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'base.json').write_text('{"top_k": 5, "bm25": {"k1": 1.2}}', encoding='utf-8')
            Path(d, 'exp.json').write_text('{"extends": "base.json", "bm25": {"bigrams": true}}', encoding='utf-8')
            cfg = load_config(Path(d, 'exp.json'))
            self.assertEqual((cfg['name'], cfg['top_k'], cfg['budget']), ('exp', 5, 800))
            self.assertEqual(cfg['bm25'], {'k1': 1.2, 'b': 0.75, 'bigrams': True})


if __name__ == '__main__':
    unittest.main()
