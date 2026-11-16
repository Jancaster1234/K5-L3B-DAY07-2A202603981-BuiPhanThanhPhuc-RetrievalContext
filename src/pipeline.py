"""Build retriever / controller from a config. The OpenAI client is created only if a method needs it."""
import json

from .config import Usage, env, openai_client, resolve
from .embeddings import LocalSTEmbedder, OpenAIEmbedder
from .retrievers import BM25Retriever, DenseRetriever, LLMRerankRetriever, RRFRetriever, TfidfRetriever
from .stores import ChromaStore, NumpyStore, PgVectorStore

NEEDS_EMBEDDINGS = {'dense', 'hybrid', 'hybrid_rerank', 'dense_rerank'}
NEEDS_LLM = {'hybrid_rerank', 'dense_rerank'}


def load_store(cfg):
    s = cfg['store']
    if s['type'] == 'pgvector':
        return PgVectorStore(env(s.get('url_env', 'DATABASE_URL')), s['strategy'])
    if s['type'] == 'numpy':
        return NumpyStore.load(resolve(s['path']))
    if s['type'] == 'chroma':
        return ChromaStore(resolve(s['path']), s['collection'], s.get('fields'))
    raise ValueError(f"store.type không hỗ trợ: {s['type']}")


def load_chunks(cfg, store=None):
    if cfg.get('chunks_path'):
        return json.loads(resolve(cfg['chunks_path']).read_text(encoding='utf-8'))
    if store is None:
        store = load_store(cfg)
    return store.chunks if hasattr(store, 'chunks') else store.all_chunks()


class Components:
    def __init__(self, cfg, client=None, usage=None):
        self.cfg = cfg
        self.usage = usage or Usage(cfg.get('prices'))
        self._client = client
        self.store = None
        self.embedder = None

    @property
    def client(self):
        if self._client is None:
            self._client = openai_client()
        return self._client

    def check_embedding_compat(self):
        meta = getattr(self.store, 'meta', {}) or {}
        want = meta.get('embedding_model')
        if want and want != self.cfg['embedding_model']:
            raise ValueError(f"index dùng {want} nhưng config dùng {self.cfg['embedding_model']}")

    def retriever(self, method=None):
        method = method or self.cfg['method']
        cfg = self.cfg
        if method in NEEDS_EMBEDDINGS and self.store is None:
            self.store = load_store(cfg)
            self.check_embedding_compat()
            if cfg.get('embedding_provider', 'local') == 'local':
                self.embedder = LocalSTEmbedder(cfg['embedding_model'], self.usage)
                if self.embedder.dimensions != self.store.dim:
                    raise ValueError(f'model dim {self.embedder.dimensions} khác index dim {self.store.dim}')
            else:
                self.embedder = OpenAIEmbedder(self.client, cfg['embedding_model'], self.usage,
                                               dimensions=cfg.get('embedding_dimensions'),
                                               cache_dir=resolve(cfg.get('cache_dir', 'cache')))
        if method == 'dense':
            return DenseRetriever(self.store, self.embedder)
        chunks = load_chunks(cfg, self.store)
        if method == 'bm25':
            return BM25Retriever(chunks, **cfg['bm25'])
        if method == 'tfidf':
            return TfidfRetriever(chunks)
        if method in ('hybrid', 'hybrid_rerank'):
            base = RRFRetriever([DenseRetriever(self.store, self.embedder), BM25Retriever(chunks, **cfg['bm25'])],
                                k=cfg['rrf_k'], candidates=cfg['candidates'])
        elif method == 'dense_rerank':
            base = DenseRetriever(self.store, self.embedder)
        else:
            raise ValueError(f'method không hỗ trợ: {method}')
        if method == 'hybrid':
            return base
        return LLMRerankRetriever(base, self.client, cfg['rerank_model'], self.usage,
                                  candidates=cfg['candidates'], reasoning_effort=cfg['reasoning_effort'])
