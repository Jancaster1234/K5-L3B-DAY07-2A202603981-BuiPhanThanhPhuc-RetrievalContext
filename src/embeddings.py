"""OpenAI embeddings with a disk cache so repeated experiments do not pay twice."""
import hashlib
import json
from pathlib import Path


BGE_QUERY_INSTRUCTION = 'Represent this sentence for searching relevant passages: '


class LocalSTEmbedder:
    """Local sentence-transformers model, identical to Day 6's query path (CPU, L2-normalised,
    BGE query instruction). Embeds queries only; document vectors already live in pgvector."""

    def __init__(self, model_name, usage=None):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device='cpu')
        self.dimensions = int(self.model.get_sentence_embedding_dimension())
        self.query_instruction = BGE_QUERY_INSTRUCTION if 'bge-' in model_name.lower() else ''
        self.usage = usage

    def embed(self, texts):
        vecs = self.model.encode([self.query_instruction + t for t in texts], normalize_embeddings=True,
                                 show_progress_bar=False, convert_to_numpy=True)
        if self.usage is not None:  # local model: tokens counted for the report, cost 0
            n = sum(len(self.model.tokenizer(t, add_special_tokens=False)['input_ids']) for t in texts)
            self.usage.add(self.model_name, n)
        return vecs.tolist()


class OpenAIEmbedder:
    def __init__(self, client, model, usage, *, dimensions=None, cache_dir=None, batch_size=100):
        self.client, self.model, self.usage = client, model, usage
        self.dimensions, self.batch_size = dimensions, batch_size
        self.cache = {}
        self.cache_file = None
        if cache_dir:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_file = cache_dir / f'emb-{model}-{dimensions or "native"}.jsonl'
            if self.cache_file.exists():
                for line in self.cache_file.read_text(encoding='utf-8').splitlines():
                    if line.strip():
                        row = json.loads(line)
                        self.cache[row['k']] = row['v']

    def _key(self, text):
        return hashlib.sha256(f'{self.model}|{self.dimensions}|{text}'.encode('utf-8')).hexdigest()

    def embed(self, texts):
        keys = [self._key(t) for t in texts]
        missing = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in self.cache]
        for i in range(0, len(missing), self.batch_size):
            batch = missing[i:i + self.batch_size]
            kwargs = {'model': self.model, 'input': [t for _, t in batch]}
            if self.dimensions:
                kwargs['dimensions'] = self.dimensions
            resp = self.client.embeddings.create(**kwargs)
            self.usage.add(self.model, resp.usage.prompt_tokens)
            rows = sorted(resp.data, key=lambda d: d.index)
            new = [{'k': k, 'v': d.embedding} for (k, _), d in zip(batch, rows)]
            for row in new:
                self.cache[row['k']] = row['v']
            if self.cache_file:
                with self.cache_file.open('a', encoding='utf-8') as f:
                    for row in new:
                        f.write(json.dumps(row) + '\n')
        return [self.cache[k] for k in keys]
