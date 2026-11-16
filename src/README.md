# Code Day 7 — Agentic RAG

Chạy từ thư mục gốc repo, Python 3.11+. Trên Windows dùng `python -X utf8` (test starter đọc file không chỉ định encoding).

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt   # Linux/macOS: .venv/bin/python
cp .env.example .env        # điền OPENAI_API_KEY; không commit .env
python -X utf8 -m unittest discover -s tests -v
```

## Thành phần

| File | Vai trò |
|---|---|
| `config.py` | Đọc `configs/*.json` (`extends` kế thừa `base.json`), OpenAI client, đếm token và chi phí (`PRICES`, USD/1M token — kiểm tra lại trên trang giá OpenAI) |
| `stores.py` | Vector store: `NumpyStore` (cosine chính xác, filter theo metadata) và `ChromaStore` (nếu Day 6 dùng Chroma) |
| `embeddings.py` | Query/chunk embedding OpenAI, cache trên đĩa (`cache/`) |
| `build_index.py` | Tạo `index/` từ chunk Day 6: dùng lại vector Day 6 (`--embeddings`) hoặc embed lại |
| `retrievers.py` | `dense`, `bm25` (tùy chọn bigram âm tiết tiếng Việt), `tfidf`, `hybrid` (RRF dense+BM25), `*_rerank` (LLM chấm 0–3 trên candidate) |
| `controller.py` | Controller OpenAI tool calling (`search` / `answer` / `abstain`) cho `starter/agentic.run_agent` |
| `run_retrieval_eval.py` | Hit@1, Evidence Hit@3, MRR, Context Evidence Rate, thời gian, chi phí với query gốc |
| `run_agent_bench.py` | Chạy agent thật; ghi `traces.json`, `cases.json`, `summary.json`, `reviews.todo.json` |
| `compare.py` | Gom `results/*/summary.json` thành bảng markdown |

Gold chỉ được dùng trong `evaluation.py` **sau** khi chạy; agent và retriever chỉ nhận câu hỏi và `filter` do người dùng cung cấp.

## Quy trình

```bash
# 1. Index (một lần)
python -X utf8 -m src.build_index --chunks data/chunks.json --output index --model text-embedding-3-small

# 2. Retrieval trên dev, cùng index và nhãn
for c in dense bm25 bm25_bigram hybrid hybrid_rerank; do python -X utf8 -m src.run_retrieval_eval --config configs/$c.json --split dev; done

# 3. Agent trên dev
python -X utf8 -m src.run_agent_bench --config configs/dense.json --split dev

# 4. Chấm tay: chép results/agent-dev-dense/reviews.todo.json -> reviews.json, điền answer_grade /
#    citation_support / abstention_correct sau khi đọc chunk được trích dẫn, rồi:
python -X utf8 starter/agent_metrics.py --cases results/agent-dev-dense/cases.json \
  --traces results/agent-dev-dense/traces.json --reviews results/agent-dev-dense/reviews.json \
  --split dev --output results/agent-dev-dense/metrics.json

# 5. Chốt configs/selected.json và commit TRƯỚC khi chạy --split test (baseline + selected)
python -X utf8 -m src.compare
```

`context_evidence_rate_auto` trong summary agent là số tự động (khớp nguyên văn gold trong context cuối), không thay cho việc chấm tay.
