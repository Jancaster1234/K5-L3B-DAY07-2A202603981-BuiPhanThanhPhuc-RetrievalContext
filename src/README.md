# Code Day 7 — Agentic RAG trên vector store Day 6

Báo cáo: [report/individual/2A202603981_BuiPhanThanhPhuc.md](../report/individual/2A202603981_BuiPhanThanhPhuc.md) · Slide trình bày: <https://claude.ai/artifact/DYSgXZo1P3w3JbpbmEs8nc>

Chạy từ thư mục gốc repo, Python 3.11+ (đã chạy với 3.12.0, Windows 10). Trên Windows dùng `python -X utf8`: test `tests/test_baseline.py` của starter đọc file không chỉ định encoding nên lỗi với cp1252.

## 1. Cài đặt

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu   # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pip install -r requirements.txt
cp .env.example .env      # điền OPENAI_API_KEY; DATABASE_URL đã có giá trị local của Day 6. Không commit .env
.venv/Scripts/python -X utf8 -m unittest discover -s tests -v      # 30 test, mock — chỉ kiểm tra code
```

Lần đầu chạy dense sẽ tải `BAAI/bge-small-en-v1.5` (~130 MB) từ Hugging Face; sau đó có thể đặt `HF_HUB_OFFLINE=1`.

## 2. Vector store Day 6 (PostgreSQL + pgvector)

Dense retrieval đọc **trực tiếp** bảng `chunks` (strategy `B`, 317 chunk, vector 384 chiều) của repo Day 6:
<https://github.com/Jancaster1234/K5-L3B-DAY06-BuiPhanThanhPhuc-2A202603981-DataIngestion> (commit `9aa4e0e`).

```bash
# trong thư mục starter/ của repo Day 6 (cần Docker Desktop đang chạy)
docker compose -p day6_student_starter up -d --wait
python scripts/check_setup.py        # "Setup OK"
# nếu volume trống: python scripts/prepare_corpus.py && python -m src.run   (tạo lại A/B/C, embedding local, $0)
```

Query embedding dùng cùng model, tiền tố BGE và chuẩn hóa L2 như Day 6 (`src/embeddings.py::LocalSTEmbedder`); `PgVectorStore` kiểm tra model và số chiều của index khi khởi tạo. Kiểm chứng: dense Day 7 cho top-3 trùng 22/22 câu với trace strategy C của Day 6.

**Không có Docker?** `data/day6_chunks_B.json` là bản xuất 317 chunk (CC BY 3.0 IGO, nguồn ở repo Day 6 `starter/data/raw/ATTRIBUTION.md`). Thêm `"chunks_path": "data/day6_chunks_B.json"` vào config để chạy `bm25`/`tfidf` và kiểm tra gold mà không cần database; dense/hybrid vẫn cần pgvector.

## 3. Thành phần

| File | Vai trò |
|---|---|
| `config.py` | Đọc `configs/*.json` (`extends` kế thừa `base.json`), OpenAI client, đếm token/chi phí (`PRICES`, USD/1M token — kiểm tra trang giá OpenAI) |
| `stores.py` | `PgVectorStore` (Day 6, dùng thật), `NumpyStore`, `ChromaStore` (dự phòng, không dùng trong kết quả) |
| `embeddings.py` | `LocalSTEmbedder` (bge-small, dùng thật), `OpenAIEmbedder` (dự phòng) |
| `retrievers.py` | `dense`, `bm25` (tùy chọn bigram), `tfidf` (starter), `hybrid` (RRF dense+BM25), `*_rerank` (LLM chấm 0–3; đã cài, chưa benchmark) |
| `controller.py` | Controller OpenAI tool calling (`search` / `answer` / `abstain`), prompt `base` hoặc `strict`, cho `starter/agentic.run_agent` |
| `build_benchmark.py` | Gộp câu Day 6 + `data/day7_new_questions.json` → `data/benchmark.json`, kiểm tra evidence nguyên văn trên trang |
| `run_retrieval_eval.py` | Hit@1, Evidence Hit@3, MRR, Context Evidence Rate với **query gốc** |
| `run_agent_bench.py` | Agent thật; ghi `traces.json` (mọi tool call, **query do agent viết lại**, kết quả, context, câu trả lời, citation, token, độ trễ, lỗi), `cases.json`, `summary.json`, `reviews.todo.json` |
| `compare.py` | Gom `results/*/summary.json` thành bảng markdown |
| `build_index.py` | Dự phòng: tạo `NumpyStore` từ chunk (không dùng trong kết quả) |

Gold chỉ được dùng trong `evaluation.py` **sau** khi chạy; agent và retriever chỉ nhận câu hỏi và `filter` do người dùng cung cấp.

## 4. Tái lập kết quả trong `results/`

```bash
export HF_HUB_OFFLINE=1   # PowerShell: $env:HF_HUB_OFFLINE=1

# Benchmark (đã có sẵn data/benchmark.json; lệnh này tạo lại và kiểm tra evidence)
python -X utf8 -m src.build_benchmark --day6 <Day6>/benchmark/questions.json --pages <Day6>/starter/data/pages.jsonl

# Retrieval trên dev (miễn phí, model local)
for c in dense bm25 bm25_bigram tfidf hybrid dense_b3000 hybrid_b3000; do
  python -X utf8 -m src.run_retrieval_eval --config configs/$c.json --split dev; done

# Agent trên dev (~$0.04 mỗi cấu hình với gpt-5-mini)
for c in dense dense_b3000 hybrid_b3000 hybrid_b3000_strict; do
  python -X utf8 -m src.run_agent_bench --config configs/$c.json --split dev; done

# Cấu hình đã chốt: configs/selected.json (= hybrid_b3000), commit 5b32e6c trước khi chạy test
for c in dense selected; do
  python -X utf8 -m src.run_retrieval_eval --config configs/$c.json --split test
  python -X utf8 -m src.run_agent_bench   --config configs/$c.json --split test; done

# Tổng hợp nhãn (reviews.json chấm theo BENCHMARK.md)
python -X utf8 starter/agent_metrics.py --cases results/agent-test-hybrid_b3000/cases.json \
  --traces results/agent-test-hybrid_b3000/traces.json --reviews results/agent-test-hybrid_b3000/reviews.json \
  --split test --output results/agent-test-hybrid_b3000/metrics.json
python -X utf8 -m src.compare
```

Agent dùng LLM nên chạy lại có thể khác vài câu (không tất định); `results/` là lần chạy được báo cáo. `run_agent_bench` không ghi đè `reviews.json` đã có. `context_evidence_rate_auto` là số tự động (khớp nguyên văn gold trong context cuối), không thay cho việc chấm tay.
