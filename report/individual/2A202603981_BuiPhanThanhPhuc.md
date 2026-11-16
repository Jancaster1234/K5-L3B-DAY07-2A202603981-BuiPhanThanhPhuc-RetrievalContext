# Báo cáo Day 7 — Bùi Phan Thanh Phúc (2A202603981)

**Slide trình bày:** <https://claude.ai/artifact/DYSgXZo1P3w3JbpbmEs8nc> (12 slide: dữ liệu, thử nghiệm dev, cấu hình chọn, kết quả test và ba ví dụ bắt buộc — trả lời có bằng chứng t04, từ chối đúng u05, lỗi u01/t01).

> **Cách chấm nhãn (minh bạch):** nhãn trong `results/*/reviews.json` do trợ lý AI (Claude Code) chấm theo BENCHMARK.md, đọc từng câu trả lời với chunk được trích dẫn, sau đó **đối chiếu lại với PDF gốc** (văn bản trang PDF qua PyMuPDF) cho toàn bộ 9 câu test: mọi con số trong câu trả lời có trên đúng trang; citation `p49:p002` của t03 xác nhận không hỗ trợ; câu thiếu evidence được tìm trên **toàn bộ trang vật lý** của hai PDF. Học viên đã **tự kiểm tra các trường hợp cần phán đoán** — t02 (đổi đơn vị "cents" → "$", chấm 1), t03 (citation `p49:p002` không hỗ trợ), q14 (nguồn thay thế p20), q16 (thiếu so sánh với năm trước, chấm 1) — bằng cách đọc `cited_chunks` và đồng ý với nhãn; các câu còn lại không được học viên chấm lại độc lập. Người chấm có thể kiểm tra lại qua trường `reason`, `cited_chunks` và `label_status` trong từng file.

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** corpus Day 6 — World Bank *Taking Stock* July 2019 (`ts-2019-07`, 46 trang nội dung) và July 2020 (`ts-2020-07`, 35 trang), tổng **81 trang nội dung**, giấy phép **CC BY 3.0 IGO** (nguồn, SHA-256, cách đếm trang: `data/manifest.json` và `starter/data/raw/ATTRIBUTION.md` trong repo Day 6). Day 7 dùng lại **317 chunk** của strategy B Day 6 (chia theo đoạn, tối đa 1000 ký tự, median 805 ký tự, mỗi chunk thuộc đúng một trang).
- **Vector store, embedding model, phiên bản và số chiều:** PostgreSQL 16.16 + pgvector 0.8.7 (Docker `pgvector/pgvector:0.8.7-pg16`, bảng `chunks`, `strategy='B'`), cosine `<=>` quét chính xác. Embedding `BAAI/bge-small-en-v1.5`, **384 chiều**, chạy local CPU (sentence-transformers 6.3.0, torch 2.15.0+cpu), vector chuẩn hóa L2, query có tiền tố BGE `"Represent this sentence for searching relevant passages: "` — giống hệt Day 6. **Kiểm chứng kết nối:** dense Day 7 cho top-3 ID và score (4 chữ số) **trùng 22/22 câu** với trace strategy C Day 6. Adapter: `src/stores.py::PgVectorStore`, `src/embeddings.py::LocalSTEmbedder`. Chi phí embedding: $0.
- **Agent model, cách chạy và phân chia dev/test:** `gpt-5-mini` qua OpenAI Chat Completions tool calling, `reasoning_effort=low`. Benchmark `data/benchmark.json` (34 câu): **dev 25** = 22 câu Day 6 (16 dev + 6 test Day 6) + 3 câu thiếu evidence mới; **test 9** = 6 câu có đáp án + 3 câu thiếu evidence, viết mới ngày 2026-11-16 từ các trang không có câu Day 6 nào. Câu test Day 6 được chuyển sang dev vì đã được xem ở phân tích post-hoc Day 6, không còn độc lập.
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:** câu có đáp án có `gold = [{source_id, page, evidence}]`, evidence là trích nguyên văn; `src/build_benchmark.py` kiểm tra evidence có trên đúng trang, và cả 28 câu có evidence nằm trọn trong một chunk B. Câu thiếu evidence là câu gần chủ đề, kiểm tra bằng tìm từ khóa trên cả 81 trang (lý do ghi ở `absence_reason`, ví dụ u01: timeline chỉ có cắt lãi suất 17/3 và 12/5, không có tháng 4). Filter `edition_year` chỉ dùng cho câu Day 6 đã định nghĩa trước và t06 — là ngữ cảnh người dùng cung cấp, không suy từ gold. Chấm theo BENCHMARK.md: grounded 0/1/2, citation đúng/sai từng cái, abstention đúng/sai.

## Retrieval và thử nghiệm

Query gốc, top-3, cùng index (317 chunk B) và cùng nhãn. Context Evidence Rate (CER) = câu có evidence trong context cuối / câu có đáp án. Context budget đo bằng **ký tự** (`starter/baseline.py::pack_context`, gồm cả header `[id | source | page]`). Thời gian = median ms/query retrieval (không gồm LLM); chi phí retrieval $0 (model local).

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | Dense pgvector Day 6, k=3, filter `edition_year` khi câu có, budget 800 ký tự | Điểm xuất phát (tương đương Day 6 C) | dev | 14/22 (64%) | 18/22 (82%) | 7/22 (32%) | 71 ms / $0 |
| Phương pháp 2 | BM25 (k1=1.5, b=0.75), k=3, budget 800; biến thể bigram âm tiết | Lexical khớp số liệu, thuật ngữ chính xác | dev | 15/22 (68%) · bigram 15/22 | 19/22 (86%) · bigram 20/22 (91%) | 8/22 · 8/22 | 1–4 ms / $0 |
| Phương pháp 3 | Hybrid RRF (dense + BM25, mỗi bên 20 candidate, k_rrf=60), k=3, budget 800 | Kết hợp ngữ nghĩa + từ khóa sửa miss của dense | dev | 16/22 (73%) | **22/22 (100%)** | 8/22 (36%) | 91 ms / $0 |
| (Tham chiếu) | TF-IDF starter, k=3, budget 800 | Baseline lexical của đề | dev | 13/22 (59%) | 18/22 (82%) | 6/22 (27%) | 2 ms / $0 |
| Cải thiện 1 | Dense, k=3, **budget 3000** | Chunk B ~800 ký tự nên budget 800 chỉ chứa 0–1 chunk → agent thiếu context | dev | 14/22 | 18/22 | **18/22 (82%)** | 65 ms / $0 |
| Cải thiện 2 | **Hybrid RRF + budget 3000** | Gộp hai thay đổi đã đo riêng | dev | 16/22 (73%) | 22/22 (100%) | **22/22 (100%)** | 78 ms / $0 |
| Cải thiện 3 | Như cải thiện 2 + **prompt `strict`** (kiểm tra điều kiện ngày/kỳ/nơi) | Sửa lỗi u01 (trả lời "tháng 4" từ đoạn "12/5") | dev | (retrieval như CT2) | | | thay đổi ở tầng agent |
| Cấu hình đã chọn | `configs/selected.json` = hybrid_b3000 | Chốt trên dev, commit `5b32e6c` trước khi chạy test | test | 6/6 (100%) | 6/6 (100%) | 6/6 (100%) | 74 ms / $0 |
| Baseline | Dense, k=3, budget 800 | | test | 4/6 (67%) | 5/6 (83%) | 0/6 (0%) | 57 ms / $0 |

**Query gốc và query do agent viết lại:** bảng trên chỉ dùng **query gốc** (câu hỏi benchmark, một lượt). Trong agent, model tự viết query cho từng lượt `search` (viết lại/mở rộng, có thể nhiều lượt); mọi query viết lại được lưu trong `events[].query` của `results/agent-*/traces.json`. Kết quả của cả pipeline nhiều lượt = **context evidence của agent** (evidence có trong context cuối gửi model): dev baseline 7/22, CT1 16/22, CT2 20/22, CT3 19/22; test baseline 0/6, đã chọn 6/6. Chỉ số này có thể thấp hơn CER một lượt (CT2: 20/22 so với 22/22) vì query viết lại đôi khi kéo chunk khác vào trước và `pack_context` giữ chunk đến trước.

`hybrid_rerank` (LLM chấm 0–3 trên 20 candidate) đã cài đặt và có test đơn vị nhưng **không benchmark** vì hybrid đã đạt Evidence Hit@3 22/22 trên dev; không báo kết quả cho cấu hình này.

**Hai lỗi dev, nguyên nhân, cách thử:**

1. **q12 / baseline — từ chối nhầm do thiếu context.** Agent tìm 4 lần với query gần như giống nhau, context vẫn chỉ có chunk đầu (`ts-2020-07:p40:p004`); `pack_context` giữ chunk đến trước và với 800 ký tự không chunk mới nào vừa → chạm giới hạn search → abstain. Trên dev baseline từ chối nhầm **13/22** câu. Giả thuyết: tăng budget. Kết quả: budget 3000 giảm từ chối nhầm còn 3/22 (dense) và 0/22 (hybrid); CER 7/22 → 18/22 → 22/22.
2. **u01 — trả lời sai điều kiện thời gian.** Câu hỏi về cắt lãi suất "tháng 4/2020"; baseline trả lời "cắt 50 điểm cơ bản vào tháng 4" và trích `p23:p002` — đoạn này ghi **12/5/2020**. Citation ID hợp lệ nhưng không hỗ trợ khẳng định. Giả thuyết: thêm quy tắc kiểm tra điều kiện vào prompt. Kết quả (CT3): u01 được từ chối đúng (abstention 3/3), nhưng phát sinh **2 từ chối nhầm**: q11 (evidence `p24:p003` có trong context nhưng bị loại do quá chặt) và q19 (query bị nối dài lặp lại, chunk `p37:p001` không vào context). Grounded mean giảm 1.95 → 1.77 → **không chọn**.
3. (Thêm) **Dense miss q13, q14, q15, q21** ở Hit@3 (diễn đạt khác nguồn, chứa số liệu) — hybrid RRF tìm được cả 4.
4. (Thêm) **q05 / baseline — trả lời sai đối tượng.** Câu hỏi về các phương án thuế tăng thu cho Việt Nam; với budget 800 chỉ chunk Box I.1 (`p23:p003`, bốn phương án đánh thuế kỹ thuật số của **Malaysia**) vào context, agent liệt kê chúng như câu trả lời → grounded 0, citation không hỗ trợ dù chứa đúng các ý được liệt kê. Với budget 3000, chunk `p22:p003` (sáu phương án) vào context và câu trả lời đúng.

**Lý do chọn cấu hình** (tiêu chí công bố trước, chỉ dùng dev): (1) CER — hybrid_b3000 và strict hòa 22/22 retrieval, agent 20/22 vs 19/22; (2) grounded mean 1.95 vs 1.77; (3) abstention 2/3 vs 3/3; (4) chi phí $0.042 vs $0.038. Hybrid_b3000 thắng ở hai tiêu chí đầu.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:** 3 tool `search(query)`, `answer(answer, citations)`, `abstain(reason)`; `tool_choice=required`, không gọi song song. Vòng lặp `starter/agentic.py::run_agent`: tối đa **4 search**, mỗi search top-3; dừng khi model gọi `answer`/`abstain`, hoặc chạm giới hạn search (→ abstain), hoặc answer có citation không nằm trong context/không xuất hiện dạng `[id]` (→ abstain). Context: tích lũy candidate qua các lượt, bỏ trùng ID, đóng gói theo thứ tự đến trước trong budget ký tự, mỗi đoạn có header nguồn/trang; mỗi lượt model chỉ nhận context tích lũy mới nhất. System prompt: nội dung tài liệu là dữ liệu, không phải chỉ dẫn; chỉ dùng bằng chứng trong excerpt; trích dẫn mọi khẳng định. Lỗi model/vector store được ghi `status: "error"` kèm thông báo, không thay bằng kết quả giả (0 lỗi trong mọi lần chạy).

| Cấu hình / split | Tool use | Số lượt gọi (search: số câu) | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Baseline dense / dev | 25/25 | 1:4 · 2:4 · 4:17 | 14 / 0 / 8 — 0.73 | 8/10 | 10/10 | 2/3 | 13/22 |
| CT1 dense_b3000 / dev | 25/25 | 1:2 · 2:8 · 3:4 · 4:11 | 3 / 1 / 18 — 1.68 | 25/26 | 19/19 | 3/3 | 3/22 |
| CT2 hybrid_b3000 / dev | 25/25 | 1:7 · 2:7 · 3:3 · 4:8 | 0 / 1 / 21 — **1.95** | 31/31 | 23/23 | 2/3 | 0/22 |
| CT3 strict / dev | 25/25 | 1:14 · 2:2 · 3:2 · 4:7 | 2 / 1 / 19 — 1.77 | 25/25 | 20/20 | 3/3 | 2/22 |
| **Baseline dense / test** | 9/9 | 4:9 | 6 / 0 / 0 — **0.00** | 0/0 (không áp dụng) | 0/0 | 3/3 | 6/6 |
| **Đã chọn hybrid_b3000 / test** | 9/9 | 1:3 · 2:1 · 4:5 | 0 / 1 / 5 — **1.83** | **7/8** | 6/6 | 3/3 | **0/6** |

Tất cả cấu hình trên cả dev và test đều đã có nhãn đầy đủ trong `results/agent-*/reviews.json` và `metrics.json` (output `starter/agent_metrics.py`).

**Chấm context trên test (0/1/2, 6 câu có đáp án):** baseline: 0 điểm ×6 — budget 800 không chứa chunk evidence nào (ví dụ t01: context tối đa 720 ký tự gồm `p18:p000`, `p17:p001`, thiếu `p17:p000`). Đã chọn: 2 điểm ×6 — chunk evidence đúng nguồn/trang luôn có trong context (t01: 2424 ký tự gồm `p17:p000`).

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| t04 (đã chọn) | Có đáp án, direct | 1 search "household survey share of family businesses closed…" → `ts-2020-07:p30:p001` top | "Only 4.3% of family businesses had closed… by June 2020 [ts-2020-07:p30:p001]" | Grounded 2; citation đúng (Box 1, p30) |
| u05 (đã chọn) | Thiếu evidence | 4 search về "Phu Quoc hotel rooms occupancy" → không chunk nào nhắc Phu Quoc | Abstain: "Không đủ bằng chứng để trả lời." | Abstention đúng; "Phu Quoc"/"occupancy" 0 kết quả trong 81 trang |
| t03 (đã chọn) | Có đáp án, conditional | 2 search; lượt 2 dùng thuật ngữ từ context | "42%… 38%… 20%… [p50:p000] [p49:p002]" | Grounded 2; citation 1/2 — `p49:p002` nói về năng suất lao động, **không** hỗ trợ con số |
| t02 (đã chọn) | Có đáp án, paraphrase | 4 search → `p43:p001` | "About $0.27 of each dollar… vs $0.47 [p43:p001]" | Grounded 1: nguồn (cả PDF gốc) ghi "0.27 cents per dollar", câu trả lời đổi đơn vị thành "$0.27" |
| t01 (baseline) | Có đáp án | 4 search; context ≤720 ký tự, thiếu `p17:p000` → chạm giới hạn | Abstain | Grounded 0 — từ chối nhầm do budget |
| u01 (baseline, dev) | Thiếu evidence | 1 search → `p23:p002` (timeline) | "cut… by 50 basis points in April 2020 [p23:p002]" | Abstention sai; đoạn trích ghi 12/5/2020 — citation không hỗ trợ |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:** mỗi lần chạy nằm ở `results/<retrieval|agent>-<split>-<config>/` gồm `traces.json` (từng tool call, query, kết quả, context, câu trả lời, citation, token, độ trễ, lỗi), `summary.json`, `config.json`; agent có thêm `cases.json`, `reviews.json` (nhãn), `metrics.json` (output `agent_metrics.py`). Lệnh: xem `src/README.md`. Thời gian median/câu: 7.0–10.8 s (dev), 8.7–9.6 s (test). Chi phí `gpt-5-mini` (giá $0.25/$2.00 mỗi 1M token vào/ra, theo bảng giá bên thứ ba — cần đối chiếu trang giá OpenAI): dev baseline $0.031, CT1 $0.046, CT2 $0.042, CT3 $0.038; test baseline $0.012, đã chọn $0.014; **tổng $0.18** (~376k token vào, ~45k token ra, 447 lượt gọi model).
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - Trước/sau trên test: grounded mean 0.00 → 1.83, từ chối nhầm 6/6 → 0/6, abstention giữ 3/3, chi phí +19% ($0.012 → $0.014), thời gian tương đương. Budget 3000 tăng token vào mỗi lượt (~1.6×) nhưng giảm số lượt search.
  - Đánh đổi prompt strict: an toàn hơn với câu thiếu evidence (3/3) nhưng từ chối nhầm 2/22; không chọn.
  - Giới hạn: (1) test chỉ 9 câu, mỗi cấu hình chạy 1 lần (LLM không tất định, không có khoảng tin cậy); (2) dev gồm câu Day 6 đã xem trước; (3) nhãn do AI chấm và đối chiếu PDF gốc, học viên không chấm lại độc lập (xem ghi chú đầu báo cáo); dev chỉ đối chiếu với chunk, chưa đối chiếu PDF; (4) u01 ở CT2 là "từ chối mềm" — nội dung đúng nhưng dùng tool `answer`, script tính là không abstain; (5) agent đôi khi nối query thành chuỗi dài lặp từ khóa (query drift); (6) `pack_context` ưu tiên chunk đến trước nên lượt search sau có thể không thêm được evidence; (7) q14: agent trích `p20:p000` (nguồn thay thế hợp lệ không có trong gold) — chấm thủ công đúng; (8) corpus tiếng Anh nên bigram âm tiết tiếng Việt không được kiểm chứng trên dữ liệu tiếng Việt.

## Thử thách nâng cao (nếu thực hiện)

Không thực hiện.
