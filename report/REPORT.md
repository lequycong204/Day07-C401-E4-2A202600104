# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Lê Quý Công <br>
**Nhóm:** C401-E4 <br>
**Ngày:** 11/04/2026 <br>

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là hai vector có hướng gần giống nhau, tức là hai câu có ý nghĩa tương tự nhau.

**Ví dụ HIGH similarity:**
- Sentence A: "Tôi thích ăn phở"
- Sentence B: "Phở là món ăn yêu thích của tôi"
- Tại sao tương đồng: Cả hai câu đều nói về việc thích ăn phở, chỉ khác cách diễn đạt.

**Ví dụ LOW similarity:**
- Sentence A: "Tôi thích ăn phở"
- Sentence B: "Tôi thích ăn bún chả"
- Tại sao khác: Cả hai câu đều nói về việc thích ăn, nhưng một câu nói về phở, một câu nói về bún chả.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity đo lường góc giữa hai vector, không phụ thuộc vào độ lớn của vector. Trong khi đó, Euclidean distance đo lường khoảng cách giữa hai vector, phụ thuộc vào độ lớn của vector. Do đó, cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Số chunks = `ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = 23`.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Nếu overlap tăng lên 100 thì số chunk tăng lên vì bước nhảy `chunk_size - overlap` nhỏ hơn. Overlap lớn hơn giúp giữ ngữ cảnh tốt hơn giữa các chunk liền kề, giảm nguy cơ mất ý ở ranh giới chunk.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** AI Engineering book - Chip Huyen

**Tại sao nhóm chọn domain này?**
> Cuốn sách này là một tài liệu tham khảo toàn diện về AI Engineering, bao gồm nhiều chủ đề khác nhau như RAG, LLM, prompt engineering, evaluation, deployment, v.v. Cuốn sách cũng có cấu trúc rõ ràng, dễ đọc, dễ hiểu, phù hợp với nhiều đối tượng độc giả rất phù hợp cho việc học tập và nghiên cứu cho khoá AI In Action.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | AI Engineering book - Chip Huyen | `data/ai_engineer.md` | ~500+ pages markdown, chunked by page marker | `{ "source": "data/ai_engineer.md", "extension": ".md", "page_id": 1, "chunk_id": 1, "doc_id": "ai_engineer_p1_c1" }` |


### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `source` | `str` | `data/ai_engineer.md` | Cho biết chunk đến từ file nào, hữu ích khi trace nguồn và debug retrieval |
| `extension` | `str` | `.md` | Phân biệt loại tài liệu nguồn để xử lý hoặc lọc theo định dạng |
| `page_id` | `int \| None` | `1` | Gắn chunk với trang logic lấy từ marker `<!-- page: N -->`, giúp truy vết và hỗ trợ filter theo vùng nội dung |
| `chunk_id` | `int` | `3` | Xác định thứ tự chunk bên trong một trang, giúp kiểm tra chunk nào được retrieve |
| `doc_id` | `str` | `ai_engineer_p1_c3` | ID chuẩn hóa của chunk/tài liệu để phục vụ delete, trace ngược và quản lý trong store |


---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| `ai_engineer.md` (~50K chars đầu) | FixedSizeChunker (`fixed_size`) | 111 | 500.0 | Trung bình |
| `ai_engineer.md` (~50K chars đầu) | SentenceChunker (`by_sentences`) | 595 | 79.5 | Cao ở mức câu, nhưng chunk rất nhỏ |
| `ai_engineer.md` (~50K chars đầu) | RecursiveChunker (`recursive`) | 132 | 360.2 | Tốt |

### Strategy Của Tôi

**Loại:** Custom strategy

**Mô tả cách hoạt động:**
> Strategy của tôi tách file markdown theo page marker `<!-- page: N -->` trước, để mỗi phần nội dung đều gắn được `page_id`. Sau đó mỗi page được chia tiếp bằng `RecursiveChunker` với ưu tiên separator `["\n\n", "\n", ". ", " ", ""]`, nghĩa là cố giữ ranh giới đoạn, dòng, câu trước khi phải cắt nhỏ hơn. Mỗi chunk được gán thêm `chunk_id`, còn `doc_id` được chuẩn hóa trong store để phục vụ delete và trace ngược. Cách này giúp retrieval vừa giữ được ngữ cảnh tự nhiên, vừa biết chính xác chunk đến từ trang nào.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> File `ai_engineer.md` đã có sẵn page marker nên rất phù hợp để khai thác metadata theo page. Cuốn sách cũng có cấu trúc dạng markdown với nhiều đoạn và section rõ ràng, nên recursive chunking tận dụng tốt các ranh giới tự nhiên hơn là cắt cứng theo ký tự.

**Code snippet (nếu custom):**
```python
def _split_markdown_pages(content: str) -> list[tuple[int | None, str]]:
    pattern = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")
    ...

chunks = RecursiveChunker(
    chunk_size=1200,
    separators=["\n\n", "\n", ". ", " ", ""],
).chunk(page_content)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| `ai_engineer.md` | best baseline: `RecursiveChunker` | 132 | 360.2 | Tốt |
| `ai_engineer.md` | **của tôi**: page-aware recursive | 1217 chunks toàn bộ sách | phụ thuộc từng page | Tốt hơn về traceability, dễ debug hơn |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | Custom page-aware recursive | 8 | Có `page_id`, `chunk_id`, retrieval dễ trace | Chưa có field `part/chapter` nên filter theo chương còn hạn chế |
| Trương Đăng Gia Huy | Custom heading-aware | 10 | Metadata rất giàu, filter theo `part="chapterN"` trực tiếp | Parser heading có thể sinh chunk rác |
| Phạm Đỗ Ngọc Minh | SentenceChunker | 8 | Semantic coherence tốt ở mức câu | Chunk quá nhỏ, thiếu metadata cấu trúc chương |
| Nguyễn Ngọc Thắng | RecursiveChunker | 10 | Giữ ngữ cảnh đoạn tốt, hợp markdown | Có thể tách mất ý nếu context trải dài nhiều chunk |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Với domain là sách markdown có cấu trúc rõ, strategy tốt nhất trong nhóm là heading-aware custom của Huy vì nó vừa giữ được ngữ nghĩa theo section, vừa hỗ trợ metadata filter theo chapter trực tiếp. Tuy nhiên, strategy page-aware recursive của tôi vẫn thực dụng và dễ triển khai hơn vì tận dụng được page marker sẵn có mà không cần parser heading phức tạp.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> `SentenceChunker.chunk` dùng regex `(?<=[.,!?\n])\s+` để tách sau các dấu kết câu hoặc xuống dòng. Sau khi tách, các câu được gom lại thành chunk theo `max_sentences_per_chunk`. Edge case chính là text rỗng thì cần trả về list rỗng, và việc split theo dấu câu đơn giản có thể tách quá mạnh với một số pattern viết tắt.

**`RecursiveChunker.chunk` / `_split`** — approach:
> `RecursiveChunker.chunk` gọi `_split` với danh sách separator ưu tiên từ lớn đến nhỏ. Base case là khi đoạn hiện tại ngắn hơn `chunk_size` thì trả luôn một chunk, còn nếu hết separator thì cắt cứng theo độ dài. Với mỗi separator, thuật toán cố gắng ghép các phần nhỏ lại thành chunk không vượt quá `chunk_size`; phần nào vẫn quá dài thì đệ quy xuống separator nhỏ hơn.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents` chuẩn hóa metadata, thêm `doc_id`, tạo embedding cho từng document rồi lưu vào ChromaDB hoặc in-memory store. Trong nhánh Chroma, mỗi record có internal ID riêng để tránh đụng ID khi add nhiều lần. `search` embed query rồi so khớp với embedding đã lưu; với in-memory thì dùng dot product, còn với Chroma thì lấy `distances` rồi quy đổi thành `score`.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` ưu tiên filter trước bằng metadata rồi mới chạy similarity search trên tập đã thu hẹp. Với Chroma, filter được truyền vào `where`; với in-memory thì lọc record bằng metadata rồi mới tính score. `delete_document` xóa tất cả chunk có cùng `doc_id`, và trả về `False` nếu không tìm thấy document cần xóa.

### KnowledgeBaseAgent

**`answer`** — approach:
> `KnowledgeBaseAgent.answer` lấy top-k chunk từ store, nối phần `content` của các chunk này thành một khối context rồi inject vào prompt theo dạng `Context ... Question ... Answer`. Prompt hiện tại tối giản, chưa có citation bắt buộc, nhưng đủ để chứng minh luồng RAG cơ bản: retrieve trước, generate sau.

### Test Results

```
42 passed in 1.23s
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Retrieval augmented generation grounds LLM outputs in external data. | RAG retrieves context from a knowledge base before the model answers. | high | -0.019 | Không |
| 2 | Cosine similarity measures the angle between two vectors. | Dot product normalized by the product of magnitudes equals cosine. | high | -0.100 | Không |
| 3 | Prompt engineering modifies model behavior through instructions. | Fine-tuning updates model weights using extra training data. | low | 0.164 | Đúng |
| 4 | Vector databases store embeddings for similarity search. | ChromaDB can index vectors and retrieve nearest chunks. | high | 0.034 | Không |
| 5 | Language models predict the next token in a sequence. | Large language models are scaled-up language models trained on more data. | high | 0.192 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Cặp bất ngờ nhất là Pair 1 và Pair 2 vì về mặt nghĩa chúng gần như paraphrase hoặc cùng định nghĩa một khái niệm, nhưng score với mock embedding lại rất thấp, thậm chí âm. Điều này cho thấy embedding mock trong lab chỉ phù hợp để test pipeline và API, không phản ánh tốt semantic similarity thật như model embedding mạnh hơn.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | What is the difference between language models and large language models? | LLMs là language models được scale lớn hơn về data, parameters và compute, nên có khả năng tổng quát hơn |
| 2 | What are the three layers of the AI stack? | Ba lớp là application layer, model layer và infrastructure layer |
| 3 | How does RAG ground LLMs in external knowledge? | RAG retrieve context từ nguồn ngoài rồi chèn vào prompt để giảm hallucination |
| 4 | What is the difference between prompt engineering and fine-tuning? | Prompt engineering điều khiển model qua input; fine-tuning cập nhật trọng số model bằng dữ liệu huấn luyện thêm |
| 5 | What is LLM-as-a-judge and when is it useful? | Dùng model để chấm/đánh giá output của model khác; hữu ích cho evaluation, ranking và QA tự động |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Difference between LM and LLM | Chunk nói về "From Language Models to Large Language Models" | ~0.45 | Yes | Trả lời đúng ý chính nhưng còn ngắn |
| 2 | Three layers of AI stack | Chunk nói về kiến trúc nhiều lớp của AI stack | ~0.44 | Yes | Nêu được ba lớp, nhưng nên chạy thêm `search_with_filter` cho Chapter 1 |
| 3 | RAG grounds LLMs | Chunk mô tả retrieval context từ knowledge base | ~0.44 | Yes | Trả lời đúng ý "ground in external knowledge" |
| 4 | Prompt engineering vs fine-tuning | Chunk liên quan đến prompt engineering, top-k có thêm phần fine-tuning | ~0.43 | Yes | Trả lời tương đối đúng nhưng cần ghép nhiều chunk |
| 5 | LLM-as-a-judge | Chunk đúng chủ đề nhưng khái niệm bị phân tán ở nhiều chỗ | ~0.40 | No | Trả lời còn thiếu use cases và limitations |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Điều tôi học được rõ nhất từ bạn Huy là metadata schema tốt có thể nâng retrieval lên rất nhiều, đặc biệt khi có thể filter trực tiếp theo `part="chapterN"` hoặc theo section title. So với việc chỉ tối ưu chunk_size, việc gắn metadata đúng ngữ nghĩa đem lại hiệu quả rõ hơn.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Qua demo của nhóm khác, tôi thấy parsing và chuẩn hóa dữ liệu đầu vào quan trọng không kém chunking. Nếu PDF sang markdown bị nhiễu hoặc mất cấu trúc, retrieval sẽ giảm chất lượng dù embedding model mạnh.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Nếu làm lại, tôi sẽ bổ sung thêm field `part`, `chapter` hoặc `section_title` ngay từ bước index thay vì chỉ giữ `page_id`. Tôi cũng sẽ thêm một bước mapping page range sang chapter để Query 2 có thể dùng `search_with_filter` đúng như yêu cầu nhóm.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **90 / 90** |

