# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi tăng `temperature`, phản hồi có xu hướng đa dạng hơn về cách diễn đạt và ví dụ được chọn. Ở mức 0.0, câu trả lời thường ổn định, trực tiếp và ít biến thể; đến 1.0–1.5, model sáng tạo hơn nhưng cũng dễ dài dòng hoặc đưa ra chi tiết ít chắc chắn hơn. Vì vậy, temperature là tham số đánh đổi giữa tính nhất quán và sự đa dạng, đúng như cách hàm `call_openai()` truyền trực tiếp giá trị này vào API.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi chọn temperature khoảng 0.2 cho chatbot hỗ trợ khách hàng. Mục tiêu của chatbot là trả lời nhất quán về chính sách, giá, vận chuyển và đổi trả; temperature thấp giúp giảm sự biến thiên không cần thiết giữa các lần hỏi. Tuy nhiên, temperature thấp không tự bảo đảm câu trả lời đúng, nên hệ thống thực tế vẫn cần dữ liệu nguồn đáng tin cậy hoặc cơ chế tra cứu tài liệu.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Mỗi ngày có `10.000 × 3 × 350 = 10.500.000` output token, tương đương 10.500 đơn vị 1K token. Theo bảng `PRICING_PER_1K_TOKENS`, GPT-4o tốn khoảng `$105/ngày` ở chiều output, còn GPT-4o-mini tốn `$6,30/ngày`; GPT-4o đắt hơn khoảng `16,67 lần` và chênh `$98,70/ngày` (chưa tính input token). GPT-4o đáng dùng cho các tác vụ cần suy luận và chất lượng cao như phân tích tài liệu phức tạp; mini phù hợp cho FAQ, phân loại ý định hoặc tóm tắt số lượng lớn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Với persona giáo viên tiểu học, câu trả lời về blockchain nên ngắn, dùng từ đời thường và ví dụ gần gũi, chẳng hạn so sánh với một cuốn sổ chung khó bị sửa lén. Với persona chuyên gia tài chính, câu trả lời có thể dài hơn và dùng các khái niệm như sổ cái phân tán, cơ chế đồng thuận, chữ ký số hoặc tính bất biến. Trong `chat_with_system_prompt()`, system message được gửi trước user message, nên nó định hướng giọng điệu, mức độ chuyên sâu và kiểu ví dụ mà model dùng để trả lời cùng một câu hỏi.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Tôi dùng đoạn văn tiếng Việt 120 từ; `count_tokens(..., model="gpt-4o")` cho kết quả 270 token, trong khi ước lượng Part 1 là `120 / 0,75 = 160` token. Như vậy số đếm bằng tokenizer cao hơn khoảng `68,75%`. Tiếng Việt có dấu, ký tự Unicode và nhiều tổ hợp âm tiết có thể bị tách thành nhiều sub-token hơn tiếng Anh, nên ước lượng theo số từ khá thiếu chính xác. Lưu ý: khi dùng Gemini, `tiktoken` chỉ là ước lượng theo tokenizer OpenAI chứ không phải số token thanh toán chính thức của Gemini.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất với chatbot hoặc trợ lý tương tác trực tiếp, vì người dùng thấy các phần đầu của câu trả lời ngay thay vì chờ toàn bộ nội dung hoàn thành. Trong code, mỗi `chunk.choices[0].delta.content` được in ngay, đồng thời được ghép thành `reply`; vì vậy streaming cải thiện cảm nhận về tốc độ phản hồi, dù không nhất thiết làm model tạo xong toàn bộ câu nhanh hơn. Non-streaming phù hợp hơn cho xử lý nền, xử lý theo lô, hoặc khi cần kiểm tra toàn bộ JSON/kết quả trước khi chuyển sang bước tiếp theo.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff tăng thời gian chờ theo từng lần lỗi, ví dụ `0,1 → 0,2 → 0,4` giây trong `retry_with_backoff()`. Cách này giảm số request gửi lại trong lúc API đang quá tải và tạo thời gian để server hồi phục. Nếu hàng nghìn client cùng retry với delay cố định một giây, chúng có thể cùng gửi request lại tại một thời điểm, tạo ra “thundering herd” và tiếp tục làm server quá tải; trong hệ thống thật thường nên thêm random jitter để giảm đồng bộ này.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> System prompt tôi chọn là: “Bạn là trợ giảng lập trình AI thực chiến. Hãy trả lời bằng tiếng Việt, ngắn gọn, chính xác, giải thích theo từng bước và chỉ đưa ví dụ code tối giản khi cần.” Cụm “ngắn gọn” giúp giảm câu trả lời lan man, giảm token output và phù hợp với giao diện terminal. Cụm “theo từng bước” và “ví dụ code tối giản” giúp người mới có thể áp dụng được ngay, thay vì chỉ nhận giải thích lý thuyết.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là lịch sử bị cắt cứng bằng `history[-6:]`, nên assistant chỉ giữ ba lượt hỏi–đáp gần nhất và có thể quên yêu cầu hoặc dữ liệu quan trọng từ đầu cuộc hội thoại. Tôi sẽ bổ sung một biến `summary`: khi history vượt ngưỡng, gọi một model nhỏ để tóm tắt các lượt cũ, sau đó gửi `system prompt + summary + các lượt gần nhất` trong request mới. Cách này giữ được ngữ cảnh dài hơn nhưng phải đánh đổi thêm chi phí, độ trễ và rủi ro bản tóm tắt làm mất chi tiết.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
