# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng  
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng — đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt  
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Kết quả 4 phản hồi thực tế từ API theo từng mức temperature:**

- **Temperature = 0.0:**
  > Một sự thật thú vị về Việt Nam là: **cánh đồng lúa nước ở đồng bằng sông Cửu Long (Mekong) được xem là “cái nón xanh” của cả thế giới**.
  > - Khi nhìn từ trên cao, các thửa ruộng lúa tạo ra những dải màu xanh lục và vàng óng ả, giống như những chiếc nón lá truyền thống của người Việt.
  > - Nơi đây không chỉ cung cấp hơn một nửa lương thực tiêu thụ trong nước mà còn xuất khẩu gạo sang hơn 100 quốc gia, khiến Việt Nam trở thành một trong những “cỗ máy” cung cấp gạo quan trọng nhất thế giới.
  > - Ngoài ra, hệ thống kênh rạch và đê điều phức tạp của đồng bằng sông Cửu Long được xây dựng thủ công qua nhiều thế hệ...

- **Temperature = 0.5:**
  > Một sự thật thú vị về Việt Nam là nước này là một trong những nhà sản xuất cà phê lớn nhất thế giới và được biết đến với loại cà phê đặc trưng mang tên **cà phê sữa đá** (còn gọi là "cà phê sữa đá" hay "cà phê phin đá").
  > - **Cà phê Robusta**: Việt Nam chiếm khoảng 30% sản lượng cà phê Robusta toàn cầu, đứng sau Brazil nhưng vượt qua hầu hết các nước khác.
  > - **Cách pha truyền thống**: Người Việt thường dùng phin (bộ lọc kim loại) để pha cà phê, tạo ra một lớp bọt “cà phê kem” dày đặc trên bề mặt.
  > - **Kết hợp sữa đặc**: Sau khi pha, cà phê nóng được đổ trực tiếp lên sữa đặc có đường...

- **Temperature = 1.0:**
  > Việt Nam là một trong số ít các quốc gia trên thế giới có **địa hình đa dạng cực kỳ phong phú** chỉ trong một diện tích tương đối nhỏ: từ dãy núi đá vôi trùng điệp ở miền Bắc, những thảo nguyên và rừng nguyên sinh ở Tây Nguyên, cho đến đồng bằng sông Hậu rộng lớn và những bãi biển dài trải dài hơn 3.000 km. Điều đặc biệt là **cùng một ngày, bạn có thể thấy tuyết rơi trên đỉnh Fansipan (đỉnh núi cao nhất Đông Nam Á) và sau đó nhanh chóng di chuyển xuống bãi biển Nha Trang để tắm nắng** – một trải nghiệm hiếm có ở bất kỳ quốc gia nào khác...

- **Temperature = 1.5:**
  > Việt Nam có một trong những hệ thống hang động độc đáo nhất thế giới: Hang Sơn Đoòng, được khai thác vào năm 1991 nhưng chỉ thực sự được khám phá và công khai vào năm 2009. Đây là hang động lớn nhất hành lang của hành lang (cũng gọi là "cave within a cave") trên hành tinh, dài khoảng 5 km, cao tới 200 m và rộng tới 150 m. Khi vào trong, bạn có thể thấy “đại dương” của ánh sáng mặt trời chiếu qua những cửa sổ đá tự nhiên, tạo ra các vệt sáng lung linh, và thậm chí còn có một con sông chảy trong lòng hang!...

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)  
Khi tăng dần temperature từ 0.0 lên 1.5, tính ngẫu nhiên và độ phân tán của quá trình lấy mẫu token tăng rõ rệt: ở mức 0.0, mô hình hoạt động theo cơ chế gần như tất định (greedy/deterministic), tập trung vào các sự thật phổ biến với số liệu quy chuẩn và câu từ chặt chẽ; ở mức 0.5 đến 1.0, không gian phân phối xác suất mở rộng giúp câu trả lời đa dạng chủ đề hơn (từ cà phê chuyển sang địa hình khí hậu), hành văn sinh động và giàu cảm xúc hơn; tuy nhiên ở mức 1.5, phân phối xác suất bị làm phẳng (flattened) dẫn đến hiện tượng câu từ lặp lại bất thường ("hành lang của hành lang"), ngữ pháp lỏng lẻo và tăng nguy cơ ảo giác (hallucination).

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**  
Đối với chatbot hỗ trợ khách hàng (Customer Support), tôi sẽ đặt temperature trong khoảng **0.0 đến 0.2** (tối đa là 0.3). Nguyên nhân là vì ứng dụng hỗ trợ khách hàng đòi hỏi tính chính xác, tính nhất quán (consistency) và độ tin cậy tuyệt đối; mức temperature thấp giúp mô hình bám sát thông tin từ cơ sở tri thức (Knowledge Base/FAQ), tuân thủ nghiêm ngặt chính sách doanh nghiệp, định dạng dữ liệu (JSON/bullet points) và giảm thiểu tối đa hiện tượng "bịa đặt" thông tin (hallucination) có thể gây ảnh hưởng nghiêm trọng đến trải nghiệm người dùng hoặc trách nhiệm pháp lý.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,  
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**  
- **Tính toán chi phí cho workload:**
  - Tổng số request mỗi ngày = $10.000 \times 3 = 30.000$ requests/ngày.
  - Tổng số output tokens mỗi ngày = $30.000 \times 350 = 10.500.000$ tokens = $10.500$ đơn vị (1K tokens).
  - Đơn giá output theo bảng giá: GPT-4o là $0.010 / 1K token; GPT-4o-mini là $0.0006 / 1K token.
  - Chi phí GPT-4o mỗi ngày = $10.500 \times \$0.010 = \$105.00$/ngày ($\approx \$3.150$/tháng).
  - Chi phí GPT-4o-mini mỗi ngày = $10.500 \times \$0.0006 = \$6.30$/ngày ($\approx \$189$/tháng).
  - **Tỷ lệ chênh lệch:** GPT-4o đắt hơn GPT-4o-mini đúng **16.67 lần** ($\frac{0.010}{0.0006} \approx 16.67$).
- **Đánh đổi sử dụng:**
  - **Trường hợp GPT-4o xứng đáng chi phí:** Các tác vụ suy luận nhiều bước phức tạp (complex multi-step reasoning), phân tích tài chính/pháp lý có rủi ro cao, trích xuất cấu trúc dữ liệu khắt khe đòi hỏi độ chuẩn xác tuyệt đối, hoặc hỗ trợ lập trình kiến trúc hệ thống lớn.
  - **Trường hợp nên dùng GPT-4o-mini:** Các tác vụ thông lượng cao (high-volume) lặp đi lặp lại như chatbot trả lời câu hỏi thường gặp (FAQ), phân loại ý định người dùng (intent classification), tóm tắt tin tức/email ngắn, hoặc tiền xử lý lọc dữ liệu trước khi gửi sang model lớn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi  
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:  
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."  
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Kết quả phản hồi thực tế từ API theo từng persona:**

- **Persona 1 (Giáo viên tiểu học cho trẻ 8 tuổi):**
  > Hãy tưởng tượng em có một cuốn sổ nhật ký rất đặc biệt, và trong cuốn sổ này, mỗi trang chỉ có thể viết một lần và không thể xóa hay thay đổi gì nữa. Khi ai đó muốn ghi lại một việc gì đó (ví dụ: “Mẹ đã mua cho em một chiếc bánh”), mọi người đều phải đồng ý rằng việc này đúng rồi mới được viết vào trang mới. Khi trang mới đã được viết xong, nó sẽ được gắn chặt vào các trang trước bằng một “điểm nối” rất mạnh, sao cho không thể tách rời hay thay đổi bất kỳ trang nào.
  > - Mỗi trang → một khối (block).
  > - Các trang nối lại → một chuỗi (chain).

- **Persona 2 (Chuyên gia tài chính với thuật ngữ kỹ thuật):**
  > Blockchain (chuỗi khối) là một **sổ cái phân tán (distributed ledger)** bất biến, được duy trì bởi một mạng lưới các nút (nodes) đồng thuận thông qua các **cơ chế đồng thuận (consensus algorithms)** như Proof-of-Work (PoW), Proof-of-Stake (PoS), Byzantine Fault Tolerance (BFT) hay các biến thể hybrid. Mọi giao dịch được nhóm thành các khối có gắn nhãn thời gian (timestamp), mã hóa băm mật mã học (cryptographic hash) liên kết khối trước thông qua cây Merkle (Merkle tree), loại bỏ điểm lỗi đơn lẻ (SPOF) và đảm bảo tính toàn vẹn dữ liệu phi tập trung.


### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`  
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Thực nghiệm trên đoạn văn mẫu tiếng Việt (113 từ):**
> *"Việt Nam là một quốc gia nằm ở khu vực Đông Nam Á với bề dày lịch sử hàng nghìn năm dựng nước và giữ nước. Đất nước hình chữ S sở hữu đường bờ biển dài hơn 3260 km cùng với cảnh quan thiên nhiên đa dạng từ đồng bằng màu mỡ đến núi non hùng vĩ. Nền kinh tế Việt Nam đang trên đà phát triển mạnh mẽ và hội nhập sâu rộng với thế giới, đặc biệt là trong lĩnh vực công nghệ thông tin và trí tuệ nhân tạo. Người dân Việt Nam luôn cần cù, hiếu học và thân thiện với bạn bè quốc tế khắp năm châu."*

- Số từ thực tế: **113 từ**
- Số token đo bằng `tiktoken` (`o200k_base`): **122 token**
- Số token ước lượng theo công thức Part 1 (`113 / 0.75`): **150.7 token**

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**  
Hai con số chênh lệch nhau khoảng **23.5%** ($\frac{|122 - 150.7|}{122} \times 100\% \approx 23.5\%$). Công thức thô `số từ / 0.75` dựa trên giả định tiếng Anh (1 từ $\approx$ 1.33 token), trong khi bộ tokenizer mới (`o200k_base`) đã nén tiếng Việt tốt hơn đáng kể (gần 1.08 token/từ) so với các bộ mã hóa cũ (`cl100k_base` thường tốn tới 1.5 - 2 token/từ). Tiếng Việt thường tốn nhiều token hơn tiếng Anh cùng độ dài vì các thuật toán Byte-Pair Encoding (BPE) được huấn luyện chủ yếu trên kho ngữ liệu tiếng Anh, nơi hầu hết từ vựng thông dụng được lưu trọn vẹn trong từ điển token (1 từ = 1 token); ngược lại, tiếng Việt có cấu trúc âm tiết đơn lập ghép thanh điệu và chứa các ký tự UTF-8 đa byte (2–3 bytes/ký tự có dấu), buộc tokenizer phải chia nhỏ từ thành nhiều subword tokens hoặc byte tokens riêng lẻ.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)  
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
**Streaming** thường quan trọng nhất trong các tình huống cần xử lý dữ liệu liên tục và thời gian thực, ví dụ:

- **Phát video/âm thanh trực tuyến** (Netflix, YouTube, Zoom) – người dùng mong muốn xem ngay mà không chờ tải toàn bộ file.  
- **Xử lý dữ liệu sensor hoặc IoT** – các thiết bị gửi luồng dữ liệu liên tục (đo nhiệt độ, vị trí, cảm biến công nghiệp) và hệ thống phải phản hồi nhanh.  
- **Phân tích log, giao dịch tài chính** – phát hiện gian lận, giám sát hệ thống cần cập nhật ngay khi dữ liệu mới tới.  
- **Mô hình AI inference thời gian thực** – ví dụ nhận diện khuôn mặt, dịch vụ chatbot, cần trả lời ngay sau khi đầu vào tới.

**Khi nào nên dùng streaming?**  
- Khi dữ liệu **quá lớn** để lưu trữ và xử lý batch truyền thống.  
- Khi yêu cầu **độ trễ thấp** (seconds hoặc milliseconds).
- Khi nguồn dữ liệu **liên tục**, không có điểm dừng rõ ràng.  
- Khi muốn **cập nhật kết quả liên tục** cho người dùng hoặc hệ thống downstream.

Nếu dữ liệu chỉ là một batch nhỏ, không yêu cầu thời gian thực, thì xử lý batch thường đơn giản và hiệu quả hơn.
non-streaming lại phù hợp hơn?** (1 đoạn văn)
Nếu dữ liệu chỉ là một batch nhỏ, không yêu cầu thời gian thực, thì xử lý batch thường đơn giản và hiệu quả hơn.
Đúng vậy! Với batch nhỏ và không cần phản hồi ngay, việc xử lý theo lô giúp triển khai dễ dàng, tài nguyên tiết kiệm và tối ưu hoá hiệu suất. 🚀
non-streaming lại phù hợp hơn?** (1 đoạn văn)
Nếu khối lượng dữ liệu chỉ là một batch nhỏ, không cần phản hồi tức thời và có thể chờ tới thời điểm xử lý (ví dụ vào cuối ngày hoặc khi tài nguyên rảnh), thì mô hình **non‑streaming** sẽ phù hợp hơn vì nó đơn giản hoá kiến trúc, giảm chi phí quản lý luồng dữ liệu liên tục và cho phép tối ưu hoá tài nguyên bằng cách chạy các job batch vào những khoảng thời gian ít tải. Điều này giúp triển khai nhanh, bảo trì dễ dàng và vẫn đáp ứng đủ yêu cầu độ trễ của bài toán.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**  
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
**Exponential backoff** (độ trễ tăng dần) có những ưu điểm so với **delay cố định** (ví dụ luôn chờ 1 giây):

1. **Giảm tải cho hệ thống**  
   Khi lỗi lặp lại, thời gian chờ kéo dài dần nên không gây “cúi đầu” (thundering herd) – nhiều client không cùng gửi lại yêu cầu cùng lúc, giảm áp lực lên server.

2. **Tăng khả năng thành công**  
   Các lỗi tạm thời (network glitch, quá tải tạm thời) thường tự hồi phục sau một khoảng thời gian. Độ trễ lâu hơn giúp tài nguyên phục hồi, tăng tỉ lệ yêu cầu sau lần retry thành công.

3. **Tiết kiệm tài nguyên**  
   Không cần duy trì các kết nối hoặc retry liên tục trong thời gian ngắn, giảm chi phí CPU, băng thông và bộ nhớ.

4. **Độ linh hoạt**  
   Có thể điều chỉnh **base delay**, **hệ số tăng** và **giới hạn max delay** để phù hợp với từng loại dịch vụ hoặc mức độ nghiêm trọng của lỗi.

5. **Tránh “đánh trúng” thời gian lỗi**  
   Với delay cố định, các retry luôn diễn ra vào cùng thời điểm, có thể trùng với chu kỳ lỗi (ví dụ server đang trong vòng reset). Exponential backoff “dịch chuyển” thời gian retry, giảm khả năng gặp lại lỗi ngay lập tức.

**Tóm lại:** exponential backoff giúp hệ thống chịu tải tốt hơn, giảm nguy cơ tắc nghẽn và tăng khả năng hoàn thành yêu cầu khi gặp lỗi tạm thời, trong khi delay cố định đơn giản nhưng có thể gây quá tải và giảm hiệu suất.
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
### Khi API bị quá tải  
1. **Tài nguyên cạn kiệt** – CPU, bộ nhớ, kết nối mạng, hoặc cơ sở dữ liệu không đáp ứng kịp các yêu cầu đồng thời.  
2. **Thời gian phản hồi tăng** → client nhận được **timeout** hoặc **502/503**.  
3. **Giảm chất lượng dịch vụ** – người dùng cuối cảm thấy chậm, có thể bỏ qua hoặc rời khỏi ứng dụng.  

### Nếu hàng nghìn client **cùng retry** (delay cố định)  
- **Hiệu ứng “thundering herd”**: mọi client lại gửi lại yêu cầu cùng lúc ở thời điểm cố định (ví dụ mỗi 1 s).  
- Lúc đó, lượng request lại **bùng nổ** mạnh hơn so với lúc ban đầu, khiến server **đổ thêm tải** và **càng quá tải hơn**.  
- Kết quả: nhiều request tiếp tục bị từ chối → vòng lặp retry vô hạn, làm tồi tệ hơn nữa.  

### Cách giảm thiểu với **exponential backoff + jitter**  
| Biện pháp | Tác dụng |
|-----------|----------|
| **Exponential backoff** (trễ tăng dần) | Giảm số request đồng thời; mỗi vòng retry chậm lại, cho server thời gian hồi phục. |
| **Jitter (randomness)** | Thêm một phần ngẫu nhiên vào thời gian chờ, tránh việc tất cả client “đồng thời” lại gửi lại ở cùng một thời điểm. |
| **Giới hạn max delay** | Ngăn thời gian chờ trở nên vô hạn, đồng thời giữ cho client không phải chờ quá lâu. |
| **Circuit breaker** | Khi lỗi quá nhiều, tạm thời “cắt” các retry và trả về lỗi nhanh cho client, tránh làm thêm tải cho server. |
| **Rate limiting / queuing** | Kiểm soát lưu lượng vào API, cho phép server xử lý theo tốc độ ổn định. |

### Tóm lại
- Khi API quá tải, việc **đồng thời retry** của hàng nghìn client sẽ làm tình trạng tồi tệ hơn (thundering herd).  
- **Exponential backoff + jitter** phân tán lại các lần retry, cho phép server có thời gian hồi phục và giảm nguy cơ “đổ” lại toàn bộ lưu lượng.  
- Kết hợp thêm **circuit breaker** và **rate limiting** sẽ bảo vệ cả phía client lẫn server khỏi vòng lặp lỗi kéo dài.
với delay cố định giống nhau?**

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**  
Tôi chọn persona trợ giảng kỹ thuật cho trợ lý của mình với system prompt:  
`"Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt, tập trung vào bản chất kỹ thuật và giải thích dễ hiểu."`  
Hai lựa chọn từ ngữ quan trọng trong prompt:  
1. **"trả lời ngắn gọn"**: Đây là chỉ thị ràng buộc cực kỳ quan trọng trên giao diện dòng lệnh (CLI); nó giúp tiết kiệm tối đa số token output sinh ra, giảm độ trễ phản hồi (latency), hạn chế chi phí API và ngăn chặn tình trạng LLM trả lời lan man dài dòng gây khó đọc cho người dùng terminal.  
2. **"bằng tiếng Việt"**: Ràng buộc ngôn ngữ hiển thị cố định giúp mô hình duy trì giao tiếp thuần Việt đồng nhất, ngăn chặn trường hợp mô hình tự động chuyển sang trả lời bằng tiếng Anh khi người dùng nhập câu hỏi ngắn hoặc chứa các thuật ngữ chuyên ngành/đoạn code tiếng Anh.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**  
Hạn chế lớn nhất của trợ lý hiện tại là cơ chế cửa sổ trượt chỉ lưu tối đa 3 lượt hội thoại gần nhất (tương đương 6 messages); điều này khiến trợ lý mắc chứng "mất trí nhớ ngắn hạn" khi cuộc trò chuyện kéo dài, quên hoàn toàn các yêu cầu hay dữ kiện mà người dùng đã thiết lập ở các lượt đầu, đồng thời mất sạch ngữ cảnh khi tắt phiên CLI.  
**Đề xuất cải thiện cụ thể:** Triển khai cơ chế **"Tóm tắt ngữ cảnh tự động" (Rolling Context Summarization)** kết hợp **Lưu trữ phiên (Session Persistence)**:  
- *Cách triển khai:* Khi danh sách lịch sử vượt quá 3 lượt, thay vì xóa bỏ hoàn toàn các tin nhắn cũ, hệ thống sẽ kích hoạt một lời gọi ngầm sử dụng model nhỏ giá rẻ (`gpt-4o-mini`) để cô đọng nội dung các lượt đối thoại cũ thành một đoạn tóm tắt ngắn gọn (`conversation_summary`). Đoạn tóm tắt này được tích hợp vào `system prompt` của các lượt chat tiếp theo dưới dạng thông tin bối cảnh nền. Đồng thời, toàn bộ ngữ cảnh và tóm tắt được lưu vào tệp JSON/SQLite cục bộ theo Session ID, cho phép người dùng khôi phục lại phiên trò chuyện khi khởi động lại ứng dụng.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
