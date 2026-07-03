# S³ — Tách Tín hiệu Ngữ nghĩa để Phát hiện Chủ đề (Semantic Signal Separation)

> **Bản chắt lọc nội dung** dùng làm nguồn cho bộ slide thuyết trình ~45 phút (tiếng Việt).
> Bài báo gốc: *S³ - Semantic Signal Separation*, Kardos et al., Aarhus University, **ACL 2025 (Long Papers)**, trang 633–666.
> Link: https://aclanthology.org/2025.acl-long.32/
>
> **Quy ước:** Giải thích bằng tiếng Việt dễ hiểu; các thuật ngữ học thuật tiếng Anh (topic model, embedding, ICA, coherence...) được **giữ nguyên**. Mỗi mục lớn có 3 lớp:
> - 🧠 **Trực giác** — hiểu ý tưởng bằng lời thường
> - 🔢 **Kỹ thuật/Công thức** — phần chính xác để trình bày sâu
> - 🎤 **Câu nói khi thuyết trình** — gợi ý lời dẫn cho slide

---

## 0. Thông tin nhanh (dùng cho slide mở đầu / kết thúc)

| Mục | Nội dung |
|---|---|
| **Tên phương pháp** | S³ = Semantic Signal Separation (Tách Tín hiệu Ngữ nghĩa) |
| **Ý tưởng một câu** | Coi mỗi **topic** là một **trục ngữ nghĩa độc lập** (independent semantic axis) trong không gian embedding, và tìm các trục đó bằng **Independent Component Analysis (ICA)**. |
| **Nhóm tác giả** | Márton Kardos, Jan Kostkan, Kenneth Enevoldsen, Arnault-Quentin Vermillet, Kristoffer Nielbo, Roberta Rocca (Aarhus University) |
| **Điểm bán hàng chính** | Nhanh nhất trong các contextual topic model: **nhanh gấp 4.5× so với á quân BERTopic**, và trung bình **nhanh gấp 27.5× so với toàn bộ baselines**. |
| **Chất lượng** | Cân bằng tốt nhất giữa **coherence** (mạch lạc) và **diversity** (đa dạng); topic sạch, dễ hiểu. |
| **Ưu điểm độc đáo** | **Không cần preprocessing** (thậm chí chạy tốt hơn khi giữ nguyên văn bản thô); cho phép mô tả topic bằng cả từ **positive** lẫn **negative**. |
| **Công cụ** | Thư viện Python **Turftopic** (interface kiểu scikit-learn); benchmark trong package **topic-benchmark**. Giấy phép MIT. |

---

## PHẦN 1 — Đặt vấn đề: Topic model là gì và tại sao cần cái mới

### 1.1 Topic model là gì?
🧠 **Trực giác:**
"Topic model" là nhóm các phương pháp **thống kê** giúp **tự động khám phá chủ đề** ẩn trong một kho văn bản lớn, mà **không cần đọc từng bài** (unsupervised — không cần nhãn). Kết quả thường được trình bày cho người dùng dưới dạng **một tập các từ khoá quan trọng** cho mỗi topic, ví dụ topic về thể thao: `{bóng đá, cầu thủ, bàn thắng, huấn luyện viên...}`.

Ứng dụng điển hình: phân tích khám phá (exploratory analysis) một kho dữ liệu văn bản — tóm tắt "kho này nói về những gì" mà không cần lao động thủ công.

🎤 *"Hãy tưởng tượng bạn có 100.000 bài báo và muốn biết chúng nói về những chủ đề gì — topic model làm việc đó tự động, và trả về cho bạn vài chục nhóm từ khoá đại diện."*

### 1.2 Cách làm cổ điển và giới hạn của nó
🔢 **Các phương pháp cổ điển (classical):**
- **LSI / LSA** — Latent Semantic Indexing / Analysis (Deerwester 1988, Dumais 2004)
- **LDA** — Latent Dirichlet Allocation (Blei 2003)

Điểm chung: đều dựa trên biểu diễn **bag-of-words (BoW)** — đếm tần suất từ, bỏ qua thứ tự và ngữ cảnh.

**Giới hạn của BoW (rất quan trọng — đây là "kẻ thù" mà S³ muốn đánh bại):**
1. **Nhạy cảm với các từ có thống kê bất thường** — nhất là **function words / stop words** (từ chức năng: "the", "of", "và", "là"...). Chúng dễ lọt vào mô tả topic và làm topic vô nghĩa, **trừ khi có preprocessing nặng**.
2. **Preprocessing tạo ra quá nhiều "bậc tự do" cho nhà nghiên cứu** (researcher degrees of freedom) — mỗi người tiền xử lý một kiểu → khó tái lập, dễ thiên lệch kết quả.
3. **BoW thưa (sparse) và số chiều cao** → tính toán kém hiệu quả và mô hình khớp kém.

🎤 *"Cách cũ đếm từ. Vấn đề là đếm từ thì 'the', 'is', 'of' luôn xuất hiện nhiều nhất — nên bạn buộc phải làm sạch dữ liệu rất kỹ, và mỗi người làm sạch một kiểu khác nhau."*

### 1.3 Cơ hội mới từ neural embeddings
🧠 **Trực giác:** Sự ra đời của **dense neural representations** (Transformer — Vaswani 2017; sentence embeddings — Reimers & Gurevych 2019) mở ra hướng mới.

**Vì sao embedding hợp với topic modeling:**
- **Có ngữ cảnh, nhạy ngữ pháp** (contextual, grammar-sensitive) → không cần vứt bỏ thông tin ngôn ngữ quý giá qua preprocessing.
- **Bền với lỗi chính tả và từ ngoài từ điển** (out-of-vocabulary).
- **Biểu diễn dày (dense) trong không gian liên tục** → có thể giả định **Gaussianity** (phân phối chuẩn), thuận cho các phép phân rã toán học.
- Cho phép **transfer learning** — tận dụng kiến thức đã học từ kho dữ liệu ngoài lớn hơn.

### 1.4 Nhưng các contextual topic model hiện tại vẫn chưa ổn
Nhiều mô hình dùng embedding đã ra đời (CTM, BERTopic, Top2Vec, FASTopic...) và vượt các mô hình phi ngữ cảnh. **Tuy nhiên vẫn còn 3 vấn đề lớn** — đây chính là 3 điểm S³ nhắm giải quyết:
1. **Chậm, không ổn định (volatile).**
2. **Vẫn cần preprocessing nặng** để đạt kết quả tốt → mà pipeline preprocessing không chuẩn hoá, và việc bỏ thông tin đặc biệt hại với văn bản ngắn.
3. **Không rõ chúng có thực sự tận dụng được thông tin ngữ cảnh/cú pháp hay không**, vì thường chỉ được đánh giá trên dữ liệu đã tiền xử lý.

🎤 *"Các mô hình dùng embedding tốt hơn cách cũ, nhưng vẫn chậm, vẫn đòi tiền xử lý, và mỉa mai là ta không chắc chúng có dùng được cái 'ngữ cảnh' mà embedding mang lại không — vì người ta vẫn test chúng trên dữ liệu đã làm sạch mất ngữ cảnh."*

---

## PHẦN 2 — Đóng góp của bài báo (Contributions)

Bài báo giới thiệu **S³**, một kỹ thuật contextual topic modeling **mới**, quan niệm việc tìm topic = **khám phá các trục ngữ nghĩa ẩn (latent semantic axes)** trong kho văn bản. Các trục này được tìm bằng cách phân rã ma trận embedding tài liệu bằng thuật toán **FastICA**.

**4 tính chất then chốt của S³ (a–d):**
- **(a)** Đơn giản về mặt khái niệm và **có nền tảng lý thuyết** (theory-driven).
- **(b)** Ngang ngửa các phương pháp hiện tại về **word-embedding coherence**, và cho **diversity gần như hoàn hảo**.
- **(c)** **Hiệu quả tính toán cao hơn** tất cả phương pháp hiện có.
- **(d)** **Tận dụng hiệu quả thông tin ngữ cảnh** (contextual information).

Ngoài ra: cung cấp **interface thống nhất kiểu scikit-learn** cho cả S³ lẫn các contextual topic model khác trong package **Turftopic**.

🎤 *"Đóng góp gồm hai phần: một là phương pháp mới S³, hai là một thư viện Turftopic gói tất cả các mô hình lại dưới cùng một giao diện dễ dùng."*

---

## PHẦN 3 — Bối cảnh liên quan (Related Work) — dùng để so sánh

### 3.1 Semantic Axes (nền tảng ý tưởng)
🧠 ICA (Jutten & Herault 1991) từng được áp dụng lên không gian embedding để tìm **semantic axes** (Musil & Mareček 2024; Yamagiwa 2023). Các nghiên cứu đó cho thấy trục do ICA tìm ra **có thể diễn giải được** và thường **trùng nhau giữa các không gian embedding và các modality khác nhau**.

**Khác biệt của S³:** các nghiên cứu trước tìm **chiều ngữ nghĩa phổ quát (universal)** của embedding; còn S³ dùng semantic axes để **tìm topic dễ diễn giải trong một kho cụ thể** — và **có tính toán mô tả topic** (điều mà các nghiên cứu trước không làm).

### 3.2 Các nhóm embedding-based topic model (chính là baselines)
Chia thành các nhóm — nắm để so sánh trên slide:

**a) Neural Topic Models — dùng mạng nơ-ron để ước lượng tham số:**
- **CTM (Contextualized Topic Models)** — Bianchi 2021a. Là generative model của BoW nhưng dùng **variational autoencoder (VAE)** để inference.
  - **ZeroShotTM**: chỉ dùng contextual embedding làm input encoder.
  - **CombinedTM**: nối (concatenate) embedding với vector BoW.
  - ⚠️ CTM cần **preprocessing nặng**; chất lượng & tốc độ **giảm mạnh khi từ vựng lớn**.
- **ECRTM** — Wu 2023. Dùng **embedding clustering regularization** để buộc topic khác biệt nhau (tránh các mô tả topic trùng nhau). ⚠️ **Không dùng contextual representation** và **chậm** hơn hầu hết mô hình khác.
- **FASTopic** — Wu 2024b. Dùng mô hình **dual-semantic-relation**: quan hệ giữa document–topic–word được coi là các **optimal transport plans**. Hiệu quả và chất lượng cao hơn các neural approach trước đó.

**b) Clustering Topic Models — tìm topic bằng cách gom cụm (cluster) embedding tài liệu; trọng số từ được ước lượng *post hoc* (sau khi cluster):**
- **Top2Vec** — Angelov 2020. Tính độ quan trọng của từ bằng **cosine similarity** giữa embedding từ và **centroid cụm**. ⚠️ Giả định cụm **hình cầu & lồi (spherical, convex)** — nếu cụm méo thì mô tả topic sai lệch.
- **BERTopic** — Grootendorst 2022. Ước lượng độ quan trọng từ bằng **class-based tf-idf (c-TF-IDF)**.
- Cả hai đều dùng **UMAP** (giảm chiều) + **HDBSCAN** (clustering). Vì HDBSCAN tự học số cụm và số cụm có thể **rất lớn**, nên cả hai đi kèm **cơ chế giảm số topic (topic reduction)**.

**Thách thức chung của contextual topic model hiện tại:** nhạy hyperparameter, topic khó diễn giải, phụ thuộc preprocessing không chuẩn hoá.

📊 **Bảng baselines để so sánh (dùng trên slide):**

| Nhóm | Mô hình | Cơ chế cốt lõi | Contextual? |
|---|---|---|---|
| Classical | **LDA** | Xác suất trên BoW | Không |
| Classical | **NMF** | Phân rã ma trận BoW | Không |
| Neural / VAE | **ZeroShotTM** | VAE, input = embedding | Có |
| Neural / VAE | **CombinedTM** | VAE, embedding + BoW | Có |
| Neural | **ECRTM** | Clustering regularization | Không |
| Neural | **FASTopic** | Optimal transport | Có |
| Clustering | **Top2Vec** | UMAP + HDBSCAN + cosine | Có |
| Clustering | **BERTopic** | UMAP + HDBSCAN + c-TF-IDF | Có |
| **Đề xuất** | **S³** | **ICA trên embedding** | **Có** |

---

## PHẦN 4 — TRÁI TIM CỦA BÀI: Phương pháp S³ hoạt động thế nào

> Đây là phần quan trọng nhất, nên dành nhiều slide. S³ có thể xem là **"hậu duệ contextual của LSA"** — LSA tìm factor trong đồng-xuất-hiện của từ; S³ tìm axis trong không gian embedding.

### 4.1 Đổi cách quan niệm về "topic"
🧠 **Trực giác — đây là cú lật cách nghĩ:**
- BERTopic/Top2Vec coi topic = **cụm (cluster)** tài liệu.
- LDA/CTM coi topic = **phân phối xác suất trên từ**.
- **S³ coi topic = một TRỤC (axis) trong không gian ngữ nghĩa** — một hướng giải thích **phần biến thiên (variance) riêng của kho văn bản**.

Ta phân rã biểu diễn ngữ nghĩa thành:
- **A** = các thành phần ẩn (latent components) — chính là **các topic**.
- **S** = độ mạnh của mỗi component trong từng tài liệu — chính là **document-topic importances** (một tài liệu thuộc topic nào, mạnh yếu ra sao).

Để các topic **thực sự khác biệt về mặt khái niệm**, ta dùng **Independent Component Analysis (ICA)** để tìm chúng — ICA tìm các thành phần **độc lập thống kê** với nhau.

🎤 *"Thay vì hỏi 'tài liệu này thuộc cụm nào', S³ hỏi 'không gian ý nghĩa của kho này có những trục độc lập nào', và mỗi trục chính là một topic."*

### 4.2 Vì sao dùng ICA chứ không phải PCA?
🧠 **Trực giác quan trọng (nên có 1 slide riêng):**
- **PCA** tìm các trục **không tương quan (uncorrelated)** và giải thích variance lớn nhất — nhưng "không tương quan" là điều kiện yếu.
- **ICA** tìm các thành phần **độc lập thống kê (statistically independent)** — điều kiện mạnh hơn nhiều. Đây chính là bài toán **blind source separation** (tách nguồn mù) — ví dụ kinh điển "cocktail party": tách nhiều giọng nói đang trộn lẫn thành từng nguồn riêng.
- Với topic modeling: mỗi "tín hiệu nguồn" độc lập = một **topic** thuần khiết, ít chồng lấn → giải thích tại sao S³ cho **diversity gần như hoàn hảo**.

Đó là lý do tên gọi **"Semantic Signal Separation"** — tách các "tín hiệu ngữ nghĩa" trộn lẫn trong tài liệu thành các topic độc lập.

### 4.3 Các bước cụ thể của mô hình (phần công thức để trình bày kỹ)

🔢 **Ký hiệu & 6 bước huấn luyện:**

**Bước 1 — Encode tài liệu.**
Mã hoá mọi tài liệu bằng một **sentence transformer**. Gọi **X** = ma trận embedding của các tài liệu.

**Bước 2 — Phân rã X bằng FastICA:**
$$X = A \cdot S$$
- **A** = mixing matrix (ma trận trộn) → **các topic**.
- **S** = source matrix → **document-topic-importances**.

> **Chi tiết tiền xử lý bên trong FastICA (rất đáng nói):**
> - FastICA (Hyvärinen & Oja 2000) là mô hình **không nhiễu (noiseless)**, nên phải **whitening** (làm trắng) ma trận embedding trước.
> - Mặc định ICA cho ra số component = số chiều embedding. Để lấy đúng **N topic**, trong bước whitening ta **giảm chiều bằng cách lấy N principal components đầu tiên** (N = số topic mong muốn).
> - ⇒ Bên trong S³ thực chất có một bước giống PCA (để giảm chiều + whitening) rồi ICA mới xoay các trục cho **độc lập**.

**Bước 3 — Encode từ vựng.**
Mã hoá **toàn bộ từ vựng** của kho bằng **cùng encoder**. Gọi **V** = ma trận embedding của các từ.

**Bước 4 — Tính unmixing matrix:**
$$C = A^{+}$$
(pseudo-inverse của mixing matrix A).

**Bước 5 — Chiếu (project) từ lên các trục ngữ nghĩa:**
$$W = V \cdot C^{T}$$
→ **W** cho biết mỗi từ nằm ở đâu trên mỗi trục topic.

**Bước 6 — Tính điểm quan trọng của từ (word importance)** cho mỗi topic (xem 4.4).

🎤 *"Toàn bộ phần 'khó' chỉ là một phép phân rã ma trận: tài liệu = topic × độ mạnh. Không có mạng nơ-ron phải train, không có vòng lặp tối ưu tốn kém — đó là lý do nó nhanh."*

### 4.4 Ba cách tính "độ quan trọng của từ" cho topic (word importance)
🔢 Sau khi có **W** (W_jt = vị trí của từ *j* trên trục topic *t*), có **3 công thức** để chọn từ mô tả topic:

**1. Axial (theo trục) — độ quan trọng = vị trí trên trục:**
$$\beta_{tj} = W_{jt}$$
→ Cho các từ **nổi bật (salient)** nhất của topic. **Coherence cao nhất.**

**2. Angular (theo góc) — cosine của góc giữa vector từ và trục:**
$$\beta_{tj} = \cos(\Theta) = \frac{W_{jt}}{\lVert W_j \rVert}$$
→ Cho các từ **đặc trưng riêng (specific)** nhất cho topic. **Diversity cao nhất.**

**3. Combined (kết hợp) — cân bằng hai cái trên:**
$$\beta_{tj} = \frac{(W_{jt})^{3}}{\lVert W_j \rVert}$$
> Dùng **luỹ thừa lẻ (mũ 3)** để **giữ dấu** của vị trí từ (âm/dương).

📌 **Khuyến nghị của tác giả:** dùng **Combined làm mặc định** — nó ngăn một từ lọt vào nhiều topic cùng lúc (khi từ đó ghi điểm cao trên nhiều trục).

### 4.5 Mô tả topic bằng từ NEGATIVE — điểm độc đáo của S³
🧠 **Trực giác:** Cả 3 công thức trên đều cho phép từ có **độ quan trọng âm** cho một topic. Nghĩa là ngoài "top từ dương" (định nghĩa dương), S³ còn có thể liệt kê **top từ âm** → cho một **định nghĩa âm (negative definition)** của topic.

- LSA cũng có tính chất này về mặt toán, nhưng **chưa từng được khai thác** trong văn liệu trước.
- **Lợi ích:** hai topic có top-từ-dương giống nhau vẫn có thể được phân biệt rõ nhờ **từ âm** (xem ví dụ ArXiv ở Phần 7).
- ⚠️ Để so sánh công bằng với các mô hình không có khái niệm "từ âm", các so sánh định lượng trong bài **bỏ qua từ âm**; từ âm chỉ dùng trong phần minh hoạ định tính.

### 4.6 Suy luận cho tài liệu mới (inference)
🔢 Với tài liệu chưa từng thấy: chỉ cần **nhân embedding của nó với unmixing matrix**:
- Gọi **X̂** = embedding tài liệu mới.
- Ma trận document-topic: $$\hat{S} = \hat{X} \cdot C^{T}$$

→ Cực nhanh, chỉ là một phép nhân ma trận.

---

## PHẦN 5 — Thiết lập thí nghiệm (Experimental Setup)

### 5.1 Datasets (6 bộ dữ liệu)
| Dataset | # Tài liệu | Vocabulary Size |
|---|---|---|
| ArXiv ML Papers (abstract ML lấy ngẫu nhiên) | 2.048 | 2.849 |
| BBC News | 1.225 | 3.851 |
| 20 Newsgroups **Preprocessed** | 16.310 | 1.612 |
| 20 Newsgroups **Raw** (thô) | 18.846 | 21.668 |
| StackExchange | 75.000 | 17.884 |
| Wiki Medical (thuật ngữ y khoa từ Wikipedia) | 6.861 | 22.145 |

> Điểm tinh tế: dùng cả **20 Newsgroups thô** lẫn **đã tiền xử lý** để **đo ảnh hưởng của preprocessing** — một thí nghiệm then chốt (xem Phần 6.2).

### 5.2 Embedding Models (4 mô hình, đủ cỡ)
| Embedding Model | # Params | Embedding Size | Loại |
|---|---|---|---|
| Averaged GloVe | 120 M | 300 | **Static** (phi ngữ cảnh) |
| all-MiniLM-L6-v2 | 22.7 M | 384 | SBERT (nhỏ) |
| all-mpnet-base-v2 | 109 M | 768 | SBERT (vừa) |
| E5-large-v2 | 335 M | 1024 | E5 (lớn, chất lượng cao) |

→ Chạy tất cả phân tích với đủ 4 embedding để xem **chất lượng embedding ảnh hưởng thế nào**.

### 5.3 Cấu hình chạy
- **Baselines:** BERTopic, Top2Vec, ZeroShotTM, CombinedTM, FASTopic, ECRTM + 2 cổ điển NMF, LDA.
- Số topic thử: **10, 20, 30, 40, 50**.
- Lấy **top 10 từ** mỗi topic để đánh giá.
- **Không tune hyperparameter** — dùng tham số mặc định của package (lý do ở Phần 8).
- Mỗi cấu hình chỉ chạy **1 lần** (do chi phí tính toán khổng lồ của toàn bộ tổ hợp mô hình × encoder × dataset).

### 5.4 Các chỉ số đánh giá chất lượng topic (Metrics)
🔢 **Hai trục chính:**

**a) Topic Diversity (d)** — đo topic **khác nhau đến đâu**, dựa trên số từ chúng dùng chung. Diversity thấp = nhiều topic trùng từ = khó phân biệt ý nghĩa.

**b) Topic Coherence (C)** — đo topic **mạch lạc ngữ nghĩa** đến đâu. Ở đây dùng **word embedding coherence** = độ tương đồng trung bình từng cặp từ trong mô tả topic, dựa trên Word2Vec.
- **External coherence (C_ex):** dùng Word2Vec **train sẵn trên kho lớn ngoài** (word2vec-google-news-300) → bắt quan hệ ngữ nghĩa **tổng quát**.
- **Internal coherence (C_in):** dùng Word2Vec **train trên chính kho đang xét** → bắt quan hệ ngữ nghĩa **đặc thù kho**.
- **Coherence tổng hợp** = trung bình nhân (geometric mean):
$$\bar{C} = \sqrt{C_{ex} \cdot C_{in}}$$

**c) Chỉ số diễn giải tổng hợp (aggregate interpretability):**
$$\sqrt{\bar{C} \cdot d}$$
> Dùng **geometric mean** thay vì arithmetic mean, vì nó phạt nặng trường hợp lệch: một topic coherence 1.0 nhưng diversity 0.0 **không nên** được 0.5 (điểm này quan trọng — bắt buộc mô hình phải tốt cả hai mặt).

### 5.5 Chỉ số độ bền vững (Robustness) — đo "rác"
🧠 Chỉ số coherence/diversity chuẩn **không** bắt được việc topic bị nhiễm **"junk terms"** (từ rác). Nên tác giả thêm 2 proxy đo trên kho thô:
1. **Tần suất tương đối của stop words** trong mô tả topic.
2. **Tần suất từ chứa ký tự phi chữ cái (non-alphabetic)** — proxy cho từ rác.
   > Lưu ý: đây không phải proxy hoàn hảo — đôi khi từ phi chữ cái vẫn có nghĩa (vd "1917" trong topic về Cách mạng Tháng Mười).

---

## PHẦN 6 — KẾT QUẢ (Results) — phần "chốt hạ" trên slide

### 6.1 Kết quả tổng quát
✅ **S³ cân bằng hơn hẳn baselines**, thường **vượt trội về hiệu năng tổng hợp**, và trung bình **nhanh gấp 27.5× so với baselines** (median runtime ratio, ghép cặp theo dataset × encoder × số topic).

**So sánh tính cách từng mô hình (nên vẽ thành sơ đồ trên slide):**
- **ECRTM, FASTopic:** topic **đa dạng hơn** nhưng **kém mạch lạc**.
- **Top2Vec:** **rất mạch lạc** nhưng **kém đa dạng**.
- **S³:** **cân bằng tối ưu** giữa coherence và diversity → đứng đầu cả về hiệu năng tổng hợp lẫn (khá ổn định) về runtime.

### 6.2 Kiểm định thống kê (rất mạnh để thuyết phục)
🔢 Chạy **linear regression** dự đoán điểm interpretability tổng hợp ($\sqrt{\bar{C} \cdot d}$):
- **Fixed effect:** loại mô hình (với **S³_com làm intercept/mốc**).
- **Random intercepts:** số topic, encoder, dataset.
- **Kết quả:** loại mô hình **dự đoán interpretability có ý nghĩa** — **F = 167.4; p < 0.001; R² = 0.673**.
- **Tất cả** mô hình khác (trừ 2 biến thể còn lại của S³) đều có hệ số **âm và có ý nghĩa (p < 0.05)** → **S³ vượt trội có ý nghĩa thống kê** về interpretability.

📊 **Bảng hệ số hồi quy (Table 3) — mốc là S³_com = 0.6061; càng âm = càng kém S³:**

| Mô hình | Hệ số | p-value |
|---|---|---|
| **Intercept (S³_com)** | **0.6061** | <0.001 |
| S³_axi | −0.0005 | 0.963 (≈ ngang S³_com) |
| S³_ang | −0.0145 | 0.178 (≈ ngang) |
| ECRTM | −0.0308 | 0.004 |
| FASTopic | −0.0427 | <0.001 |
| Top2Vec | −0.0463 | <0.001 |
| ZeroShotTM | −0.1170 | <0.001 |
| CombinedTM | −0.1204 | <0.001 |
| BERTopic | −0.2141 | <0.001 |
| NMF | −0.2221 | <0.001 |
| LDA | −0.2722 | <0.001 |

→ Ba biến thể S³ đứng đầu, bỏ xa các mô hình cổ điển (LDA, NMF, BERTopic ở đáy về interpretability).

### 6.3 Ảnh hưởng của Preprocessing (phát hiện thú vị nhất)
🌟 **S³ là mô hình DUY NHẤT liên tục chạy TỐT HƠN trên văn bản thô (natural text) so với dữ liệu đã tiền xử lý.**
- Vài baseline có cải thiện coherence khi được dùng kho thô, nhưng nhìn chung hiệu năng **ngang hoặc kém đi** khi bỏ preprocessing.
- Các biến thể S³ **hưởng lợi nhiều nhất** khi bỏ preprocessing → chứng tỏ S³ **thực sự khai thác được thông tin thêm** trong văn bản thô.
- Đáng chú ý: khi có preprocessing nặng, S³ đôi khi bị vài baseline vượt; **nhưng trên kho thô, S³ cao hơn tất cả** — kể cả các mô hình được train trên dữ liệu đã tiền xử lý.

🎤 *"Đây là điểm phản trực giác nhất: mọi mô hình khác cần bạn làm sạch dữ liệu; S³ thì ngược lại — càng để nguyên văn bản, nó càng chạy tốt."*

### 6.4 Stop words
- Trên kho thô, nhiều mô hình nhét đầy stop words vào top-10 từ — nặng nhất là **BoW models và BERTopic** (đôi khi **100%** mô tả topic là stop words!).
- **CTM và FASTopic** khá hơn.
- **ECRTM, Top2Vec, và mọi biến thể S³** hầu như **không có stop words** trong mô tả topic.
- Với ký tự phi chữ cái: không thấy pattern rõ, các mô hình khá giống nhau.

### 6.5 Ảnh hưởng của Embedding Model
- **S³:** ổn định qua các loại embedding — và **cho topic chất lượng cao nhất với E5 (embedding lớn nhất)** → S³ **tận dụng được embedding chất lượng cao**.
- **Top2Vec:** bị ảnh hưởng **tệ nhất** — kém hẳn với GloVe và E5.
- **FASTopic:** tốt nhất với **GloVe**, nhưng **càng dùng embedding lớn càng kém** — do dính **curse of dimensionality** (chất lượng khớp giảm khi số chiều embedding tăng).

### 6.6 Ảnh hưởng của cách tính Term Importance (3 biến thể S³)
- Hiệu năng 3 phương pháp **khá gần nhau**, khác biệt chủ yếu là **đánh đổi coherence ↔ diversity**:
  - **Angular** → topic đa dạng hơn.
  - **Axial** → topic mạch lạc nhất.
  - **Combined** → ở giữa; thường chỉ khác Axial vài từ (⇒ từ rất liên quan thường cũng đủ đặc trưng cho trục).
- 📌 **Khuyến nghị: dùng Combined mặc định.**

---

## PHẦN 7 — Đánh giá định tính (Qualitative) — dùng ví dụ trực quan trên slide

### 7.1 So sánh chất lượng topic (20 Newsgroups, 20 topics)
🧠 Xếp hạng chủ quan về độ dễ hiểu của topic:

**❌ Kém nhất — LDA, NMF, BERTopic** (đầy function words, acronyms):
- LDA: `that, to, you, of, from, the, and, in, was, on`
- BERTopic/e5-large-v2: `the, of, to, in, space, it, edu, is, that, and`

**⚠️ Khá hơn nhưng còn nhiễu — CTM, ECRTM:**
- CombinedTM/MiniLM: `145, ax, 0d, _o, a86, mk, m3, mp, 0g, mm`
- ECRTM: `verbeek, billington, cassels, c5ff, nyr, det, bos, guerin, nieuwendyk, ashton`

**🙂 Tốt hơn, ít nhiễu — FASTopic** (nhưng đôi khi trộn 2 topic khác nhau vào một):
- `miles, dealer, auto, engine, ford, oil, cars, honda, toyota, mustang` (topic xe hơi — tốt)
- `moon, launch, henry, bike, medical, car, dod, orbit, shuttle, mission` (trộn lẫn không gian + xe + y tế)

**✅ Tốt nhất — Top2Vec và S³** (sạch, đặc trưng, dễ hiểu):
- S³_axi/mpnet (y khoa): `epilepsy, medical, toxins, medicines, malpractice, resurection, diseases, homeopathy, poisoning, remedies`
- S³_axi/e5 (xung đột Trung Đông): `zionists, israelis, israeli, intifada, zionist, israel, palestinians, likud, palestinian, isreal`

**Nhận xét embedding:**
- **Top2Vec bị E5 làm hại**; ngược lại **S³ cho chất lượng cao nhất với E5**.
- Với GloVe (phi ngữ cảnh): mọi mô hình kém đi; **S³ vẫn khá ổn** (chỉ nhiễu hơn chút). **FASTopic gần như không đổi** với embedding phi ngữ cảnh.

### 7.2 Minh hoạ sức mạnh: Semantic Axes trên ArXiv ML Papers
🌟 Trích 5 topic, mỗi topic có **top 5 từ dương + top 5 từ âm** (Table 2). Từ âm giúp phân biệt các topic có từ-dương giống nhau (vd Topic 0 và Topic 4 đều có "clustering", chỉ nhờ từ âm mới thấy khác biệt):

| # | Positive | Negative |
|---|---|---|
| 0 | clustering, histograms, clusterings, histogram, classifying | reinforcement, exploration, planning, tactics, reinforce |
| 1 | textual, pagerank, litigants, marginalizing, entailment | matlab, waveforms, microcontroller, accelerometers, microcontrollers |
| 2 | sparsestmax, denoiseing, denoising, minimizers, minimizes | automation, affective, chatbots, questionnaire, attitudes |
| 3 | rebmigraph, subgraph, subgraphs, graphsage, graph | adversarial, adversarially, adversarialization, adversary, security |
| 4 | clustering, estimations, algorithm, dbscan, estimation | cnn, deepmind, deeplabv3, convnet, deepseenet |

**Concept Compass (la bàn khái niệm):** Chọn 2 trục để vẽ từ ngữ lên mặt phẳng 2D và xem chúng tương tác:
- **Trục Topic 1:** vấn đề Ngôn ngữ (linguistic) ↔ Vật lý/Sinh học/Thị giác.
- **Trục Topic 4:** giải pháp Thuật toán (algorithmic) ↔ Deep Learning.
- Phát hiện: giao của (ngôn ngữ × ML cổ điển) là **probability theory**; từ cao ở cả hai trục là **pagerank**; (ngôn ngữ × deep learning) tập trung vào **embeddings, attention**; góc (thuật toán, mức thấp) có **numerical methods, matlab, sensors**; góc deep learning có **computer vision, tensorflow**.

🎤 *"Đây là thứ mà các topic model khác không làm được: không chỉ liệt kê topic, S³ còn cho bạn một 'bản đồ ngữ nghĩa' để đặt bất kỳ từ nào lên và xem nó nằm ở đâu giữa các trục ý nghĩa."*

---

## PHẦN 8 — Kết luận

S³ là phương pháp topic modeling **mới trong không gian ngữ nghĩa liên tục**, lấy cảm hứng từ các phương pháp phân rã ma trận cổ điển như **LSA**, nhưng quan niệm topic là **các trục ngữ nghĩa**. Qua đánh giá định lượng + định tính:
- Tìm được topic **vừa mạch lạc vừa đa dạng**.
- Chạy **tốt hơn khi KHÔNG preprocessing**.
- **Nhanh hơn** các contextual topic model hiện có, mà chất lượng topic **tốt hơn hoặc tương đương**.

---

## PHẦN 9 — Hạn chế (Limitations) — nên có 1 slide để thể hiện tính khách quan

1. **Chỉ số định lượng:** các metric đo chất lượng topic dựa trên giả định mạnh và có nhiều hạn chế đã biết (Rahimi 2024). Các phân tích bổ sung trong bài chỉ bù đắp một phần.
2. **Cài đặt mô hình:** mọi contextual baseline đều được **cài lại (reimplement)** trong Turftopic. BERTopic/Top2Vec được cho là chạy giống bản gốc; nhưng CTM có **khác biệt kiến trúc nhỏ** → runtime và topic có thể lệch chút so với bản gốc.
3. **Không tune hyperparameter:** BERTopic, LDA nổi tiếng nhạy hyperparameter, về lý thuyết có thể tốt hơn nếu được tối ưu. Tác giả cố ý không tune để **tránh researcher degrees of freedom / p-hacking** và tránh tối ưu chính metric đang đánh giá.
4. **Thí nghiệm ngẫu nhiên (stochastic):** chỉ chạy **1 seed** (do pipeline quá tốn thời gian); dùng nhiều embedding model để bù phần nào.
5. **Ảnh hưởng preprocessing:** chỉ thử trên **1 kho** (20 Newsgroups) — cần mở rộng nhiều kho để chắc chắn hơn.
6. **Không đánh giá document-topic proportions** cho downstream task (classification/clustering), vì cho rằng trong thực tế người ta sẽ dùng thẳng sentence embedding cho các task đó. Việc đánh giá tính diễn giải trên người dùng thật để dành cho tương lai.

---

## PHẦN 10 — Chi tiết kỹ thuật bổ sung (Appendix — dùng cho Q&A hoặc slide phụ)

- **Phần cứng chạy runtime:** 2× Intel Xeon Silver 4210 (tổng 20 cores / 40 threads), 187 GiB RAM. **Runtime KHÔNG tính thời gian embedding** (để so sánh công bằng giữa các topic model).
- **Hyperparameter S³:** dùng **mặc định scikit-learn FastICA** — parallel estimation, SVD solver cho whitening, whitening matrix được rescale để đảm bảo unit variance.
- **Top2Vec/BERTopic:** UMAP `n_neighbors=15, n_components=5, min_dist=0.1, metric=cosine`; HDBSCAN `min_cluster_size=15, metric=euclidean, cluster_selection_method=eom`.
- **NPMI Coherence (C_NPMI):** tác giả cũng đo metric lịch sử này, nhưng **cảnh báo nó không đáng tin** — các mô hình mà họ (và đánh giá định tính) cho là chất lượng thấp lại ghi điểm NPMI **rất cao**, trong khi Top2Vec/S³/FASTopic ghi điểm âm. ⇒ **Internal word embedding coherence** là metric tốt hơn.
- **Lỗi thú vị của baseline:** trên ArXiv ML + mpnet, **BERTopic chỉ ước lượng đúng 1 topic** (toàn stop words) nên diversity = 1.0 một cách vô nghĩa; **Top2Vec chỉ ước lượng 2 topic** cho cả kho → thực tế không dùng được.
- **License:** cả topic-benchmark (CLI + kết quả) lẫn Turftopic đều theo **MIT**.

---

## GỢI Ý BỐ CỤC SLIDE (~45 phút, ~35–42 slide)

| # | Khối slide | Số slide gợi ý | Thời lượng |
|---|---|---|---|
| 1 | Tiêu đề + giới thiệu tác giả/hội nghị | 1 | 1' |
| 2 | Topic model là gì? (Phần 1.1) | 2 | 3' |
| 3 | Cách cổ điển & giới hạn BoW (1.2) | 2 | 3' |
| 4 | Embeddings mở ra cơ hội (1.3–1.4) | 2 | 3' |
| 5 | Đóng góp của bài (Phần 2) | 1 | 2' |
| 6 | Bối cảnh & baselines (Phần 3 + bảng) | 3 | 4' |
| 7 | **Cú lật cách nghĩ: topic = trục** (4.1) | 2 | 3' |
| 8 | **PCA vs ICA + cocktail party** (4.2) | 2 | 3' |
| 9 | **6 bước thuật toán + công thức** (4.3) | 4 | 6' |
| 10 | 3 cách tính word importance (4.4) | 2 | 3' |
| 11 | Từ negative — điểm độc đáo (4.5) | 1 | 2' |
| 12 | Setup: datasets, embeddings, metrics (Phần 5) | 3 | 4' |
| 13 | **Kết quả tổng quát + hồi quy** (6.1–6.2) | 3 | 4' |
| 14 | **Phát hiện preprocessing** (6.3) | 1 | 2' |
| 15 | Stop words / embedding / variant (6.4–6.6) | 2 | 3' |
| 16 | Ví dụ định tính + Concept Compass (Phần 7) | 3 | 4' |
| 17 | Kết luận + hạn chế (8–9) | 2 | 2' |
| 18 | Q&A / Turftopic demo | 1 | — |

**Mẹo trình bày:**
- Dành ~40% thời gian cho **Phần 4 (phương pháp)** — đó là phần khán giả cần hiểu nhất.
- Ba slide "đắt giá" nhất để gây ấn tượng: **(1)** analogy cocktail party cho ICA, **(2)** biểu đồ nhanh 4.5× / 27.5×, **(3)** Concept Compass ArXiv.
- Chuẩn bị sẵn hình từ bài báo: **Figure 1** (ý tưởng trục ngữ nghĩa), **Figure 2** (coherence–diversity + rank tốc độ), **Figure 4** (preprocessing), **Figure 7** (concept compass).
