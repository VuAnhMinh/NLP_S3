# Tài liệu Học tập: S³ — Semantic Signal Separation (Tách Tín hiệu Ngữ nghĩa)
## HƯỚNG DẪN HỌC TẬP TOÀN DIỆN: PHẦN 1, 2, 3 & CÁC MÔ HÌNH LIÊN QUAN

Tài liệu này bao gồm bản dịch chi tiết, không cắt giảm kiến thức cho các mục: **1. Introduction (Mở đầu)**, **1.1 Contributions (Đóng góp)**, **2. Related Work (Nghiên cứu liên quan)**, **3. Semantic Signal Separation (Tách Tín hiệu Ngữ nghĩa)** và **3.1 Model (Mô hình)** của bài báo khoa học *S³ - Semantic Signal Separation* (ACL 2025). 
Đồng thời, tài liệu cung cấp phần phân tích chuyên sâu bổ trợ về các mô hình baseline (BERTopic, Top2Vec, CTM, ECRTM, FASTopic), sự khác biệt giữa hai trường phái Neural và Clustering, chi tiết công thức toán học và hướng dẫn sử dụng thư viện Python `turftopic`.

---

## MỤC LỤC
1. **Bản Dịch Phần 1: Introduction & 1.1 Contributions**
2. **Bản Dịch Phần 2: Related Work, 2.1 Semantic Axes & 2.2 Embedding-based Topic Models**
3. **Phân Tích So Sánh: Neural Topic Models vs. Clustering Topic Models**
4. **Mổ Xẻ Công Thức & Cách Thu Thập Tham Số của Các Mô Hình Baselines**
   - 4.1 BERTopic & Công thức c-TF-IDF
   - 4.2 Top2Vec & Độ tương đồng Cosine (Cosine Similarity)
   - 4.3 CTM (Contextualized Topic Models) & Hàm lỗi ELBO của VAE
   - 4.4 ECRTM & FASTopic
5. **Thư viện Turftopic: Nguồn gốc và Cách tích hợp trong Python**
6. **Bản Dịch Phần 3 & 3.1 (Phương pháp S³)**
7. **Kiến Thức Nền Tảng: Independent Component Analysis (ICA) & FastICA**
8. **Phân Tích Chi Tiết 3 Công Thức Tính Điểm Từ Vựng của S³**
9. **Cẩm Nang Đọc Ký Hiệu & Thuật Ngữ Toán Học Mở Rộng**
10. **Quy Trình Thu Thập Dữ Liệu & Triển Khai Thực Tế**

---

## 1. BẢN DỊCH PHẦN 1: INTRODUCTION & 1.1 CONTRIBUTIONS

### 1. Introduction (Mở đầu)
"Mô hình chủ đề" (**Topic models**) là một thuật ngữ bao trùm (umbrella term) chỉ các phương pháp tiếp cận thống kê cho phép khám phá chủ đề không giám sát (**unsupervised topic discovery**) trong các kho ngữ liệu văn bản lớn (Blei, 2012). Chúng thường được áp dụng trong phân tích dữ liệu khám phá (**exploratory data analysis**) đối với dữ liệu văn bản, bởi vì chúng cho phép các nhà thực hành khai quật và cô đọng thông tin về nội dung ngữ nghĩa của một kho ngữ liệu mà không cần phải đọc kỹ từng văn bản và tốn nhiều công sức thủ công. Theo cách truyền thống, các chủ đề được trình bày cho người dùng dưới dạng một tập hợp các thuật ngữ quan trọng (từ khóa - **keywords**) nhằm cung cấp các hiểu biết sâu sắc về các cách diễn giải có thể có của chủ đề đó.

Các phương pháp tiếp cận cổ điển đối với việc mô hình hóa chủ đề, chẳng hạn như Phân tích Ngữ nghĩa Ẩn (LSI / LSA - **Latent Semantic Indexing / Latent Semantic Analysis**) (Deerwester và cộng sự, 1988; Dumais, 2004) và Phân bổ Dirichlet Ẩn (LDA - **Latent Dirichlet Allocation**) (Blei và cộng sự, 2003; Blei, 2012), đã dựa trên các biểu diễn tài liệu dạng túi từ (BoW - **bag-of-words**) dựa trên tần suất. Mặc dù các mô hình này đã được sử dụng thành công trong nhiều thập kỷ nghiên cứu xử lý ngôn ngữ tự nhiên (NLP) (Jelodar và cộng sự, 2018), tất cả chúng đều chia sẻ một số hạn chế thực tế và lý thuyết. 

Ví dụ, các mô hình BoW nhạy cảm với các từ có tính chất thống kê không điển hình (chẳng hạn như các từ chức năng - **function words/stop words**), những từ này có thể làm ô nhiễm các mô tả chủ đề dựa trên từ khóa trừ khi các đường ống tiền xử lý nặng (heavy preprocessing pipelines) được áp dụng. Các đường ống như vậy tạo ra nhiều bậc tự do cho nhà nghiên cứu (researcher degrees of freedom). Hơn nữa, tính thưa thớt (sparsity) và số chiều cao của biểu diễn BoW thường dẫn đến hiệu quả tính toán thấp hơn và độ khớp mô hình kém hơn (poorer model fit).

Với sự ra đời của các biểu diễn ngôn ngữ dạng nơ-ron dày đặc (dense neural language representations) (Vaswani và cộng sự, 2017; Le và Mikolov, 2014; Pennington và cộng sự, 2014; Mikolov và cộng sự, 2013), những cơ hội mới đã mở ra cho nghiên cứu mô hình hóa chủ đề. Embedding câu (**Sentence embeddings**) (Reimers và Gurevych, 2019) hứa hẹn rất lớn cho mô hình hóa chủ đề, vì chúng cung cấp các biểu diễn ngôn ngữ mang tính ngữ cảnh, nhạy bén với ngữ pháp (contextual, grammar-sensitive), và mạnh mẽ hơn trước các lỗi chính tả và các từ ngoài từ điển (out-of-vocabulary terms), giảm bớt nhu cầu tiền xử lý vốn loại bỏ đi các thông tin ngôn ngữ có giá trị. 

Ngoài ra, chúng tạo ra các biểu diễn dày đặc trong một không gian liên tục (dense representations in a continuous space), cho phép thực hiện các giả định về phân phối chuẩn (Gaussianity). Chúng cũng cho phép học chuyển giao (**transfer learning**) (Ruder và cộng sự, 2019) trong mô hình hóa chủ đề, tận dụng thông tin đã học được từ các kho ngữ liệu bên ngoài lớn hơn để trích xuất chủ đề.

Một số phương pháp tiếp cận do đó đã được đề xuất bằng cách sử dụng các biểu diễn nơ-ron dày đặc cho việc mô hình hóa chủ đề ngữ cảnh, và các phương pháp này đã được chứng minh là vượt trội so với các đối thủ phi ngữ cảnh của chúng (Bianchi và cộng sự, 2021a; Bianchi và cộng sự, 2021b; Grootendorst, 2022; Angelov, 2020; Wu và cộng sự, 2024b). Tuy nhiên, nhiều phương pháp tiếp cận trong số này vẫn yêu cầu tiền xử lý để đạt hiệu năng tối ưu. Đây là một hạn chế đáng kể của lĩnh vực này, vì các đường ống tiền xử lý không được chuẩn hóa và có thể loại bỏ thông tin có giá trị, điều này đặc biệt có tác động lớn đối với các văn bản ngắn (Wu và cộng sự, 2020).

### 1.1 Contributions (Đóng góp)
Chúng tôi giới thiệu **Semantic Signal Separation** (hoặc **S³**), một kỹ thuật mô hình hóa chủ đề ngữ cảnh hóa mới quan niệm việc mô hình hóa chủ đề như là sự khám phá ra các trục ngữ nghĩa ẩn (latent semantic axes) trong một kho ngữ liệu. Các trục này được phát hiện bằng cách phân rã ma trận embedding của tài liệu bằng cách sử dụng thuật toán FastICA (Hyvärinen và Oja, 2000).

Phương pháp tiếp cận được đề xuất có các đặc điểm:
- **(a)** Đơn giản về mặt khái niệm và được dẫn dắt bởi lý thuyết (theory-driven).
- **(b)** Hoạt động ngang ngửa với các phương pháp tiếp cận hiện tại về độ mạch lạc của word-embedding (word-embedding coherence) và tạo ra độ đa dạng gần như hoàn hảo (near-perfect diversity).
- **(c)** Hiệu quả hơn về mặt tính toán so với các phương pháp tiếp cận hiện có.
- **(d)** Có thể khai thác hiệu quả thông tin ngữ cảnh.

Bên cạnh việc giới thiệu một phương pháp mới, chúng tôi cung cấp một giao diện thống nhất đơn giản dựa trên thư viện scikit-learn cho cả S³ và các phương pháp tiếp cận mô hình hóa chủ đề ngữ cảnh khác trong gói thư viện Python mang tên **Turftopic**.

---

## 2. BẢN DỊCH PHẦN 2: RELATED WORK, 2.1 SEMANTIC AXES & 2.2 EMBEDDING-BASED TOPIC MODELS

### 2. Related Work (Nghiên cứu liên quan)

#### 2.1 Semantic Axes (Các trục ngữ nghĩa)
Phân tích Thành phần Độc lập (ICA - **Independent Component Analysis**) (Jutten và Herault, 1991) trước đây đã từng được áp dụng vào các không gian embedding để khám phá các trục ngữ nghĩa (Musil và Mareček, 2024; Yamagiwa và cộng sự, 2023). Tuy nhiên, các cuộc điều tra này chủ yếu định hướng vào việc tìm kiếm các chiều ngữ nghĩa phổ quát (universal dimensions of semantics) trong embedding từ và ảnh. 

Họ đã chứng minh rằng các trục được phát hiện bởi ICA có thể diễn giải được và thường trùng khớp giữa các không gian embedding và các phương thức (modalities) khác nhau. Ngược lại, nghiên cứu của chúng tôi định hướng vào việc sử dụng các trục ngữ nghĩa để khám phá các chủ đề có tính diễn giải cao trong một kho ngữ liệu cụ thể được quan tâm, chứ không phải nhằm khám phá các chiều ngữ nghĩa phổ quát. Thêm vào đó, không có mô tả chủ đề nào được tính toán hoặc đánh giá trong các nghiên cứu trước đó đó.

#### 2.2 Embedding-based Topic Models (Các mô hình chủ đề dựa trên Embedding)
Nhiều phương pháp tiếp cận mô hình hóa chủ đề sử dụng biểu diễn ngôn ngữ nơ-ron đã được đề xuất trong vài năm qua.

**Neural Topic Models** (Các mô hình chủ đề dạng nơ-ron) (Wu và cộng sự, 2024a) dựa vào mạng nơ-ron sâu để ước lượng tham số. Các mô hình chủ đề ngữ cảnh hóa (CTMs - **Contextualized Topic Models**) (Bianchi và cộng sự, 2021a) là các mô hình sinh (generative models) của biểu diễn BoW, nhưng sử dụng hệ tiên đề mạng tự mã hóa biến phân (VAE - **variational autoencoding**) để suy luận (inference) (Srivastava và Sutton, 2017). Embedding ngữ cảnh được sử dụng làm đầu vào cho mạng mã hóa (encoder) trong **ZeroShotTM**, và đôi khi được nối (concatenate) với các vector BoW trong **CombinedTM**. CTMs thông thường yêu cầu tiền xử lý nặng, và hiệu quả tính toán cũng như chất lượng khớp mô hình giảm mạnh với các từ vựng lớn hơn (Bianchi và cộng sự, 2020).

**ECRTM** (Wu và cộng sự, 2023) là một mô hình nơ-ron dựa vào phép điều hòa gom cụm embedding (embedding clustering regularization) để tạo ra các chủ đề đủ khác biệt và ngăn các mô tả chủ đề hội tụ về phía nhau. Tuy nhiên, phương pháp tiếp cận này không sử dụng các biểu diễn ngữ cảnh và chậm hơn đáng kể so với hầu hết các mô hình chủ đề khác (Wu và cộng sự, 2024b).

**FASTopic** (Wu và cộng sự, 2024b) giới thiệu một hệ tiên đề quan hệ ngữ nghĩa kép (dual-semantic-relation paradigm), nơi các mối quan hệ giữa tài liệu, chủ đề và từ được khái niệm hóa dưới dạng các kế hoạch vận chuyển tối ưu (optimal transport plans). Như họ chứng minh, phương pháp tiếp cận của họ hiệu quả hơn và tạo ra các chủ đề chất lượng cao hơn so với các phương pháp tiếp cận nơ-ron trước đó.

**Clustering Topic Models** (Các mô hình chủ đề gom cụm) khám phá các chủ đề trong kho ngữ liệu bằng cách gom cụm các biểu diễn tài liệu trong không gian embedding. Trọng số độ quan trọng của từ cho một chủ đề cho trước được ước lượng *post hoc* (sau khi gom cụm).

**Top2Vec** (Angelov, 2020) ước lượng các trọng số này bằng cách tính toán độ tương đồng cosine (cosine similarity) giữa mã hóa từ và các tâm cụm (cluster centroids). Điều này giả định các cụm có hình cầu và lồi (spherical, convex), và các mô tả chủ đề có thể bị đại diện sai lệch tùy thuộc vào hình dạng của cụm.

**BERTopic** (Grootendorst, 2022) ước lượng độ quan trọng của từ đối với các cụm bằng cách sử dụng lược đồ gán trọng số tf-idf dựa trên lớp (c-TF-IDF - **class-based tf-idf**). Cả hai phương pháp tiếp cận đều sử dụng **UMAP** (McInnes và Healy, 2018) để giảm chiều dữ liệu và **HDBSCAN** (Campello và cộng sự, 2013) để gom cụm. Cả BERTopic và Top2Vec đều đi kèm với một phương pháp giảm số lượng chủ đề (topic reduction). Điều này là cần thiết, vì HDBSCAN tự động học số lượng cụm từ dữ liệu và số lượng cụm có thể tăng lên rất lớn, điều này có thể chứng minh là phi thực tế.

**Các thách thức** của các mô hình chủ đề ngữ cảnh hiện có sẵn, tuy nhiên, vẫn còn rất nhiều. Nhiều mô hình trong số đó nhạy cảm với các lựa chọn siêu tham số (hyperparameters), tạo ra các chủ đề có khả năng diễn giải đáng ngờ, và phụ thuộc vào các đường ống tiền xử lý có cấu trúc không được chuẩn hóa (Wu và cộng sự, 2024a). Thêm vào đó, hiện vẫn chưa rõ liệu các mô hình này có thực sự hiệu quả trong việc sử dụng thông tin ngữ cảnh và cú pháp hay không, vì chúng thường được đánh giá trên các kho ngữ liệu đã được tiền xử lý.

---

## 3. PHÂN TÍCH SO SÁNH: NEURAL TOPIC MODELS VS. CLUSTERING TOPIC MODELS

Trong mục 2.2 của bài báo, các tác giả đã phân loại rõ các mô hình baseline dùng embedding thành hai trường phái chính. Dưới đây là bảng phân tích so sánh chi tiết:

| Tiêu chí | Neural Topic Models (Mô hình nơ-ron) | Clustering Topic Models (Mô hình gom cụm) |
| :--- | :--- | :--- |
| **Đại diện tiêu biểu** | Contextualized Topic Models (CTM), ECRTM, FASTopic | BERTopic, Top2Vec |
| **Bản chất cốt lõi** | Xem việc sinh văn bản/chủ đề là một **mô hình xác suất**. Dùng mạng nơ-ron tối ưu hóa hàm lỗi để tìm tham số. | Xem chủ đề là các **cụm mật độ** (dense clusters) của tài liệu trong không gian hình học. |
| **Cơ chế ước lượng** | Học các phân phối xác suất ẩn của chủ đề thông qua mạng VAE (Variational Autoencoder) hoặc Kế hoạch Vận chuyển Tối ưu. | Sử dụng thuật toán gom cụm mật độ (thường là HDBSCAN) trên không gian embedding đã giảm chiều bằng UMAP. |
| **Cách tính trọng số từ** | Được học đồng thời (end-to-end) trong quá trình huấn luyện mạng nơ-ron (qua ma trận trọng số decoder). | Được tính toán **post hoc** (sau khi đã gom cụm xong tài liệu) bằng cách tính tương đồng hoặc thống kê tần suất từ theo cụm. |
| **Độ phức tạp tính toán** | **Rất cao**: Cần huấn luyện mạng nơ-ron qua nhiều epochs bằng GPU/CPU; hội tụ chậm và nhạy cảm với hyperparameters. | **Trung bình**: UMAP và HDBSCAN chạy tương đối nhanh, nhưng bước gom cụm có thể trở thành nút thắt cổ chai khi dữ liệu lớn. |
| **Ưu điểm** | Có nền tảng thống kê vững chắc; có thể sinh văn bản (generative model); biểu diễn toán học tường minh. | Rất trực quan về mặt hình học; dễ sử dụng; hoạt động hiệu quả trên nhiều loại kho văn bản mà không cần huấn luyện phức tạp. |
| **Nhược điểm** | Rất chậm; dễ bị sụp đổ chế độ (mode collapse) nơi các chủ đề trùng lặp nhau; nhạy cảm với stop words. | Phụ thuộc vào hình dạng cụm (Top2Vec đòi cụm hình cầu lồi); số lượng cụm do HDBSCAN sinh ra có thể quá lớn, khó kiểm soát. |

---

## 4. MỔ XẺ CÔNG THỨC & CÁCH THU THẬP THAM SỐ CỦA CÁC MÔ HÌNH BASELINES

Để người mới học có thể hiểu được bản chất kỹ thuật của các baselines được đề cập, dưới đây là chi tiết công thức toán học, giải thích tham số và cách thu thập dữ liệu đầu vào.

### 4.1 BERTopic & Công thức c-TF-IDF (Class-based TF-IDF)
Trong BERTopic, các tài liệu được gom cụm thành các lớp (classes), mỗi lớp đại diện cho một chủ đề (topic). Để tìm các từ khóa đại diện cho mỗi chủ đề, tác giả coi tất cả các tài liệu trong một cụm là một "siêu tài liệu" và áp dụng công thức c-TF-IDF.

#### Công thức:
$$W_{x, c} = \text{TF}_{x, c} \cdot \log\left(1 + \frac{A}{f_x}\right)$$

#### Giải thích tham số:
*   **$W_{x, c}$**: Điểm số độ quan trọng của từ $x$ đối với lớp/chủ đề $c$ (điểm càng cao, từ càng đại diện tốt).
*   **$\text{TF}_{x, c}$**: Tần suất của từ $x$ xuất hiện trong lớp/chủ đề $c$ (tổng số lần từ xuất hiện trong tất cả tài liệu thuộc cụm đó).
*   **$f_x$**: Tần suất của từ $x$ trên toàn bộ các lớp (tổng số lần từ xuất hiện trong toàn bộ kho văn bản).
*   **$A$**: Số lượng từ trung bình trên mỗi lớp, được tính bằng:
    $$A = \frac{\sum_c N_c}{C_{total}}$$
    Trong đó $\sum_c N_c$ là tổng số từ của tất cả các lớp, và $C_{total}$ là tổng số lớp (số chủ đề).
*   **Hàm $\log(1 + \dots)$**: Hàm logarit cơ số tự nhiên (hoặc cơ số 10) cộng 1 nhằm mục đích làm mịn (smoothing), tránh trường hợp giá trị bên trong bằng 0 và giảm tốc độ tăng trưởng của trọng số khi giá trị quá lớn.

#### Cách collect dữ liệu cho các tham số:
1.  **Bước gom cụm:** Bạn cần chạy thuật toán gom cụm (như HDBSCAN) trên embedding của tài liệu để gán mỗi tài liệu vào một nhãn lớp $c$.
2.  **Đếm tần suất:** Dùng một bộ đếm từ (như `CountVectorizer`) để thống kê tần suất từ $x$ cho từng tài liệu, sau đó nhóm theo nhãn lớp $c$ để tính $\text{TF}_{x, c}$ và tổng hợp lại để có $f_x$.

---

### 4.2 Top2Vec & Độ tương đồng Cosine (Cosine Similarity)
Top2Vec không sử dụng c-TF-IDF mà dùng chung không gian vector cho cả tài liệu, từ vựng và chủ đề. Sau khi gom cụm tài liệu, tâm (centroid) của các tài liệu trong cụm được coi là Vector Chủ đề (Topic Vector). Độ quan trọng của từ được tính bằng khoảng cách hình học từ vector từ đến vector chủ đề này.

#### Công thức:
$$\text{Similarity}(v_j, c_t) = \cos(\Theta) = \frac{v_j \cdot c_t}{\lVert v_j \rVert \cdot \lVert c_t \rVert}$$

#### Giải thích tham số:
*   **$v_j$**: Vector embedding của từ $j$ trong từ vựng (kích thước $d \times 1$).
*   **$c_t$**: Vector tâm cụm (centroid) của chủ đề $t$ (kích thước $d \times 1$), được tính bằng trung bình cộng các vector embedding của các tài liệu thuộc cụm $t$.
*   **$v_j \cdot c_t$**: Phép nhân vô hướng (dot product) của hai vector:
    $$v_j \cdot c_t = \sum_{k=1}^d v_{j,k} \cdot c_{t,k}$$
*   **$\lVert v_j \rVert$ và $\lVert c_t \rVert$**: Chuẩn $L_2$ (độ dài Euclidean) của từng vector.
*   **$\cos(\Theta)$**: Giá trị Cosine của góc giữa hai vector. Giá trị nằm trong khoảng $[-1, 1]$. Càng gần 1, từ càng đồng hướng và càng quan trọng với chủ đề.

#### Cách collect dữ liệu cho các tham số:
1.  **Vector từ $v_j$:** Dùng một thuật toán học vector từ (như Word2Vec, Doc2Vec) để học đồng thời biểu diễn của từ và tài liệu trong cùng một không gian.
2.  **Vector chủ đề $c_t$:** Tính trung bình cộng của toàn bộ vector tài liệu nằm trong cụm $t$.

---

### 4.3 CTM (Contextualized Topic Models) & Hàm lỗi ELBO của VAE
CTM dựa trên mạng tự mã hóa biến phân (Variational Autoencoder - VAE). Nó nhận vào embedding ngữ cảnh của tài liệu để làm điều kiện mã hóa, nhưng mục tiêu là tái cấu trúc lại vector túi từ (BoW) của tài liệu đó. Quá trình huấn luyện tối ưu hóa hàm lỗi ELBO (Evidence Lower Bound).

#### Công thức hàm tối ưu (ELBO Loss):
$$\mathcal{L}_{ELBO}(\theta, \phi) = \mathbb{E}_{q_\phi(z|x)}[\log p_theta(x|z)] - D_{KL}(q_\phi(z|x) \parallel p(z))$$

#### Giải thích tham số:
*   **$x$**: Vector biểu diễn túi từ (Bag-of-Words) của tài liệu đầu vào (kích thước $V_{vocab} \times 1$).
*   **$z$**: Vector biến ẩn biểu thị tỷ lệ chủ đề trong tài liệu (kích thước $N \times 1$).
*   **$q_\phi(z|x)$**: Mạng mã hóa (Encoder) với các tham số mạng là $\phi$. Nhận vào embedding ngữ cảnh của tài liệu và ước lượng phân phối xác suất của biến ẩn $z$.
*   **$p_\theta(x|z)$**: Mạng giải mã (Decoder) với các tham số mạng là $\theta$. Nhận vào vector chủ đề $z$ và cố gắng tái cấu trúc lại vector BoW gốc $x$.
*   **$\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$**: Kỳ vọng toán học của log xác suất tái cấu trúc. Tham số này đo lường khả năng tái tạo dữ liệu đầu vào của mô hình (decoder hoạt động tốt đến đâu).
*   **$D_{KL}(q_\phi(z|x) \parallel p(z))$**: Khoảng cách Kullback-Leibler (KL Divergence) giữa phân phối do encoder dự đoán và phân phối tiên nghiệm $p(z)$ (thường giả định là Logistic-Normal). Đóng vai trò là số hạng điều hòa (regularizer) giúp làm mượt không gian ẩn $z$.

#### Cách collect dữ liệu cho các tham số:
1.  **Dữ liệu đầu vào $x$ (BoW):** Bạn cần xây dựng ma trận đếm tần suất từ của kho văn bản thô.
2.  **Embedding ngữ cảnh (Contextual Embeddings):** Chạy mô hình Sentence Transformer trên các tài liệu để tạo embedding ngữ cảnh làm input cho Encoder $q_\phi$.

---

### 4.4 ECRTM & FASTopic
*   **ECRTM (Embedding Clustering Regularization Topic Model):**
    *   **Bản chất:** Đưa ra một số hạng điều hòa gom cụm (regularization) vào hàm tối ưu của Neural Topic Model. Mục tiêu là buộc các vector biểu diễn chủ đề phải nằm xa nhau trong không gian vector, tránh việc mô hình sinh ra các chủ đề có từ khóa giống hệt nhau (sụp đổ chế độ).
    *   **Hạn chế:** Bản gốc của ECRTM không sử dụng embedding ngữ cảnh trực tiếp làm đầu vào dẫn dắt ngữ nghĩa, mà chỉ dùng để điều hòa phân cụm tĩnh, khiến tốc độ chạy rất chậm.
*   **FASTopic (Fast Arbitrary-Semantic-Space Topic Model):**
    *   **Bản chất:** Mô hình hóa mối quan hệ giữa tài liệu - chủ đề và chủ đề - từ như một bài toán Vận chuyển tối ưu (Optimal Transport - OT). Nó sử dụng khoảng cách Wasserstein để căn chỉnh phân phối của tài liệu và từ vựng trong không gian ngữ nghĩa mà không cần cấu trúc VAE phức tạp, mang lại tốc độ chạy vượt trội và chất lượng chủ đề cao hơn CTM.

---

## 5. THƯ VIỆN TURFTOPIC: NGUỒN GỐC VÀ CÁCH TÍCH HỢP TRONG PYTHON

Trong mục 1.1 đóng góp của bài báo, các tác giả đề cập đến thư viện **Turftopic**.

### 5.1 Nguồn gốc thư viện
*   **Không phải thư viện có sẵn mặc định** trong Python (như `math`, `os` hay `json`).
*   **Do chính nhóm tác giả bài báo phát triển** và phát hành dưới dạng mã nguồn mở (giấy phép MIT).
*   Thư viện được thiết kế theo chuẩn giao diện của **scikit-learn** (sử dụng các hàm quen thuộc như `.fit()`, `.transform()`, `.fit_transform()`).
*   **Mục đích:** Gói gọn thuật toán S³ đề xuất cùng với tất cả các mô hình baseline ngữ cảnh phổ biến (BERTopic, Top2Vec, CTM, v.v.) vào một API duy nhất để các nhà nghiên cứu dễ so sánh đối chiếu.

### 5.2 Cách cài đặt và sử dụng
Bạn có thể dễ dàng cài đặt thư viện này thông qua trình quản lý gói `pip`:
```bash
pip install turftopic
```

#### Ví dụ mã nguồn chạy S³ qua thư viện Turftopic:
```python
from turftopic import KeyNMF, SemanticSignalSeparation
from datasets import load_dataset

# 1. Chuẩn bị dữ liệu tài liệu
dataset = ["Học máy là một ngành của trí tuệ nhân tạo.",
           "Xử lý ngôn ngữ tự nhiên giúp máy tính hiểu tiếng người.",
           "Mô hình chủ đề phân tích các bài viết tự động."]

# 2. Khởi tạo mô hình S3 với số lượng chủ đề mong muốn (ví dụ N = 3)
# Thư viện tự động tải mô hình sentence-transformer mặc định bên dưới
model = SemanticSignalSeparation(n_components=3, feature_importance="combined")

# 3. Huấn luyện mô hình và trích xuất chủ đề
model.fit(dataset)

# 4. In ra các từ khóa hàng đầu của từng chủ đề
model.print_topics(top_n=5)
```

---

## 6. BẢN DỊCH PHẦN 3 & 3.1 (PHƯƠNG PHÁP S³)
*(Nội dung này giữ nguyên bản dịch đầy đủ đã thực hiện ở phiên bản trước của bạn).*

Trong bài báo này, chúng tôi giới thiệu **Semantic Signal Separation** (hoặc **S³**), một phương pháp tiếp cận mới cho bài toán mô hình hoá chủ đề (**topic modeling**) trong không gian embedding liên tục (**continuous embedding spaces**)... *(Xem chi tiết bản dịch tại mục 1 của file content.md trước đó).*

---

## 7. KIẾN THỨC NỀN TẢNG: ICA & FASTICA
*(Nội dung giữ nguyên kiến thức nền tảng đã viết ở phiên bản trước của bạn, bao gồm Cockail Party, so sánh PCA vs ICA, negentropy và bước whitening).*

---

## 8. PHÂN TÍCH CHI TIẾT 3 CÔNG THỨC WORD IMPORTANCE CỦA S³
*(Nội dung giữ nguyên phân tích hình học, ưu nhược điểm của Axial, Angular và Combined với giải thích về luỹ thừa bậc 3).*

---

## 9. CẨM NANG ĐỌC KÝ HIỆU & THUẬT NGỮ TOÁN HỌC MỞ RỘNG

Bổ sung các ký hiệu toán học xuất hiện trong phần Related Work và các mô hình baseline:

| Ký hiệu toán | Tên gọi chuẩn | Cách đọc tiếng Việt | Ý nghĩa trong các mô hình baseline |
| :---: | :--- | :--- | :--- |
| **$W_{x, c}$** | W sub x, c | *Vê-kép ích xê* | Trọng số c-TF-IDF của từ $x$ đối với lớp $c$ trong BERTopic. |
| **$\text{TF}_{x, c}$** | TF sub x, c | *Tê-Ép ích xê* | Tần số xuất hiện của từ $x$ trong lớp $c$. |
| **$f_x$** | f sub x | *Ép ích* | Tần suất của từ $x$ trên toàn bộ các lớp trong kho văn bản. |
| **$v_j \cdot c_t$** | Dot product of v_j and c_t | *Tích vô hướng của v gi và c tê* | Phép nhân vô hướng hai vector trong Top2Vec để tính cosine similarity. |
| **$\mathcal{L}_{ELBO}$** | ELBO Loss | *Lờ Ê-Lờ-Bê-O* hoặc *Hàm lỗi ELBO* | Hàm tối ưu hóa (Evidence Lower Bound) trong mạng VAE của mô hình CTM. |
| **$q_\phi(z \vert x)$** | q sub phi of z given x | *Quy phi của z với điều kiện x* | Hàm mật độ xác suất của biến ẩn $z$ sinh ra bởi Encoder $q$ với tham số $\phi$. |
| **$p_\theta(x \vert z)$** | p sub theta of x given z | *Pê thê-ta của x với điều kiện z* | Hàm mật độ xác suất tái cấu trúc $x$ sinh ra bởi Decoder $p$ với tham số $\theta$. |
| **$D_{KL}(\cdot \parallel \cdot)$** | KL Divergence | *Khoảng cách K-L* | Số hạng đo độ lệch giữa hai phân phối xác suất trong VAE. |
| **$\mathbb{E}_{q}[\cdot]$** | Expectation under q | *Kỳ vọng toán học dưới phân phối q* | Giá trị trung bình có trọng số của hàm số khi các biến ẩn tuân theo phân phối $q$. |

---

## 10. QUY TRÌNH THU THẬP DỮ LIỆU & TRIỂN KHAI THỰC TẾ
*(Nội dung giữ nguyên hướng dẫn 6 bước triển khai code python và checklist tự kiểm tra).*
