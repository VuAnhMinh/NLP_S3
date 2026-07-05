# Tài liệu Học tập: S³ — Semantic Signal Separation (Tách Tín hiệu Ngữ nghĩa)

Tài liệu này dịch chi tiết **Phần 3 (Semantic Signal Separation)** và **Phần 3.1 (Model)** của bài báo khoa học *S³ - Semantic Signal Separation* (ACL 2025), đồng thời cung cấp kiến thức nền tảng chuyên sâu về toán học, trực giác thuật toán, cách đọc các ký hiệu công thức và hướng dẫn thu thập dữ liệu thực tế cho người mới bắt đầu.

---

## MỤC LỤC
1. **Bản Dịch Đầy Đủ & Chính Xác Phần 3 & 3.1**
2. **Kiến Thức Nền Tảng: Independent Component Analysis (ICA) & FastICA**
3. **Phân Tích Chi Tiết 3 Công Thức Tính Độ Quan Trọng Của Từ (Word Importance)**
4. **Hướng Dẫn Thu Thập Dữ Liệu & Quy Trình Tính Toán Thực Tế**
5. **Hướng Dẫn Đọc Tên Ký Hiệu & Thuật Ngữ Toán Học**

---

## 1. BẢN DỊCH ĐẦY ĐỦ & CHÍNH XÁC PHẦN 3 & 3.1

Dưới đây là bản dịch hoàn chỉnh, không cắt giảm kiến thức từ bài báo gốc. Các thuật ngữ học thuật quan trọng được giữ nguyên tiếng Anh bên cạnh nghĩa tiếng Việt để thuận tiện cho việc tra cứu chuyên sâu.

### 3. Semantic Signal Separation (Tách Tín hiệu Ngữ nghĩa)
Trong bài báo này, chúng tôi giới thiệu **Semantic Signal Separation** (hoặc **S³**), một phương pháp tiếp cận mới cho bài toán mô hình hoá chủ đề (**topic modeling**) trong không gian embedding liên tục (**continuous embedding spaces**), nhằm vượt qua các thách thức đã đề cập phía trên của các phương pháp topic modeling ngữ cảnh (**contextual topic modeling**) hiện tại.

Thay vì diễn giải các chủ đề dưới dạng các cụm (**clusters**) hoặc phân phối xác suất của từ (**word probabilities**), chúng tôi quan niệm các chủ đề là các **trục ngữ nghĩa (semantic axes)** giải thích sự biến thiên (**variation**) đặc thù của một kho ngữ liệu (**corpus**). Điều này đạt được bằng cách phân rã (**decomposing**) các biểu diễn ngữ nghĩa thành:
- Các **thành phần ẩn (latent components) $A$**: được giả định chính là các chủ đề (topics).
- **Độ mạnh của các thành phần trong từng tài liệu $S$**: chính là mức độ quan trọng giữa tài liệu và chủ đề (document-topic importances).

Để các chủ đề có tính chất tách biệt rõ ràng về mặt khái niệm (**conceptually distinct**), chúng tôi sử dụng phương pháp Phân tích Thành phần Độc lập (**Independent Component Analysis - ICA**) (Jutten và Herault, 1991) để tìm ra chúng. Độ quan trọng của từ đối với chủ đề (**term importances**) được ước lượng từ độ mạnh của các thành phần chủ đề trong embedding của từ (**word embeddings $V$**).

Xét trên một số khía cạnh, phương pháp của chúng tôi có thể được coi là hậu duệ ngữ cảnh (**contextual successor**) của phương pháp Phân tích Ngữ nghĩa Ẩn (**Latent Semantic Analysis - LSA**) (Dumais, 2004; Deerwester và cộng sự, 1988) - phương pháp vốn khám phá ra các nhân tố (factors) dựa trên sự đồng xuất hiện của từ (word-occurrences).

### 3.1 Model (Mô hình)
Các biểu diễn tài liệu (**document representations**) thu được bằng cách mã hoá các tài liệu bằng một mô hình **Sentence Transformer**.

1. Gọi $X$ là ma trận mã hoá của các tài liệu (**document encodings matrix**).
   
Việc phân rã các biểu diễn tài liệu thành các trục ngữ nghĩa độc lập được thực hiện bằng phương pháp Phân tích Thành phần Độc lập (**Independent Component Analysis - ICA**) (Jutten và Herault, 1991). Trong nghiên cứu này, chúng tôi sử dụng thuật toán **FastICA** (Hyvärinen và Oja, 2000) để xác định các thành phần ngữ nghĩa ẩn. 

Là một bước tiền xử lý, phép làm trắng (**whitening**) được áp dụng lên ma trận embedding, vì FastICA là một mô hình không nhiễu (**noiseless model**). Do mặc định ICA sẽ tìm ra số lượng thành phần bằng đúng số chiều của embedding, chúng tôi giảm chiều dữ liệu của embedding trong quá trình làm trắng bằng cách chỉ lấy $N$ thành phần chính đầu tiên (**principal components**), với $N$ là số lượng chủ đề (topics) mong muốn.

2. Phân rã $X$ sử dụng FastICA:
   $$X = A \cdot S$$
   Trong đó, $A$ là **ma trận trộn (mixing matrix)**, và $S$ là **ma trận nguồn (source matrix)** chứa độ quan trọng giữa tài liệu và chủ đề (document-topic importances).

Độ quan trọng của từ đối với chủ đề (**term importances**), dùng để lựa chọn các từ ngữ mô tả chủ đề, được tính toán bằng cách chiếu (**projecting**) các từ lên các trục ngữ nghĩa đã tìm được.

3. Mã hoá từ vựng của kho ngữ liệu bằng cùng một mô hình encoder đó. Gọi ma trận mã hoá từ vựng là $V$.
4. Gọi ma trận tách (hay ma trận mở trộn - **unmixing matrix**) là $C$, được tính bằng nghịch đảo giả (**pseudo-inverse**) của ma trận trộn $A$:
   $$C = A^+$$
5. Chiếu các từ lên các trục ngữ nghĩa đã phát hiện bằng cách nhân ma trận embedding của từ với ma trận unmixing:
   $$W = V \cdot C^T$$
6. Tính toán điểm quan trọng của từ (word importance scores) cho từng chủ đề.

Chúng tôi xem xét ba phương pháp để tính toán độ quan trọng của từ:
1. **Độ quan trọng của từ theo Trục (Axial word importances)**: được định nghĩa là vị trí của từ trên các trục ngữ nghĩa. Độ quan trọng của từ $j$ đối với chủ đề $t$ là:
   $$\beta_{tj} = W_{jt}$$
2. **Chủ đề theo Góc (Angular topics)**: có thể được tính toán bằng cách lấy cosine của góc giữa vector từ đã chiếu và các trục ngữ nghĩa:
   $$\beta_{tj} = \cos(\Theta) = \frac{W_{jt}}{\lVert W_j \rVert}$$
3. **Độ quan trọng kết hợp (Combined word importance)**: là sự kết hợp của cả hai cách tiếp cận trên:
   $$\beta_{tj} = \frac{(W_{jt})^3}{\lVert W_j \rVert}$$
   *(Chúng tôi lấy luỹ thừa bậc lẻ của vị trí từ để giữ nguyên dấu của nó).*

Độ quan trọng theo trục (**axial word importance**) giúp tạo ra các mô tả chủ đề chứa các từ **nổi bật nhất (most salient)** của chủ đề đó, trong khi độ quan trọng theo góc (**angular importance**) sẽ gán trọng số cao nhất cho những từ **đặc trưng riêng biệt nhất (most specific)**. Độ quan trọng kết hợp (**combined importance**) hướng tới việc cân bằng cả hai khía cạnh này.

Lưu ý rằng tất cả các công thức trên đều cho phép các từ có **độ quan trọng âm (negative importance)** đối với một chủ đề nhất định. Mặc dù điều này cũng xuất hiện trong phương pháp LSA, các nghiên cứu trước đây chưa từng khai thác khái niệm này. Việc diễn giải mô hình có thể được mở rộng bằng cách kiểm tra các từ có điểm số thấp nhất trên một chủ đề cho trước, cung cấp một **định nghĩa phủ định (negative definition)** cho chủ đề đó.

Để đảm bảo tính so sánh tương đương với các phương pháp không cho phép định nghĩa phủ định, các thử nghiệm đối sánh mô hình của chúng tôi sẽ bỏ qua các từ có giá trị âm, tuy nhiên một ví dụ minh hoạ thực tế sẽ được trình bày ở Mục 6.2.

**Suy luận (Inference)** tỷ lệ chủ đề trong các tài liệu mới (chưa từng thấy trước đây) có thể đạt được bằng cách nhân embedding của tài liệu mới đó với ma trận unmixing $C$:
1. Gọi mã hoá của các tài liệu mới chưa từng thấy là $\hat{X}$.
2. Tính toán ma trận tài liệu - chủ đề mới:
   $$\hat{S} = \hat{X} \cdot C^T$$

---

## 2. KIẾN THỨC NỀN TẢNG: ICA & FastICA

Để hiểu sâu sắc về thuật toán S³, trước tiên bạn cần nắm vững hai khái niệm toán học cốt lõi: **Independent Component Analysis (ICA)** và thuật toán **FastICA**.

### 2.1 Bản chất của ICA (Independent Component Analysis)
**Bài toán Tách nguồn mù (Blind Source Separation - BSS):**
Hãy tưởng tượng kịch bản "Cocktail Party": Trong một phòng tiệc ồn ào có 3 người nói chuyện cùng một lúc ($s_1, s_2, s_3$ là các nguồn âm thanh độc lập). Bạn đặt 3 chiếc micro ở các góc phòng. Mỗi chiếc micro sẽ ghi lại một hỗn hợp âm thanh khác nhau ($x_1, x_2, x_3$) phụ thuộc vào khoảng cách từ micro tới từng người nói. 
Nhiệm vụ của ICA là: Chỉ từ các tín hiệu hỗn hợp thu được ($x_1, x_2, x_3$), hãy tách chúng ngược trở lại thành 3 giọng nói ban đầu độc lập ($s_1, s_2, s_3$) mà không cần biết trước giọng nói của họ ra sao hay micro được đặt ở vị trí nào.

Trong bài báo này:
*   **Các nguồn độc lập ($s$)** chính là các **Chủ đề (Topics)** ẩn chứa trong văn bản.
*   **Bản trộn quan sát được ($x$)** chính là **Embedding của tài liệu** ($X$) - vì một tài liệu thường là sự pha trộn của nhiều chủ đề khác nhau.
*   **Ma trận trộn ($A$)** mô tả cách các chủ đề kết hợp lại để tạo ra embedding tài liệu.
*   **Ma trận unmixing ($C$)** là bộ lọc giúp gỡ rối bản trộn, lấy lại các chủ đề thuần khiết ban đầu.

### 2.2 Sự khác biệt cốt lõi giữa PCA và ICA
Nhiều người thường nhầm lẫn giữa PCA (Phân tích Thành phần Chính) và ICA. Dưới đây là bảng so sánh giúp bạn phân biệt:

| Tiêu chí | PCA (Principal Component Analysis) | ICA (Independent Component Analysis) |
| :--- | :--- | :--- |
| **Mục tiêu chính** | Tìm các hướng có **biến thiên (variance) lớn nhất** để nén dữ liệu. | Tìm các hướng **độc lập thống kê** để tách nguồn tín hiệu bị trộn. |
| **Mối quan hệ trục** | Các trục bắt buộc phải **vuông góc** với nhau (Orthogonal). | Các trục **không nhất thiết phải vuông góc**, chúng xoay tự do để đạt độc lập. |
| **Ràng buộc toán** | **Không tương quan (Uncorrelated)**: Chỉ đảm bảo hiệp phương sai bằng 0 ($\text{Cov}(y_1, y_2) = 0$). | **Độc lập thống kê (Statistically Independent)**: Đòi hỏi phân phối chung bằng tích các phân phối biên $p(y_1,y_2) = p(y_1)p(y_2)$. |
| **Giả định phân phối** | Phù hợp nhất với dữ liệu phân phối chuẩn (Gaussian). | Bắt buộc dữ liệu nguồn phải **phi chuẩn (Non-Gaussian)**. |

> [!NOTE]
> **Trực giác về "Không tương quan" và "Độc lập":**
> Nếu hai biến độc lập với nhau, chắc chắn chúng không tương quan. Nhưng điều ngược lại không đúng. Ví dụ, nếu $x$ là một biến ngẫu nhiên đối xứng qua 0, và $y = x^2$. Khi đó, $x$ và $y$ hoàn toàn không tương quan (hệ số tương quan tuyến tính bằng 0), nhưng chúng cực kỳ phụ thuộc vào nhau ($y$ được tính trực tiếp từ $x$). PCA chỉ tìm các biến không tương quan tuyến tính, trong khi ICA đi tìm sự độc lập thực sự (ở mọi bậc thống kê cao hơn).

### 2.3 Thuật toán FastICA và Bước Làm trắng (Whitening)
FastICA là một thuật toán cực kỳ nhanh và ổn định để giải quyết bài toán ICA do Aapo Hyvärinen đề xuất năm 1997. Cách hoạt động của FastICA dựa trên việc **tối đa hoá tính phi chuẩn (Non-Gaussianity)** của các tín hiệu đầu ra. 

Theo Định lý Giới hạn Trung tâm (Central Limit Theorem), tổng của nhiều biến ngẫu nhiên độc lập sẽ có xu hướng tiến về phân phối chuẩn (Gaussian) hơn là các biến thành phần. Vì vậy, các tín hiệu bị trộn lẫn (như embedding tài liệu) sẽ mang tính Gaussian cao. Để tách chúng ra thành các nguồn độc lập, thuật toán tìm các hướng xoay sao cho tín hiệu chiếu lên đó **ít mang tính Gaussian nhất**. Độ phi chuẩn được đo bằng các hàm toán học như **Kurtosis** (độ nhọn của phân phối) hoặc **Negentropy**.

#### Bước tiền xử lý: Làm trắng dữ liệu (Whitening / Sphering)
Trước khi chạy FastICA, bắt buộc phải thực hiện bước "làm trắng" dữ liệu:
1.  **Trung hoà dữ liệu (Centering):** Trừ đi giá trị trung bình để dữ liệu có kỳ vọng bằng 0.
2.  **Làm trắng (Whitening):** Biến đổi tuyến tính ma trận sao cho các biến thành phần không còn tương quan tuyến tính và đều có phương sai bằng 1. Về mặt hình học, bước này biến đổi một "đám mây" dữ liệu hình elip dài thành một "đám mây" hình cầu tròn trịa (spherical). Hiệp phương sai của ma trận sau khi làm trắng sẽ là ma trận đơn vị $I$.
3.  **Giảm chiều dữ liệu (Dimensionality Reduction):** FastICA thông thường sẽ tìm ra số lượng thành phần bằng đúng số chiều của vector embedding ban đầu (ví dụ: 384 chiều). Để lấy đúng $N$ chủ đề mong muốn, S³ lồng ghép việc giảm chiều ngay trong bước làm trắng bằng cách sử dụng **SVD (Singular Value Decomposition)** để chỉ giữ lại $N$ thành phần chính (Principal Components) có biến thiên lớn nhất.

---

## 3. PHÂN TÍCH CHI TIẾT 3 CÔNG THỨC WORD IMPORTANCE

Sau khi dùng FastICA phân rã ma trận tài liệu, ta thu được ma trận unmixing $C$. Khi chiếu từ vựng lên ma trận này, ta có ma trận trọng số $W = V \cdot C^T$ kích thước $(\text{Số từ vựng} \times N)$. 
Với mỗi cột $t$ đại diện cho chủ đề $t$, phần tử $W_{jt}$ chính là toạ độ của từ $j$ trên trục chủ đề $t$. Từ đây, có 3 cách tính điểm quan trọng của từ để chọn ra các từ mô tả chủ đề hay nhất.

### 3.1 Minh hoạ hình học (Dựa theo Figure 3 trong bài báo)
Hãy tưởng tượng không gian 2 chiều với hai trục chủ đề là $T_0$ (trục hoành) và $T_t$ (trục tung). Một từ trong từ vựng có vector embedding đã được chiếu là $\vec{v}$ (toạ độ của nó là $W_{j}$).
*   **$v_t$ (hoặc $W_{jt}$)**: Hình chiếu vuông góc của $\vec{v}$ lên trục chủ đề $T_t$. Đây chính là khoảng cách từ gốc toạ độ đến điểm chiếu trên trục $T_t$.
*   **$\lVert v \rVert$ (hoặc $\lVert W_j \rVert$)**: Độ dài (chuẩn Euclidean) của vector từ $\vec{v}$ từ gốc toạ độ.
*   **$\Theta$ (Theta)**: Góc hợp bởi vector $\vec{v}$ và trục chủ đề $T_t$.

```
         T_t (Trục chủ đề t)
          ^
          |      . vector v (embedding từ j)
          |     /|
      v_t |    / |
   (W_jt) |   /  |
          |  /   |
          | / Θ  |
          |/_____|____________> T_0
        Gốc (0)  Hình chiếu của v lên T_0
```

### 3.2 Phân tích chi tiết từng công thức

#### Công thức 1: Axial Word Importance (Độ quan trọng theo Trục)
$$\beta_{tj} = W_{jt}$$

*   **Ý nghĩa hình học:** Lấy trực tiếp toạ độ $v_t$ của từ trên trục chủ đề.
*   **Tính chất:** Các từ có vector dài (nằm xa gốc toạ độ) và hướng gần trục sẽ có điểm rất cao.
*   **Ưu điểm:** Giúp chọn ra các từ **nổi bật nhất (most salient)**, xuất hiện cực kỳ phổ biến trong các tài liệu thuộc chủ đề này. Tạo ra các mô tả chủ đề có độ mạch lạc ngữ nghĩa cực kỳ cao (High Coherence).
*   **Nhược điểm:** Dễ bị lẫn các từ mang tính bao quát, từ thông dụng (ví dụ: trong chủ đề "phần mềm", nó có thể chọn ra các từ chung chung như "máy tính", "hệ thống", "sử dụng").

#### Công thức 2: Angular Word Importance (Độ quan trọng theo Góc)
$$\beta_{tj} = \cos(\Theta) = \frac{W_{jt}}{\lVert W_j \rVert}$$

*   **Ý nghĩa hình học:** Đây chính là hàm Cosine của góc $\Theta$. Công thức này triệt tiêu hoàn toàn độ dài $\lVert W_j \rVert$ của vector từ, chỉ quan tâm đến **hướng** của vector.
*   **Tính chất:** Một từ nằm rất sát trục $T_t$ (góc $\Theta$ nhỏ, $\cos(\Theta) \approx 1$) sẽ đạt điểm tối đa, dù vector của nó rất ngắn (từ hiếm gặp).
*   **Ưu điểm:** Chọn ra các từ **đặc trưng riêng biệt nhất (most specific)** cho chủ đề. Giúp các chủ đề cực kỳ phân biệt nhau, đạt độ đa dạng tối đa (Near-perfect Diversity).
*   **Nhược điểm:** Có thể chọn ra các từ quá hiếm, từ chuyên ngành sâu hoặc từ viết tắt ít gặp, làm giảm tính dễ hiểu tổng quát của chủ đề đối với người đọc phổ thông.

#### Công thức 3: Combined Word Importance (Độ quan trọng Kết hợp)
$$\beta_{tj} = \frac{(W_{jt})^3}{\lVert W_j \rVert}$$

*   **Ý nghĩa hình học:** Bằng toạ độ trục luỹ thừa 3 chia cho độ dài vector từ.
*   **Tại sao lại dùng mũ 3?** 
    1.  Mũ 3 là luỹ thừa bậc lẻ, giúp **giữ nguyên dấu** âm/dương của toạ độ $W_{jt}$. Nếu toạ độ âm (từ mang nghĩa phủ định chủ đề), kết quả vẫn âm. Nếu dùng mũ 2 (bình phương), dấu âm sẽ biến mất.
    2.  Luỹ thừa bậc cao ($x^3$) hoạt động như một bộ phóng đại: nó khuếch đại các giá trị toạ độ lớn và thu nhỏ các giá trị toạ độ nhỏ, giúp làm nổi bật hẳn những từ thực sự quan trọng.
*   **Ưu điểm (Khuyên dùng):** Đây là sự dung hoà hoàn hảo. Nó vừa giữ lại thuộc tính nổi bật từ Axial ($W_{jt}$ nằm trên tử số với luỹ thừa mạnh), vừa tích hợp tính đặc trưng từ Angular (chia cho độ dài $\lVert W_j \rVert$ ở mẫu số). Bài báo khuyến nghị sử dụng công thức này làm mặc định.

---

## 4. HƯỚNG DẪN THU THẬP DỮ LIỆU & QUY TRÌNH TÍNH TOÁN

Để chạy công thức trên thực tế, bạn cần đi qua các bước chuẩn bị dữ liệu đầu vào cụ thể như sau:

### Bước 1: Thu thập Kho tài liệu (Tạo dữ liệu cho ma trận $X$)
*   **Dữ liệu cần thu thập:** Một tập hợp gồm $D$ tài liệu văn bản (ví dụ: 10.000 bài báo, bài viết blog, hoặc phản hồi khách hàng).
*   **Tiền xử lý:** Với S³, điều kỳ diệu là bạn **không cần loại bỏ stop words**, không cần lemmatization hay xóa ký tự đặc biệt. Hãy cứ giữ nguyên văn bản thô vì mô hình Sentence Transformer cần ngữ cảnh đầy đủ để tạo embedding tốt nhất.
*   **Mã hoá (Embedding):** Sử dụng thư viện `sentence-transformers` trong Python để chuyển văn bản thành ma trận $X$ kích thước $(D \times d)$.
    ```python
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2') # d = 384
    # X_raw có kích thước (D_documents x d_dimensions)
    X_raw = model.encode(documents) 
    ```

### Bước 2: Trích xuất và mã hoá Từ vựng (Tạo dữ liệu cho ma trận $V$)
*   **Dữ liệu cần thu thập:** Danh sách tất cả các từ đơn/cụm từ xuất hiện trong kho ngữ liệu của bạn (ví dụ: dùng `CountVectorizer` của scikit-learn để lấy danh sách từ vựng duy nhất, giả sử có $V_{vocab}$ từ).
*   **Mã hoá (Embedding):** Đưa toàn bộ danh sách từ này qua **cùng một mô hình** Sentence Transformer ở Bước 1. Việc dùng chung mô hình là bắt buộc để đảm bảo vector từ và vector tài liệu nằm trong cùng một không gian hình học.
    ```python
    from sklearn.feature_extraction.text import CountVectorizer
    # Trích xuất từ vựng từ kho văn bản
    vectorizer = CountVectorizer(stop_words=None)
    vectorizer.fit(documents)
    vocab = vectorizer.get_feature_names_out() # Danh sách V_vocab từ
    
    # Mã hoá từ vựng thành ma trận V kích thước (V_vocab x d_dimensions)
    V = model.encode(vocab)
    ```

### Bước 3: Xác định số lượng chủ đề $N$
*   Chọn một số nguyên $N$ làm số lượng chủ đề bạn muốn khám phá (ví dụ: $N = 20$).

### Bước 4: Áp dụng FastICA (Tìm ma trận $A$ và $S$)
*   Chạy thuật toán FastICA trên ma trận $X_{raw}^T$ (chuyển vị để đưa về dạng số chiều $\times$ số tài liệu) với tham số giảm chiều về $N$ trong bước whitening.
*   Kết quả trả về ma trận trộn $A$ kích thước $(d \times N)$ và ma trận nguồn $S$ kích thước $(N \times D)$.
    ```python
    from sklearn.decomposition import FastICA
    # Thiết lập FastICA để lấy ra N components
    ica = FastICA(n_components=N, whiten='unit-variance', random_state=42)
    # FastICA fit trên X_raw (D x d), học cách phân rã
    S_matrix = ica.fit_transform(X_raw) # S_matrix kích thước (D x N)
    A_matrix = ica.mixing_ # A_matrix kích thước (d x N)
    ```

### Bước 5: Tính toán ma trận Unmixing $C$ và ma trận chiếu $W$
*   Tính nghịch đảo giả Moore-Penrose của $A$ để có $C = A^+$ kích thước $(N \times d)$.
*   Nhân ma trận từ vựng $V$ với chuyển vị của $C$ để thu được $W = V \cdot C^T$ kích thước $(V_{vocab} \times N)$.
    ```python
    import numpy as np
    # Tính nghịch đảo giả của A
    C = np.linalg.pinv(A_matrix) # C có kích thước (N x d)
    
    # Chiếu từ vựng lên các trục chủ đề
    W = np.dot(V, C.T) # W có kích thước (V_vocab x N)
    ```

### Bước 6: Áp dụng các công thức tính điểm $\beta_{tj}$
*   Từ ma trận $W$, áp dụng một trong 3 công thức ở Mục 3.2 để tính ra điểm số $\beta_{tj}$ cho từng từ $j$ trong từng chủ đề $t$.
*   Sắp xếp các từ theo thứ tự giảm dần của $\beta_{tj}$ để lấy ra Top 10 từ mô tả cho mỗi chủ đề.
*   (Tùy chọn) Sắp xếp theo thứ tự tăng dần (giá trị âm lớn nhất) để tìm ra các từ định nghĩa phủ định (Negative definition) cho chủ đề.

---

## 5. HƯỚNG DẪN ĐỌC TÊN KÝ HIỆU & THUẬT NGỮ TOÁN HỌC

Để tự tin trình bày và hiểu bản chất công thức, đây là bảng hướng dẫn cách đọc các ký hiệu toán học xuất hiện trong bài báo:

| Ký hiệu toán | Tên gọi chuẩn | Cách đọc tiếng Việt | Ý nghĩa trong thuật toán |
| :---: | :--- | :--- | :--- |
| **$\beta_{tj}$** | Beta sub t j | *Bê-ta tê gi* | Điểm số độ quan trọng của từ $j$ đối với chủ đề $t$. |
| **$\Theta$** | Theta | *Thi-ta* hoặc *Thê-ta* | Góc hợp giữa vector từ và trục chủ đề ngữ nghĩa. |
| **$\cos(\Theta)$** | Cosine of Theta | *Cốt thi-ta* | Đo lường mức độ trùng khớp về mặt hướng (chỉ hướng, bỏ qua độ dài). |
| **$A^+$** | A-plus / Moore-Penrose Pseudo-inverse | *A cộng* hoặc *A nghịch đảo giả* | Ma trận nghịch đảo tổng quát của ma trận trộn $A$ khi ma trận này không vuông hoặc không khả nghịch. |
| **$C^T$** | C transpose | *C chuyển vị* | Ma trận nhận được bằng cách đổi các hàng của ma trận $C$ thành các cột và ngược lại. |
| **$\lVert W_j \rVert$** | L2 Norm of W_j | *Chuẩn của W gi* hoặc *Độ dài vector W gi* | Khoảng cách hình học từ gốc toạ độ tới điểm biểu diễn của từ $j$. |
| **$\hat{X}$** | X-hat | *Ích-xét hắt* hoặc *Ích-xét mũ* | Ma trận biểu diễn embedding của các tài liệu mới cần dự đoán. |
| **$\hat{S}$** | S-hat | *Ét-xét hắt* hoặc *Ét-xét mũ* | Ma trận phân phối chủ đề dự đoán cho các tài liệu mới. |
| **$V$** | Vocabulary embedding matrix | *Ma trận V* | Ma trận chứa các vector biểu diễn cho toàn bộ từ vựng. |
| **$W$** | Projected word matrix | *Ma trận Vê-kép* | Ma trận kết quả sau khi chiếu các từ lên các trục chủ đề. |
| **$X = A \cdot S$** | Matrix factorization equation | *Ích-xét bằng A nhân Ét-xét* | Phương trình phân rã ma trận: Tài liệu = Trộn $\times$ Nguồn chủ đề. |
