# Act 2 — trích từ S3_Slides.pptx (bản của bạn cùng nhóm)

> Nguồn: `main/S3_Slides.pptx` (14 slide). Đây là bản trích để đối chiếu / import vào `main.html`.
> Ảnh nhúng: 2 icon 256×256 (slide 6 — nhánh Neural / Clustering).

## S1 — Quá trình phát triển Topic Modeling (timeline 4 giai đoạn)
- GĐ1 Bag-of-Words: LSA (1988) · LDA (2003)
- GĐ2 Contextual Embedding: BERT · SBERT (~2019)
- GĐ3 Neural / Clustering: CTM · BERTopic · Top2Vec
- GĐ4 S³ (2025): ICA trên Embedding
- *Notes:* Tổng quan 4 giai đoạn.

## S2 — Giai đoạn 1: Bag-of-Words
- Biểu diễn văn bản bằng đếm từ — bỏ ngữ pháp & thứ tự.
- Bảng document-term (mèo/đuổi/chuột/chó/sợ trên d₁,d₂,d₃).
- Điểm yếu: mất ngữ cảnh; "bank" nghĩa nào cũng như nhau; ma trận thưa; bắt buộc tiền xử lý nặng.
- LSA & LDA đều xây trên BoW, khác cách "tìm chủ đề".

## S3 — LSA (1988): SVD
- A = U · Σ · Vᵀ (Document-Concept × Quan trọng × Term-Concept).
- Nén ma trận thưa xuống K chiều → mỗi chiều = 1 khái niệm ẩn.
- S³ tự nhận "hậu duệ contextual của LSA".
- Điểm yếu: SVD chỉ trực giao (uncorrelated), không độc lập → 1 trục lẫn nhiều chủ đề; input vẫn BoW.

## S4 — LDA (2003): mô hình sinh xác suất
- 3 bước: chọn phân phối chủ đề (Dirichlet) → sinh từng từ → suy luận ngược (variational/Gibbs).
- Bảng so sánh LSA vs LDA (nền tảng toán, mô hình sinh, tiền xử lý).
- Điểm yếu: giả định Dirichlet có thể sai; nhạy α,β; suy luận tốn; vẫn BoW.

## S5 — Giai đoạn 2: Contextual Embedding
- Bảng BoW vs Sentence Embedding (kích thước, thưa/dày, ngữ cảnh, ngữ pháp, tiền xử lý).
- Transfer learning: pretrain trên hàng tỷ câu.
- Câu hỏi mở: có embedding rồi, tìm topic bằng cách nào? → 2 nhánh.

## S6 — Hai nhánh khai thác Embedding (🖼️ 2 icon)
- Nhánh Neural: CTM (VAE), ECRTM (VAE+Sinkhorn), FASTopic (Optimal Transport).
- Nhánh Clustering: UMAP → HDBSCAN → từ khóa; Top2Vec (cosine+centroid), BERTopic (c-TF-IDF).

## S7 — CTM: Variational Autoencoder
- Encoder nén embedding → vector topic (K); Decoder tái tạo BoW (|V|) → "cửa ngõ" quay lại đếm từ.
- Ví dụ thất bại (20NG): topic = "145, ax, 0d, _o, a86..." toàn ký hiệu.

## S8 — ECRTM & FASTopic
- ECRTM: VAE+Sinkhorn (1000 lặp), chậm nhất, không dùng contextual, topic lỗi "verbeek, billington...".
- FASTopic: Optimal Transport, gộp nhầm "moon, bike, car, orbit...", kém khi embedding lớn, ~200 epoch.
- Điểm yếu chung Neural: chậm; nhạy hyperparameter; vẫn quay lại đếm từ.

## S9 — Nhánh Clustering: pipeline 3 bước
- UMAP giảm chiều → HDBSCAN phân cụm → gán từ khóa (Top2Vec cosine/centroid; BERTopic c-TF-IDF).
- Thất bại (Appendix F): BERTopic 20NG chỉ 1 cụm toàn stopword; Top2Vec ArXiv chỉ 2 cụm; corpus thô stopword tới 100%.

## S10 — Điểm yếu nhánh Clustering
1. Quá nhiều tham số (UMAP + HDBSCAN 5-6 tham số).
2. Số topic nổ không kiểm soát → cần topic reduction.
3. Phân cụm sai → lỗi chảy xuôi, không sửa được.
4. Tài liệu bị bỏ rơi (noise points).

## S11 — 3 Thách thức chung (Neural + Clustering)
1. Nhạy hyperparameter, khó tái lập.
2. Vẫn phụ thuộc tiền xử lý (CTM decoder BoW; BERTopic c-TF-IDF).
3. Chưa rõ tận dụng context thật (đánh giá trên corpus đã xử lý).

## S12 — Nền tảng: Trục ngữ nghĩa đã được khám phá
- Musil & Mareček (2024); Yamagiwa et al. (2023).
- Bảng: prior work (trục phổ quát, không sinh từ khóa) vs S³ (trục đặc thù corpus, sinh từ khóa, đánh giá).
- S³ là người đầu tiên biến kỹ thuật trục ngữ nghĩa thành topic model đầy đủ.

## S13 — S³: hậu duệ contextual của LSA
- LSA: BoW + SVD + trục trực giao → S³: contextual embedding + ICA + trục độc lập.
- Vì sao ICA thay SVD: uncorrelated (yếu) vs independent (mạnh); cocktail party analogy.

## S14 — S³ giải quyết 3 thách thức
- Bảng: hyperparameter (FastICA default), tiền xử lý (không đếm từ), context (mô hình DUY NHẤT tốt hơn trên corpus thô).
- Điểm mấu chốt: loại bỏ hoàn toàn đếm từ/TF-IDF khỏi pipeline.
