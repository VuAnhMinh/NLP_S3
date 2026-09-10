# NLP_S3

Repo chứa **hai** bộ slide, mỗi bộ một thư mục riêng:

| Thư mục | Nội dung |
|---|---|
| [`NLP/`](NLP/) | **S³ — Semantic Signal Separation** (ACL 2025) — https://aclanthology.org/2025.acl-long.32/ |
| [`relation/`](relation/) | **LeWorldModel** — bộ slide riêng, độc lập với `NLP/` |

Slide viết bằng [reveal.js](https://revealjs.com/) (tải qua CDN) trong file HTML đơn — **không cần build, không cần cài dependency**. Chỉ cần trình duyệt (và mạng để tải CDN).

---

## 👉 Anh em làm việc ở đâu?

> **Làm việc chính trong thư mục [`NLP/main/`](NLP/main/).** Đây là bản thuyết trình chung của cả nhóm.

- **[`NLP/main/main.html`](NLP/main/main.html)** — SLIDE CHÍNH (bản thuyết trình, mạch "hook-first" theo 4 Act).
- **[`NLP/main/content.md`](NLP/main/content.md)** — kịch bản lời nói đi kèm (câu dẫn, thời lượng từng slide).

Mọi chỉnh sửa cho buổi báo cáo → sửa 2 file trong `NLP/main/`. Các file trong `NLP/temp/` chỉ là bản nháp/tài liệu học tham khảo, **không** dùng để trình bày.

---

## Cấu trúc thư mục

```
NLP/                     ← NỘI DUNG SLIDE S³
  main/                  ← LÀM VIỆC Ở ĐÂY (main.html + content.md)
  temp/                  ← bản nháp & tài liệu học (tham khảo)
  example/  thien_part/  bản mẫu / hình phần đánh giá
  2025.acl-long.32.pdf   bài báo gốc

relation/                ← bộ slide LeWorldModel (giữ nguyên, không đụng tới)

s3_reproduction/         ← code tái lập S³ + CLI train checkpoint (cli.py)
demo/                    ← Streamlit demo phân tích trục topic (app.py)
server/                  ← server FastAPI/WebSocket định tuyến câu hỏi ngân hàng
benchmark/cafebert_full/ ← pipeline benchmark 480 run (4 corpus × 6 model)
report/                  ← báo cáo ACL (paper.tex → paper.pdf)
artifacts/               ← checkpoint + embedding (KHÔNG commit, tạo lại bằng make train-*)

Makefile                 lệnh chạy/train/build nhanh — xem `make help`
vercel.json              cấu hình đường dẫn khi deploy Vercel
```

---

## Lệnh Make

Gõ `make help` để xem đầy đủ. Đổi cổng slide bằng `PORT=xxxx` (mặc định 8000).
Các lệnh Python dùng `.venv/Scripts/python.exe` — đổi bằng biến `PY=...` nếu cần.

### ⭐ Quan trọng (code / thực nghiệm)

| Lệnh | Việc |
|---|---|
| `make demo` | Chạy **Streamlit demo** phân tích trục topic S³ (`demo/app.py`) — cần có checkpoint trước (xem `make train-*`) |
| `make train-all` | Tạo lại **toàn bộ** checkpoint S³: `visfd` + `vietnamese-news` + `uts-bank` |
| `make train-visfd` | Tạo lại checkpoint **visfd** (encoder CafeBERT, k = 10…50) |
| `make train-news` | Tạo lại checkpoint **vietnamese-news** (encoder CafeBERT, k = 10…50) |
| `make train-bank` | Tạo lại checkpoint **uts-bank** (encoder E5, k = 10/14/20/30) — cho tab định tuyến ngân hàng trong demo |
| `make paper` | Biên dịch **`report/paper.pdf`** (XeLaTeX + bibtex, chạy 3 lượt) |

> Checkpoint `.joblib` **không** được commit vào git (nằm trong `artifacts/`). Nếu mất, chạy `make train-*` để tạo lại — nhanh (visfd ~1 phút, vietnamese-news ~6 phút, uts-bank ~20 giây khi có GPU + model đã cache). `uts-bank` sẽ tự tải dataset từ HuggingFace.

### Benchmark CafeBERT/S³ (`benchmark/cafebert_full/`)

Pipeline này dùng **virtualenv riêng** (`.venv-cafebert`, xem [README benchmark](benchmark/cafebert_full/README.md)).

| Lệnh | Việc |
|---|---|
| `make cafebert-sources` | Tải và khoá revision 4 nguồn dữ liệu benchmark |
| `make cafebert-checkpoint` | Tải CafeBERT pretrained (revision đã pin) + manifest |
| `make cafebert-smoke` | Chạy smoke grid trước khi chạy full benchmark |
| `make cafebert-seed42` | Chạy primary seed 42 |
| `make cafebert-sensitivity` | Chạy seed 11, 29, 47 |
| `make cafebert-audit` | Audit coverage, metric và provenance |
| `make cafebert-report` | Sinh report, biểu đồ và bảng LaTeX timing |
| `make cafebert-reference-audit` | Audit artifact 480 run đã commit |
| `make cafebert-reference-report` | Tái sinh report/LaTeX từ artifact đã commit |

### Slide (chạy & mở nhanh)

| Lệnh | Việc |
|---|---|
| `make main` | Server + mở **NLP/main/main.html** — *slide chính để trình bày* |
| `make run2` | Alias của `make main` |
| `make run` | Server + mở **NLP/temp/slides.html** (bản HỌC đầy đủ) |
| `make run3` | Server + mở **NLP/temp/slides_3.html** (HỌC SÂU Phần 3) |
| `make rungoogle` / `rungoogle1` | Server + mở **NLP/temp/google/google_slides*.html** |
| `make open` / `open2` | Mở **slides.html** / **main.html** trực tiếp (`file://`, không qua server) |
| `make opengoogle` / `opengoogle1` | Mở google slides trực tiếp |
| `make pptx` / `openpptx` | Dựng lại / mở **NLP/main/main.pptx** |
| `make relation` / `openrelation` | Server + mở / mở trực tiếp **relation/main/index.html** |
| `make serve` | Chỉ chạy local server tại `http://localhost:8000/` |
| `make deploy` | Deploy lên **Vercel production** |
| `make clean` | Xoá file tạm (`.vercel/`) |

Ví dụ:
```bash
make demo                 # mở demo phân tích trục topic
make train-all            # tạo lại toàn bộ checkpoint đã mất
make paper                # biên dịch lại report/paper.pdf
make main                 # mở slide chính để tập thuyết trình
make run3 PORT=9000       # mở bản học sâu Phần 3 ở cổng 9000
```

Ghi chú:
- `make main / run / run3 / ...` tự khởi động local server rồi mở trình duyệt; `Ctrl+C` để dừng.
- `make open*` mở thẳng file (`file://`) — nhanh hơn nhưng vài trình duyệt chặn do CORS; khi đó dùng bản qua server.
- Các lệnh slide dùng `python3` và `open` (macOS). Máy khác chỉnh `open` → `xdg-open` (Linux) / `start` (Windows).

---

## Xem trên web (Vercel) — mở được trên điện thoại

Dự án deploy qua **Vercel** (miễn phí). Một domain, nhiều đường dẫn (cấu hình trong `vercel.json`):

| URL | Nội dung |
|---|---|
| `minh-internal-nlp-slides.vercel.app/` | **NLP/main/main.html** — slide chính |
| `…/nlp`, `…/2`, `…/main` | NLP/main/main.html (cùng slide chính) |
| `…/1` | NLP/temp/slides.html (bản HỌC đầy đủ) |
| `…/3` | NLP/temp/slides_3.html (HỌC SÂU Phần 3) |
| `…/g1` `…/g2` `…/g3` | NLP/temp/google/... |
| `…/relation` | relation/main/index.html — slide LeWorldModel |

Repo đã kết nối Git với Vercel → mỗi `git push` lên `main` sẽ **tự deploy**. Muốn đẩy ngay bản local (chưa commit) thì dùng `make deploy`.

---

## Benchmark CafeBERT/S³ tái lập

Thư mục [`benchmark/cafebert_full/`](benchmark/cafebert_full/) là pipeline độc lập cho thực nghiệm luận văn: bốn corpus tiếng Việt, sáu cấu hình topic model, WEC-in, diversity, C_NPMI, provenance document-ID và timing theo stage. Pipeline không thay thế `s3_reproduction/` hoặc demo hiện có.

Đọc [hướng dẫn chạy benchmark](benchmark/cafebert_full/README.md) trước. Checkpoint CafeBERT là model pretrained công khai được pin revision và tải lại bằng script; S³ không có checkpoint pretrained riêng trong thực nghiệm này.

---

## Điều hướng slide

- `→ / ←` hoặc `Space`: chuyển slide kế tiếp/trước.
- `↓ / ↑`: xuống/lên slide con (nếu có).
- `Esc`: xem tổng quan tất cả slide (overview).
- `F`: toàn màn hình · `S`: speaker notes · `M`: menu nhảy nhanh theo mục.
- Thanh tiến trình + số thứ tự slide hiển thị ở góc dưới.
