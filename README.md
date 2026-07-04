# NLP_S3

Slide thuyết trình về bài báo **S³ — Tách Tín hiệu Ngữ nghĩa**:
https://aclanthology.org/2025.acl-long.32/

Slide được viết bằng [reveal.js](https://revealjs.com/) (tải qua CDN) trong một file HTML duy nhất — không cần build, không cần cài dependency.

## Cấu trúc thư mục

- `slides.html` — file slide chính, mở trực tiếp là chạy được.
- `content.md` — nội dung/ghi chú soạn thảo cho slide.
- `2025.acl-long.32.pdf` — bài báo gốc.
- `vercel.json` — cấu hình rewrite `/` → `/slides.html` khi deploy lên Vercel.

## Chạy trên local

### Dùng Makefile (khuyên dùng)

```bash
make open   # mo slides.html truc tiep bang trinh duyet
make serve  # chay local server tai http://localhost:8000/slides.html (PORT=xxxx de doi cong)
make clean  # don file tam (.vercel/)
```

### Hoặc chạy tay

Vì `slides.html` chỉ load tài nguyên qua CDN (không gọi API riêng), bạn có thể mở trực tiếp bằng trình duyệt:

```bash
# Windows
start slides.html

# macOS
open slides.html

# Linux
xdg-open slides.html
```

Nếu trình duyệt chặn một số tính năng khi mở bằng `file://` (ví dụ do CORS), hãy chạy qua một local server đơn giản rồi mở `http://localhost:<port>`:

```bash
# Cách 1: dùng Python (có sẵn trên hầu hết máy)
python -m http.server 8000

# Cách 2: dùng Node (nếu đã cài Node.js)
npx serve .
```

Sau đó mở trình duyệt tại `http://localhost:8000/slides.html`.

## Điều hướng slide

- `→ / ←` hoặc `Space`: chuyển slide kế tiếp/trước đó.
- `↓ / ↑`: xuống/lên slide con (nếu có).
- `Esc`: xem tổng quan tất cả slide (overview mode).
- Progress bar và số thứ tự slide hiển thị ở góc dưới màn hình.

## Deploy

Dự án được deploy qua **Vercel**, cấu hình rewrite tại `vercel.json` để domain gốc trỏ thẳng vào `slides.html`.
