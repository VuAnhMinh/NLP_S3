PORT ?= 8000

.PHONY: help serve open open2 run run2 run3 rungoogle opengoogle clean

help:
	@echo "Targets:"
	@echo "  make serve       - chay local server tai http://localhost:$(PORT)/slides.html (PORT=xxxx de doi cong)"
	@echo "  make run         - chay server + mo slides.html   (ban de HOC)"
	@echo "  make run2        - chay server + mo slides_2.html (ban THUYET TRINH)"
	@echo "  make run3        - chay server + mo slides_3.html (HOC SAU Phan 3)"
	@echo "  make rungoogle   - chay server + mo google/google_slides.html (ban HOC CHUYEN SAU PHAN 3)"
	@echo "  make open        - mo slides.html   truc tiep bang trinh duyet"
	@echo "  make open2       - mo slides_2.html truc tiep bang trinh duyet"
	@echo "  make opengoogle  - mo google/google_slides.html truc tiep bang trinh duyet"
	@echo "  make clean       - xoa file tam (.vercel/)"

serve:
	python -m http.server $(PORT)

run:
	@echo "Mo http://localhost:$(PORT)/slides.html  (Ctrl+C de dung)"
	@(sleep 1; open "http://localhost:$(PORT)/slides.html") >/dev/null 2>&1 &
	python3 -m http.server $(PORT)

run2:
	@echo "Mo http://localhost:$(PORT)/slides_2.html  (Ctrl+C de dung)"
	@(sleep 1; open "http://localhost:$(PORT)/slides_2.html") >/dev/null 2>&1 &
	python3 -m http.server $(PORT)

run3:
	@echo "Mo http://localhost:$(PORT)/slides_3.html  (Ctrl+C de dung)"
	@(sleep 1; open "http://localhost:$(PORT)/slides_3.html") >/dev/null 2>&1 &
	python3 -m http.server $(PORT)

rungoogle:
	@echo "Mo http://localhost:$(PORT)/google/google_slides.html  (Ctrl+C de dung)"
	@(sleep 1; open "http://localhost:$(PORT)/google/google_slides.html") >/dev/null 2>&1 &
	python3 -m http.server $(PORT)

open:
	open slides.html

open2:
	open slides_2.html

opengoogle:
	open google/google_slides.html

clean:
	rm -rf .vercel
