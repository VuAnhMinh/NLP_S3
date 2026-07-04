PORT ?= 8000

.PHONY: help serve open clean

help:
	@echo "Targets:"
	@echo "  make serve   - chay local server tai http://localhost:$(PORT)/slides.html (PORT=xxxx de doi cong)"
	@echo "  make open    - mo slides.html truc tiep bang trinh duyet mac dinh"
	@echo "  make clean   - xoa file tam (.vercel/)"

serve:
	python -m http.server $(PORT)

run:
	python -m http.server $(PORT)

open:
	start slides.html

clean:
	rm -rf .vercel
