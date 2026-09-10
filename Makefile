PORT ?= 8000
PY   ?= .venv/Scripts/python.exe

.PHONY: help serve main run run2 run3 rungoogle rungoogle1 open open2 opengoogle opengoogle1 pptx openpptx deploy clean \
        demo train-visfd train-news train-bank train-all paper \
        cafebert-sources cafebert-checkpoint cafebert-smoke cafebert-seed42 cafebert-sensitivity cafebert-audit cafebert-report cafebert-reference-audit cafebert-reference-report

help:
	@echo "=== QUAN TRONG (code / thuc nghiem) ==="
	@echo "  make demo        - chay Streamlit demo phan tich truc topic S3 (demo/app.py)"
	@echo "  make train-all   - tao lai TAT CA checkpoint S3 (visfd + vietnamese-news + uts-bank)"
	@echo "  make train-visfd - tao lai checkpoint visfd (CafeBERT, k=10..50)"
	@echo "  make train-news  - tao lai checkpoint vietnamese-news (CafeBERT, k=10..50)"
	@echo "  make train-bank  - tao lai checkpoint uts-bank (E5, k=10/14/20/30)"
	@echo "  make paper       - bien dich report/paper.pdf (XeLaTeX + bibtex)"
	@echo ""
	@echo "=== Benchmark CafeBERT/S3 (benchmark/cafebert_full/) ==="
	@echo "  make cafebert-sources    - tai va khoa revision 4 nguon benchmark"
	@echo "  make cafebert-checkpoint - tai CafeBERT pretrained revision da pin + manifest"
	@echo "  make cafebert-smoke      - chay smoke grid truoc full benchmark"
	@echo "  make cafebert-seed42     - chay primary seed 42"
	@echo "  make cafebert-sensitivity- chay seed 11,29,47"
	@echo "  make cafebert-audit      - audit coverage, metric va provenance"
	@echo "  make cafebert-report     - sinh report, bieu do va bang LaTeX timing"
	@echo "  make cafebert-reference-audit  - audit artifact 480 run da commit"
	@echo "  make cafebert-reference-report - tai sinh report/LaTeX tu artifact da commit"
	@echo ""
	@echo "=== Slide NLP (S3) - thu muc NLP/ ==="
	@echo "  make main        - chay server + mo NLP/main/main.html      (SLIDE CHINH - THUYET TRINH)"
	@echo "  make run         - chay server + mo NLP/temp/slides.html    (ban de HOC day du)"
	@echo "  make run3        - chay server + mo NLP/temp/slides_3.html  (HOC SAU Phan 3)"
	@echo "  make rungoogle   - chay server + mo NLP/temp/google/google_slides.html"
	@echo "  make rungoogle1  - chay server + mo NLP/temp/google/google_slides_1.html"
	@echo "  make open2       - mo NLP/main/main.html    truc tiep bang trinh duyet"
	@echo "  make open        - mo NLP/temp/slides.html  truc tiep bang trinh duyet"
	@echo "  make pptx        - dung lai NLP/main/main.pptx tu build_pptx.py"
	@echo "  make openpptx    - mo NLP/main/main.pptx bang PowerPoint/Keynote"
	@echo ""
	@echo "=== Slide relation (LeWorldModel) - thu muc relation/ ==="
	@echo "  make relation    - chay server + mo relation/main/index.html"
	@echo "  make openrelation- mo relation/main/index.html truc tiep bang trinh duyet"
	@echo ""
	@echo "=== chung ==="
	@echo "  make serve       - chay local server tai http://localhost:$(PORT)/ (PORT=xxxx de doi cong)"
	@echo "  make deploy      - deploy len Vercel production (/ /nlp /1 /2 /3 /g... /relation)"
	@echo "  make clean       - xoa file tam (.vercel/)"

serve:
	python3 -m http.server $(PORT)

# ---------- NLP ----------

main:
	@echo "Mo http://localhost:$(PORT)/NLP/main/main.html  (Ctrl+C de dung)"
	@(sleep 1; open "http://localhost:$(PORT)/NLP/main/main.html") >/dev/null 2>&1 &
	python3 -m http.server $(PORT)

# alias: run2 = main (giu tuong thich cu)
run2: main

run:
	@echo "Mo http://localhost:$(PORT)/NLP/temp/slides.html  (Ctrl+C de dung)"
	@(sleep 1; open "http://localhost:$(PORT)/NLP/temp/slides.html") >/dev/null 2>&1 &
	python3 -m http.server $(PORT)

run3:
	@echo "Mo http://localhost:$(PORT)/NLP/temp/slides_3.html  (Ctrl+C de dung)"
	@(sleep 1; open "http://localhost:$(PORT)/NLP/temp/slides_3.html") >/dev/null 2>&1 &
	python3 -m http.server $(PORT)

rungoogle:
	@echo "Mo http://localhost:$(PORT)/NLP/temp/google/google_slides.html  (Ctrl+C de dung)"
	@(sleep 1; open "http://localhost:$(PORT)/NLP/temp/google/google_slides.html") >/dev/null 2>&1 &
	python3 -m http.server $(PORT)

rungoogle1:
	@echo "Mo http://localhost:$(PORT)/NLP/temp/google/google_slides_1.html  (Ctrl+C de dung)"
	@(sleep 1; open "http://localhost:$(PORT)/NLP/temp/google/google_slides_1.html") >/dev/null 2>&1 &
	python3 -m http.server $(PORT)

open:
	open NLP/temp/slides.html

open2:
	open NLP/main/main.html

opengoogle:
	open NLP/temp/google/google_slides.html

opengoogle1:
	open NLP/temp/google/google_slides_1.html

pptx:
	cd NLP/main && python3 build_pptx.py

openpptx:
	open NLP/main/main.pptx

# ---------- relation ----------

relation:
	@echo "Mo http://localhost:$(PORT)/relation/main/index.html  (Ctrl+C de dung)"
	@(sleep 1; open "http://localhost:$(PORT)/relation/main/index.html") >/dev/null 2>&1 &
	python3 -m http.server $(PORT)

openrelation:
	open relation/main/index.html

# ---------- chung ----------

demo:
	$(PY) -m streamlit run demo/app.py

# ---------- s3_reproduction: tao lai checkpoint ----------

train-visfd:
	$(PY) -m s3_reproduction.cli --backend turftopic --dataset visfd --n-topics 10 20 30 40 50

train-news:
	$(PY) -m s3_reproduction.cli --backend turftopic --dataset vietnamese-news --n-topics 10 20 30 40 50

train-bank:
	$(PY) -m s3_reproduction.cli --backend turftopic --dataset uts-bank --encoder e5 --n-topics 10 14 20 30

train-all: train-visfd train-news train-bank

# ---------- report: bien dich paper.pdf ----------

paper:
	cd report && xelatex -interaction=nonstopmode paper.tex && bibtex paper && xelatex -interaction=nonstopmode paper.tex && xelatex -interaction=nonstopmode paper.tex
	@echo "-> report/paper.pdf"

cafebert-sources:
	python3 -m benchmark.cafebert_full.fetch_sources

cafebert-checkpoint:
	python3 -m benchmark.cafebert_full.fetch_cafebert_checkpoint

cafebert-smoke:
	python3 -m benchmark.cafebert_full.run_cafebert_smoke

cafebert-seed42:
	python3 -m benchmark.cafebert_full.run_cafebert_full --seeds 42

cafebert-sensitivity:
	python3 -m benchmark.cafebert_full.run_cafebert_full --seeds 11,29,47

cafebert-audit:
	python3 -m benchmark.cafebert_full.run_cafebert_full --dedupe-only
	python3 -m benchmark.cafebert_full.audit_cafebert_full

cafebert-report:
	python3 -m benchmark.cafebert_full.generate_cafebert_full_report
	python3 -m benchmark.cafebert_full.generate_cafebert_timing_appendix

cafebert-reference-audit:
	S3_CAFEBERT_RESULTS_DIR="$(CURDIR)/benchmark/cafebert_full/reference" python3 -m benchmark.cafebert_full.audit_cafebert_full

cafebert-reference-report:
	S3_CAFEBERT_RESULTS_DIR="$(CURDIR)/benchmark/cafebert_full/reference" python3 -m benchmark.cafebert_full.generate_cafebert_full_report
	S3_CAFEBERT_RESULTS_DIR="$(CURDIR)/benchmark/cafebert_full/reference" python3 -m benchmark.cafebert_full.generate_cafebert_timing_appendix

deploy:
	vercel deploy --prod

clean:
	rm -rf .vercel
