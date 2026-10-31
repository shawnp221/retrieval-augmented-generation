.PHONY: install up down test step1 step2 step3 step4 step5 step6 step7 ask

install: ; pip install -e ".[dev,cohere]"
up:      ; docker compose up -d
down:    ; docker compose down
test:    ; pytest -q

step1: ; python scripts/01_chunking.py
step2: ; python scripts/02_embedding.py
step3: ; python scripts/03_store.py --reset
step4: ; python scripts/04_filter.py
step5: ; python scripts/05_search.py
step6: ; python scripts/06_rerank.py
step7: ; python scripts/07_answer.py
ask:   ; python scripts/ask.py --team sales "What is the refund limit for invoice INV-12345?"
