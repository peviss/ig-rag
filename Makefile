.PHONY: build up down logs ingest ask search chat stats clean install dev-api dev-web web-install

QUERY ?= ""

# ── Docker: full stack ────────────────────────────────────────────────────────

build:
	docker compose build

up:
	docker compose up -d
	@echo "UI → http://localhost:8080   API → http://localhost:8000"

down:
	docker compose down

logs:
	docker compose logs -f

# ── Docker: CLI commands (use --profile cli) ──────────────────────────────────

ingest:
	docker compose run --rm ig-rag ingest

ask:
	docker compose run --rm ig-rag ask $(QUERY)

search:
	docker compose run --rm ig-rag search $(QUERY)

chat:
	docker compose run --rm ig-rag chat

stats:
	docker compose run --rm ig-rag stats

clean:
	rm -rf ig_store/

# ── Local dev ─────────────────────────────────────────────────────────────────

install:
	uv sync

dev-api:
	uv run uvicorn api:app --reload --port 8000

dev-web:
	cd web && npm run dev

web-install:
	cd web && npm install
