# Atelier — common dev/ops tasks
#
# Run `make help` for a full list. Targets are grouped: deps, run, eval,
# cleanup, docker, qdrant ops.

.PHONY: help install run eval clean fresh docker-build docker-up docker-down docker-logs docker-shell qdrant-drop format lint check

help:  ## List available targets
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ── Local dev (uv) ────────────────────────────────────────────────────────────

install:  ## Sync deps from pyproject.toml / uv.lock
	uv sync

run:  ## Run Streamlit locally (foreground)
	uv run streamlit run app.py

eval:  ## Run the offline DeepEval CLI; refreshes artifacts/eval_results.json
	uv run python evaluate.py

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:  ## Remove .data/ + legacy root state files (sessions, checkpoints, usage, embedding cache)
	rm -rf .data/ embedding_cache/
	rm -f sessions.json checkpoints.db checkpoints.db-shm checkpoints.db-wal usage.db usage.db-shm usage.db-wal eval_checkpoints.db
	@echo "Cleared all local state. Re-run 'make run' to start fresh."

fresh: clean  ## clean + drop Qdrant collections + .deepeval cache
	rm -rf .deepeval/
	@$(MAKE) qdrant-drop

# ── Docker (local; uses the same Dockerfile that prod builds) ────────────────

docker-build:  ## Build the production image locally as atelier:local
	docker compose build

docker-up:  ## Run the container locally with .env + ./.data volume
	docker compose up -d
	@echo "Streamlit: http://localhost:8501"

docker-down:  ## Stop and remove the local container
	docker compose down

docker-logs:  ## Tail container logs
	docker compose logs -f atelier

docker-shell:  ## Exec a shell inside the running container
	docker compose exec atelier bash

# ── Qdrant ops ────────────────────────────────────────────────────────────────

qdrant-drop:  ## Drop all atelier_* collections in QDRANT_URL (uses .env)
	@uv run python -c "import os; from dotenv import load_dotenv; from qdrant_client import QdrantClient; \
load_dotenv(); c = QdrantClient(url=os.environ['QDRANT_URL'], api_key=os.environ['QDRANT_API_KEY']); \
cols = [x.name for x in c.get_collections().collections if x.name.startswith('atelier_')]; \
[c.delete_collection(n) or print('dropped', n) for n in cols]; \
print('done' if cols else 'no atelier_* collections found')"

# ── Quality ───────────────────────────────────────────────────────────────────

format:  ## Auto-format all Python with ruff (rewrites files in place)
	uv run ruff format .
	uv run ruff check --fix --select I .

lint:  ## Lint with ruff; no rewrites
	uv run ruff check .
	uv run ruff format --check .

check:  ## Compile-check every Python file in the repo
	uv run python -m compileall -q backend app.py evaluate.py main.py pages
