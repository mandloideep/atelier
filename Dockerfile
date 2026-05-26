FROM python:3.14-slim

WORKDIR /app

# ── Layer 0: system deps that some Python wheels need at install time ────────
# Most deps ship pure-wheels; build-essential covers the rare source build.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Layer 1: install deps (cached until requirements.txt changes) ─────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Layer 2: backend package ───────────────────────────────────────────────────
COPY backend/ backend/

# ── Layer 3: documents + bundled artifacts (goldens, eval baseline) ───────────
COPY documents/ documents/
COPY artifacts/ artifacts/

# ── Layer 4: application files (change most often — last for cache efficiency) ─
COPY app.py .
COPY evaluate.py .
COPY main.py .
COPY pages/ pages/

# Runtime state lives under /app/data/ (single mount root, three subpaths).
# Defaults baked in so the container works without explicit overrides; Dokploy
# can still override via the Environment tab if needed.
RUN mkdir -p /app/data/state /app/data/checkpoints /app/data/embedding_cache
ENV ATELIER_SESSIONS_FILE=/app/data/state/sessions.json \
    ATELIER_CHECKPOINTS_DB=/app/data/checkpoints/checkpoints.db \
    ATELIER_EMBEDDING_CACHE_DIR=/app/data/embedding_cache/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
