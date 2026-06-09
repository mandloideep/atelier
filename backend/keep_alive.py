import logging
import threading
import time

from backend.vector_store import qdrant_client

log = logging.getLogger(__name__)

# Qdrant Cloud free-tier clusters get suspended after a few days of
# inactivity. One cheap authenticated request per day is enough to reset
# the inactivity timer without showing up on the billed-traffic side.
_PING_INTERVAL_SECONDS = 24 * 60 * 60
_started = False
_lock = threading.Lock()


def _loop() -> None:
    while True:
        try:
            qdrant_client.get_collections()
            log.info("qdrant keep-alive ping ok")
        except Exception as e:
            log.warning("qdrant keep-alive ping failed: %s", e)
        time.sleep(_PING_INTERVAL_SECONDS)


def start() -> None:
    """Start the keep-alive thread once per process. Safe to call from
    Streamlit reruns — the lock + flag make repeat calls a no-op."""
    global _started
    with _lock:
        if _started:
            return
        _started = True
        t = threading.Thread(target=_loop, name="qdrant-keep-alive", daemon=True)
        t.start()
