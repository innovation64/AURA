"""Run AURA Server: python -m aura.server"""

import logging
import os


def main() -> None:
    import uvicorn

    host = os.getenv("AURA_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("AURA_SERVER_PORT", "8300"))
    log_level = os.getenv("AURA_LOG_LEVEL", "info").lower()

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    print(f"Starting AURA Server on http://{host}:{port}")
    print(f"API docs: http://{host}:{port}/docs")
    print(f"Health:   http://{host}:{port}/v1/health")

    uvicorn.run(
        "aura.server.app:app",
        host=host,
        port=port,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
