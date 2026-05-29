from __future__ import annotations

import uvicorn

from src.config import load_config


def run() -> None:
    cfg = load_config()
    uvicorn.run(
        "src.service.app:app",
        host=cfg.service.host,
        port=cfg.service.port,
        reload=False,
    )


if __name__ == "__main__":
    run()
