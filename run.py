"""Dev entrypoint: `python run.py` — serves the proxy + cockpit on PORT (4321)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import uvicorn  # noqa: E402

from tokenoptimizer.config import settings  # noqa: E402

if __name__ == "__main__":
    print(f"TokenOptimizer · mode={settings.mode} · http://localhost:{settings.port}")
    uvicorn.run(
        "tokenoptimizer.app:app",
        host=settings.host,
        port=settings.port,
        reload=bool(os.getenv("RELOAD")),
    )
