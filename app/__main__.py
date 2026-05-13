from __future__ import annotations

import uvicorn

from app.config import settings

uvicorn.run("app.server:app", host=settings.host, port=settings.port)
