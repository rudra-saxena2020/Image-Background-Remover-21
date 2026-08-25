"""
Product Background Remover — FastAPI application entry point.

BiRefNet is loaded once at startup and kept in memory.
"""

from __future__ import annotations

import logging
import os
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load BiRefNet at startup; nothing to clean up at shutdown."""
    from app.services.birefnet_service import birefnet
    logger.info("Starting up — loading BiRefNet model …")
    try:
        birefnet.load()
        logger.info("BiRefNet ready")
    except Exception as exc:
        logger.error("Failed to load BiRefNet: %s", exc, exc_info=True)
        # App still starts; /api/health will report model_loaded=false
    yield


app = FastAPI(
    title="Product Background Remover",
    description="AI-powered background removal for e-commerce product images using BiRefNet",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Inference-Time",
        "X-Total-Time",
        "X-Input-Width",
        "X-Input-Height",
        "X-Inference-Width",
        "X-Inference-Height",
        "X-Device",
        "X-Mode",
        "X-Cache-Hit",
        "X-History-Id",
        "Content-Disposition",
    ],
)


@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return response

# Mount routes under /api prefix (the shared proxy routes /api → this service)
from app.api.health import router as health_router
from app.api.background import router as background_router
from app.api.shoots import router as shoots_router
from app.api.generate import router as generate_router

app.include_router(health_router, prefix="/api")
app.include_router(background_router, prefix="/api")
app.include_router(shoots_router, prefix="/api")
app.include_router(generate_router, prefix="/api")


@app.get("/")
async def root():
    return {"service": "Product Background Remover API", "version": "1.0.0"}
