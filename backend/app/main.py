"""CausalIQ FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.campaigns import router as campaigns_router

app = FastAPI(
    title="CausalIQ API",
    description="Causal Inference and Uplift Modeling Platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(campaigns_router)


@app.get("/health")
def health() -> dict:
    """Return service liveness status."""
    return {"status": "ok", "service": "CausalIQ API"}


@app.get("/")
def root() -> dict:
    """Return API welcome message and docs link."""
    return {"message": "CausalIQ API", "docs": "/docs"}
