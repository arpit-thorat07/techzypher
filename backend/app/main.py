from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routes.generation import router as generation_router

# Initialize storage directories on startup
settings.init_directories()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "FastAPI Backend for Sketch2Shape AI Studio. "
        "Accepts sketches/images and manages 3D model generation pipelines."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS Middleware for React frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes under /api
app.include_router(generation_router, prefix=settings.API_PREFIX)


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint verifying backend availability."""
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "docs_url": "/docs",
    }


@app.get(f"{settings.API_PREFIX}/health", tags=["Health"])
async def health_check():
    """Health check endpoint for status monitoring."""
    return {
        "status": "healthy",
        "uploads_dir_exists": settings.UPLOADS_DIR.is_dir(),
        "outputs_dir_exists": settings.OUTPUTS_DIR.is_dir(),
    }
