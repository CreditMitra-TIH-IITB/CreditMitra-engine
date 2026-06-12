from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import statements
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    description="Backend API for Credit Mitra Statement Processing",
    version="1.0.0",
)

# Set up CORS for React frontend (Vite defaults to 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "tauri://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    statements.router, prefix=f"{settings.API_V1_STR}/statements", tags=["statements"]
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
