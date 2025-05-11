from fastapi import APIRouter

from .simulation import simulation_router

api_router = APIRouter(prefix="/api")
api_router.include_router(simulation_router)


@api_router.get("/health", status_code=200)
def health():
    return {"status": "ok"}
