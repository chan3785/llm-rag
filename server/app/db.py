from collections.abc import AsyncIterator

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for ORM models. Define tables by subclassing this."""


_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def init_db(database_url: str | None) -> None:
    global _engine, _sessionmaker
    if not database_url:
        return
    _engine = create_async_engine(database_url, pool_pre_ping=True)
    _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)


async def close_db() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None


async def check_db() -> str:
    """Returns "ok", "not_configured", or "error: <reason>"."""
    if _engine is None:
        return "not_configured"
    try:
        async with _engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return "ok"
    except Exception as exc:
        return f"error: {exc.__class__.__name__}"


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: `session: AsyncSession = Depends(get_session)`."""
    if _sessionmaker is None:
        raise HTTPException(status_code=503, detail="Database is not configured (DATABASE_URL).")
    async with _sessionmaker() as session:
        yield session
