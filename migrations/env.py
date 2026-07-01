import asyncio
import ssl
from logging.config import fileConfig
from urllib.parse import urlparse, urlunparse

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _async_url() -> tuple[str, bool]:
    """Return (url, needs_ssl).

    Strips all query params from the URL — asyncpg dialect rejects unknown
    params like sslmode, channel_binding, pgbouncer, etc. that Neon includes.
    SSL is passed separately via connect_args.
    """
    from app.settings import settings

    url = str(settings.database_url)
    url = (
        url.replace("postgresql://", "postgresql+asyncpg://")
        .replace("postgres://", "postgresql+asyncpg://")
    )
    parsed = urlparse(url)
    needs_ssl = "sslmode=disable" not in (parsed.query or "")
    clean_url = urlunparse(parsed._replace(query=""))
    return clean_url, needs_ssl


def run_migrations_offline() -> None:
    url, _ = _async_url()
    context.configure(
        url=url,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection):  # type: ignore[no-untyped-def]
    context.configure(connection=connection)
    with context.begin_transaction():
        context.run_migrations()


async def _run_async_migrations() -> None:
    url, needs_ssl = _async_url()
    if needs_ssl:
        # macOS Python (python.org installer) lacks bundled CA certs.
        # Connection is still encrypted — we skip host cert verification only.
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        connect_args: dict = {"ssl": ssl_ctx}
    else:
        connect_args = {}
    engine = create_async_engine(url, connect_args=connect_args)
    async with engine.connect() as conn:
        await conn.run_sync(_do_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    asyncio.run(_run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
