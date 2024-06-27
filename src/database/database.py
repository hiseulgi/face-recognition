import rootutils

ROOT = rootutils.autosetup()

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session

from src.utils.logger import get_logger

log = get_logger()

Base = declarative_base()


class Database:

    def __init__(self, host: str, port: int, user: str, password: str, db: str) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db = db

    def connect(self) -> None:
        try:
            self.engine = create_engine(
                f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"
            )
            self.SessionLocal = sessionmaker(
                autocommit=False, autoflush=False, bind=self.engine, class_=Session
            )

            self.async_engine = create_async_engine(
                f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"
            )
            self.SessionAsync = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.async_engine,
                class_=AsyncSession,
            )
            log.info("Connected to database")
        except Exception as e:
            log.error(f"Error connecting to database: {e}")

    def setup_pgvector(self) -> None:
        with self.engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            log.info("pgvector extension created")

    def get_db(self):
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    async def get_async_db(self):
        async with self.SessionAsync() as db:
            try:
                yield db
            finally:
                await db.close()

    def close(self) -> None:
        self.engine.dispose()
        log.info("Database connection closed")

    async def async_close(self) -> None:
        await self.async_engine.dispose()
        log.info("Async database connection closed")
