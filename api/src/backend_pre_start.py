import logging
from tenacity import after_log, before_log, retry, stop_after_attempt, wait_fixed

from src.db.session import AsyncSessionLocal
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

max_tries = 60 * 5  # 5 minutes
wait_seconds = 1


@retry(
    stop=stop_after_attempt(max_tries),
    wait=wait_fixed(wait_seconds),
    before=before_log(logger, logging.INFO),
    after=after_log(logger, logging.WARN),
)
async def init() -> None:
    try:
        db = AsyncSessionLocal()
        # Try to create session to check if DB is awake
        await db.execute(text("SELECT 1"))
        await db.close()
    except Exception as e:
        logger.error(e)
        raise e


async def main() -> None:
    logger.info("Initializing service")
    await init()
    logger.info("Service finished initializing")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())