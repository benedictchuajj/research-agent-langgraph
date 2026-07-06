import asyncio
import logging
import time
from typing import Optional

import arxiv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import (
    ARXIV_BACKOFF_BASE,
    ARXIV_BACKOFF_CAP,
    ARXIV_MAX_RETRIES,
    ARXIV_MIN_INTERVAL,
)

logger = logging.getLogger(__name__)

_lock = asyncio.Lock()
_last_call_time: float = 0.0


class RateLimitedError(Exception):
    pass


@retry(
    retry=retry_if_exception_type((arxiv.UnexpectedEmptyPageError, TimeoutError, ConnectionError)),
    wait=wait_exponential(multiplier=ARXIV_BACKOFF_BASE, max=ARXIV_BACKOFF_CAP),
    stop=stop_after_attempt(ARXIV_MAX_RETRIES),
    reraise=True,
)
def _fetch_results(query: str, max_results: int) -> list[arxiv.Result]:
    client = arxiv.Client(page_size=max_results, delay_seconds=0, num_retries=2)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    return list(client.results(search))


async def search_arxiv(query: str, max_results: int = 20) -> list[dict]:
    global _last_call_time

    async with _lock:
        now = time.monotonic()
        wait_for = ARXIV_MIN_INTERVAL - (now - _last_call_time)
        if wait_for > 0:
            logger.debug("Rate limiter: waiting %.2fs", wait_for)
            await asyncio.sleep(wait_for)

        loop = asyncio.get_running_loop()
        try:
            results = await asyncio.wait_for(
                loop.run_in_executor(None, _fetch_results, query, max_results),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            raise RateLimitedError("arxiv request timed out")
        except Exception as e:
            logger.error("arxiv fetch failed: %s", e)
            raise RateLimitedError(f"arxiv fetch failed: {e}") from e

        _last_call_time = time.monotonic()

    papers = []
    for r in results:
        papers.append({
            "arxiv_id": r.entry_id.split("/abs/")[-1],
            "title": r.title,
            "authors": [a.name for a in r.authors],
            "published": r.published.isoformat() if r.published else "",
            "abstract": r.summary,
            "url": r.entry_id,
            "pdf_url": r.pdf_url,
        })
    return papers


async def fetch_paper(arxiv_id: str) -> Optional[dict]:
    global _last_call_time

    async with _lock:
        now = time.monotonic()
        wait_for = ARXIV_MIN_INTERVAL - (now - _last_call_time)
        if wait_for > 0:
            await asyncio.sleep(wait_for)

        loop = asyncio.get_running_loop()
        try:
            client = arxiv.Client(page_size=1, delay_seconds=0, num_retries=2)
            search = arxiv.Search(id_list=[arxiv_id])
            results = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: list(client.results(search))),
                timeout=30.0,
            )
        except Exception as e:
            logger.error("arxiv fetch failed for %s: %s", arxiv_id, e)
            raise RateLimitedError(str(e)) from e

        _last_call_time = time.monotonic()

    if not results:
        return None

    r = results[0]
    return {
        "arxiv_id": r.entry_id.split("/abs/")[-1],
        "title": r.title,
        "authors": [a.name for a in r.authors],
        "published": r.published.isoformat() if r.published else "",
        "abstract": r.summary,
        "url": r.entry_id,
        "pdf_url": r.pdf_url,
    }
