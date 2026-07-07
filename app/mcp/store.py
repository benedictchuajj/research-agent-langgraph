import json
import logging
from pathlib import Path
from typing import Optional

import frontmatter

from app.config import INDEX_DIR, PAPERS_DIR

logger = logging.getLogger(__name__)

_REJECTIONS_FILE = INDEX_DIR / "rejections.json"


def _paper_path(arxiv_id: str) -> Path:
    return PAPERS_DIR / f"{arxiv_id}.md"


def paper_exists(arxiv_id: str) -> bool:
    return _paper_path(arxiv_id).exists()


def write_paper(
    arxiv_id: str,
    title: str,
    authors: list[str],
    published: str,
    subtopics: list[str],
    url: str,
    abstract: str,
    summary: str,
) -> Path:
    path = _paper_path(arxiv_id)

    post = frontmatter.Post("")
    post.metadata = {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": authors,
        "published": published,
        "subtopics": subtopics,
        "url": url,
    }

    content = f"## Abstract\n\n{abstract}\n\n## Summary\n\n{summary}\n"
    post.content = content

    with open(path, "w") as f:
        frontmatter.dump(post, f)

    logger.info("Wrote paper %s to %s", arxiv_id, path)
    return path


def read_paper(arxiv_id: str) -> Optional[dict]:
    path = _paper_path(arxiv_id)
    if not path.exists():
        return None

    post = frontmatter.load(path)
    return {
        "arxiv_id": post.metadata.get("arxiv_id", arxiv_id),
        "title": post.metadata.get("title", ""),
        "authors": post.metadata.get("authors", []),
        "published": post.metadata.get("published", ""),
        "subtopics": post.metadata.get("subtopics", []),
        "url": post.metadata.get("url", ""),
        "abstract": post.content.split("## Summary")[0].replace("## Abstract", "").strip(),
        "summary": post.content.split("## Summary")[-1].strip() if "## Summary" in post.content else "",
        "path": str(path),
    }


def list_papers(subtopics: Optional[list[str]] = None) -> list[dict]:
    papers = []
    for path in sorted(PAPERS_DIR.glob("*.md")):
        try:
            post = frontmatter.load(path)
            meta = dict(post.metadata)
            meta["path"] = str(path)

            if subtopics:
                paper_topics = set(meta.get("subtopics", []))
                if not paper_topics.intersection(set(subtopics)):
                    continue

            papers.append(meta)
        except Exception as e:
            logger.warning("Failed to read %s: %s", path, e)
            continue

    return papers


def read_rejections() -> dict:
    if not _REJECTIONS_FILE.exists():
        return {}
    try:
        with open(_REJECTIONS_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read rejections: %s", e)
        return {}


def is_rejected(arxiv_id: str) -> bool:
    return arxiv_id in read_rejections()


def write_rejection(arxiv_id: str, reason: str) -> None:
    rejections = read_rejections()
    rejections[arxiv_id] = reason
    try:
        with open(_REJECTIONS_FILE, "w") as f:
            json.dump(rejections, f, indent=2)
        logger.info("Recorded rejection for %s: %s", arxiv_id, reason)
    except OSError as e:
        logger.error("Failed to write rejections: %s", e)


def write_digest(markdown: str) -> Path:
    path = PAPERS_DIR / "digest.md"
    with open(path, "w") as f:
        f.write(markdown)
    logger.info("Wrote digest to %s", path)
    return path
