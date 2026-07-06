import logging
from pathlib import Path
from typing import Optional

import yaml

from app.config import SUBTOPICS_FILE

logger = logging.getLogger(__name__)

_taxonomy: Optional[dict] = None


def load_taxonomy() -> dict:
    global _taxonomy
    if _taxonomy is not None:
        return _taxonomy

    path = Path(SUBTOPICS_FILE)
    if not path.exists():
        logger.warning("subtopics.yaml not found at %s, using empty taxonomy", path)
        _taxonomy = {}
        return _taxonomy

    with open(path) as f:
        _taxonomy = yaml.safe_load(f) or {}
    return _taxonomy


def match_subtopics(text: str) -> list[str]:
    taxonomy = load_taxonomy()
    if not taxonomy:
        return []

    text_lower = text.lower()
    matches = []

    for topic_name, topic_data in taxonomy.items():
        keywords = topic_data.get("keywords", [])
        for keyword in keywords:
            if keyword.lower() in text_lower:
                matches.append(topic_name)
                break

    return matches


def get_all_subtopics() -> list[str]:
    return list(load_taxonomy().keys())


def get_keywords_for_subtopic(subtopic: str) -> list[str]:
    taxonomy = load_taxonomy()
    topic_data = taxonomy.get(subtopic, {})
    return topic_data.get("keywords", [])
