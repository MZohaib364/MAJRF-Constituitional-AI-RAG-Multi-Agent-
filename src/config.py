from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Article:
    """Structured representation of a constitutional article."""

    article_id: str
    title: str
    text: str
    category: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "article_id": self.article_id,
            "title": self.title,
            "text": self.text,
            "category": self.category,
        }


@dataclass
class Rulebook:
    """Container for rulebook configuration."""

    metadata: Dict[str, str]
    constitutional_articles: Dict[str, Dict[str, object]]
    conflict_rules: Dict[str, Dict[str, str]]
    legislative_lists: Dict[str, Dict[str, object]]
    subject_keywords: Dict[str, List[str]]
    policy_domains: Dict[str, Dict[str, object]]
    jurisdiction_mapping: Dict[str, str]


@dataclass
class ProjectConfig:
    """Global configuration object for the compliance checker."""

    root_dir: Path
    rulebook: Rulebook
    articles: List[Article] = field(default_factory=list)

    @property
    def article_lookup(self) -> Dict[str, Article]:
        return {article.article_id: article for article in self.articles}


def load_project_config(
    root_dir: str | Path = ".",
    rulebook_path: str | Path = "Data/rules/rulebook.json",
    constitution_dir: str | Path = "Data/constitution",
) -> ProjectConfig:
    base_path = Path(root_dir).resolve()
    rulebook = _load_rulebook(base_path / rulebook_path)
    articles = _load_constitution_articles(base_path / constitution_dir)
    return ProjectConfig(root_dir=base_path, rulebook=rulebook, articles=articles)


def _load_rulebook(path: Path) -> Rulebook:
    if not path.exists():
        raise FileNotFoundError(f"Rulebook not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return Rulebook(
        metadata=data.get("metadata", {}),
        constitutional_articles=data.get("constitutional_articles", {}),
        conflict_rules=data.get("conflict_resolution_rules", {}),
        legislative_lists=data.get("legislative_lists", {}),
        subject_keywords=data.get("subject_keywords", {}),
        policy_domains=data.get("policy_domain_classification", {}),
        jurisdiction_mapping=data.get("jurisdiction_mapping", {}),
    )


# Match Article with various dash types: em dash (—), en dash (–), or regular dash (-)
ARTICLE_PATTERN = re.compile(r"^Article\s+([0-9A-Za-z]+)\s+[–—\-]\s+(.*)$")


def _load_constitution_articles(directory: Path) -> List[Article]:
    if not directory.exists():
        raise FileNotFoundError(f"Constitution directory not found: {directory}")

    articles: List[Article] = []
    for file_path in sorted(directory.glob("*.txt")):
        category = file_path.stem
        content = file_path.read_text(encoding="utf-8")
        articles.extend(_parse_articles_from_text(content, category))
    return articles


def _parse_articles_from_text(text: str, category: str) -> List[Article]:
    results: List[Article] = []
    current_id: Optional[str] = None
    current_title: Optional[str] = None
    buffer: List[str] = []

    def flush():
        nonlocal current_id, current_title, buffer
        if current_id and current_title and buffer:
            results.append(
                Article(
                    article_id=current_id,
                    title=current_title.strip(),
                    text="\n".join(buffer).strip(),
                    category=category,
                )
            )
        current_id, current_title, buffer = None, None, []

    for line in text.splitlines():
        match = ARTICLE_PATTERN.match(line.strip())
        if match:
            flush()
            current_id, current_title = match.group(1), match.group(2)
        else:
            buffer.append(line)

    flush()
    return results

