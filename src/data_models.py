from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PolicySegment:
    text: str
    domain_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class ArticleEvidence:
    article_id: str
    title: str
    text_snippet: str
    relevance: float
    category: str


@dataclass
class ConflictFinding:
    article_id: str
    article_title: str
    description: str
    severity: str
    jurisdiction: str
    confidence: float
    evidence: str


@dataclass
class Diagnosis:
    is_constitutional: bool
    primary_conflicts: List[str]
    recommended_actions: List[str]
    confidence: float
    summary: str


@dataclass
class ComplianceState:
    policy_id: str
    source: str
    raw_text: str
    segments: List[PolicySegment] = field(default_factory=list)
    detected_domains: List[str] = field(default_factory=list)
    retrieved_articles: List[ArticleEvidence] = field(default_factory=list)
    conflicts: List[ConflictFinding] = field(default_factory=list)
    diagnosis: Optional[Diagnosis] = None
    llm_analysis: Optional[Dict[str, object]] = None
    log: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "source": self.source,
            "raw_text": self.raw_text,
            "segments": [segment.__dict__ for segment in self.segments],
            "detected_domains": self.detected_domains,
            "retrieved_articles": [evidence.__dict__ for evidence in self.retrieved_articles],
            "conflicts": [conflict.__dict__ for conflict in self.conflicts],
            "diagnosis": self.diagnosis.__dict__ if self.diagnosis else None,
            "llm_analysis": self.llm_analysis,
            "log": self.log,
        }

