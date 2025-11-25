from __future__ import annotations

from statistics import mean
from typing import List

from ..data_models import ComplianceState, Diagnosis
from .base import Agent


class FinalComplianceAgent(Agent):
    name = "FinalComplianceAgent"

    def __init__(self, high_confidence_threshold: float = 0.75):
        self.high_confidence_threshold = high_confidence_threshold

    def run(self, state: ComplianceState) -> ComplianceState:
        conflicts = state.conflicts
        llm_analysis = state.llm_analysis or {}
        if not conflicts and not llm_analysis:
            state.diagnosis = Diagnosis(
                is_constitutional=True,
                primary_conflicts=[],
                recommended_actions=["Maintain documentation for audit.", "Monitor for future amendments."],
                confidence=0.7,
                summary="No conflicts detected based on current heuristic rules.",
            )
            return state

        conflict_ids = [conflict.article_id for conflict in conflicts]
        confidence_scores = [conflict.confidence for conflict in conflicts]
        average_confidence = mean(confidence_scores) if confidence_scores else 0.5
        llm_confidence = float(llm_analysis.get("confidence", 0)) if isinstance(llm_analysis, dict) else 0.0
        combined_confidence = max(average_confidence, llm_confidence)
        risk_level = (llm_analysis.get("risk_level") or "").lower() if isinstance(llm_analysis, dict) else ""
        is_constitutional = (
            max(confidence_scores) < self.high_confidence_threshold
            if confidence_scores
            else risk_level in {"low", ""}
        )

        recommendations = [
            "Seek legal review for the highlighted articles.",
            "Revise policy language to incorporate due process safeguards.",
            "Coordinate with relevant jurisdictional authority before enforcement.",
        ]
        if isinstance(llm_analysis, dict) and isinstance(llm_analysis.get("recommended_actions"), list):
            recommendations = llm_analysis["recommended_actions"]

        state.diagnosis = Diagnosis(
            is_constitutional=is_constitutional,
            primary_conflicts=conflict_ids,
            recommended_actions=recommendations,
            confidence=combined_confidence or 0.5,
            summary=llm_analysis.get("reasoning", "Conflicts detected with constitutional provisions.")
            if isinstance(llm_analysis, dict)
            else "Conflicts detected with constitutional provisions.",
        )
        return state

