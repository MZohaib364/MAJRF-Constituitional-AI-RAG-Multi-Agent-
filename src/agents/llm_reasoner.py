from __future__ import annotations

import json
import os
from typing import Dict, Optional

from ..data_models import ComplianceState
from ..llm import GroqLLM
from .base import Agent


class LLMReasonerAgent(Agent):
    name = "LLMReasonerAgent"

    def __init__(
        self,
        enabled: bool = False,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.2,
    ) -> None:
        self.enabled = enabled and bool(os.getenv("GROQ_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.llm: Optional[GroqLLM] = None
        if self.enabled:
            try:
                self.llm = GroqLLM(model=model, temperature=temperature)
            except ValueError:
                self.enabled = False

    def run(self, state: ComplianceState) -> ComplianceState:
        if not self.enabled or not self.llm or not state.retrieved_articles:
            state.log.append({"agent": self.name, "status": "skipped"})
            setattr(self, "_log_written", True)
            return state

        # Build grounded prompt with actual article text to prevent hallucinations
        articles_context = self._build_articles_context(state)
        
        # Extract article IDs from context for strict validation
        article_ids = [article.article_id for article in state.retrieved_articles[:5]]
        article_ids_str = ", ".join(article_ids)
        
        system_prompt = (
            "You are a constitutional law expert for Pakistan. "
            "Analyze the policy text ONLY against the provided constitutional articles. "
            "CRITICAL RULES:\n"
            "1. You may ONLY reference these article IDs: " + article_ids_str + "\n"
            "2. DO NOT mention any article number not in the list above\n"
            "3. DO NOT make up article content or numbers\n"
            "4. If you want to reference an article not provided, DO NOT mention it at all\n"
            "5. Base your analysis strictly on the article text provided below\n\n"
            f"RELEVANT CONSTITUTIONAL ARTICLES:\n{articles_context}\n\n"
            "Produce a JSON analysis with these exact fields:\n"
            f"- articles_involved: list containing ONLY article IDs from this list: [{article_ids_str}]\n"
            "- risk_level: one of 'low', 'medium', 'high'\n"
            "- jurisdiction: one of 'federal', 'provincial', 'fundamental_rights'\n"
            "- reasoning: string explaining how the policy relates to the provided articles (cite specific article text from above)\n"
            "- recommended_actions: list of specific steps\n"
            "- confidence: float between 0 and 1\n\n"
            f"VALID ARTICLE IDs: {article_ids_str}\n"
            "If an article is not in this list, you MUST NOT reference it in any way."
        )
        
        user_prompt = (
            f"POLICY TEXT:\n{state.raw_text}\n\n"
            f"DETECTED DOMAINS: {', '.join(state.detected_domains)}\n\n"
            f"RULE-BASED CONFLICTS DETECTED: {len(state.conflicts)} conflict(s)\n"
            f"{self._format_conflicts(state.conflicts)}\n\n"
            "Analyze this policy against the constitutional articles provided above. "
            "Cite specific article text in your reasoning. Do not reference articles not provided."
        )

        analysis: Dict[str, object]
        try:
            analysis = self.llm.structured_response(system_prompt, user_prompt)
            # Validate that cited articles exist in retrieved articles
            analysis = self._validate_article_citations(analysis, state)
            state.llm_analysis = analysis
            setattr(self, "_log_written", True)
            state.log.append({"agent": self.name, "status": "completed"})
        except Exception as exc:  # noqa: BLE001
            state.log.append({"agent": self.name, "status": f"error: {exc}"})
            setattr(self, "_log_written", True)
        return state

    def _build_articles_context(self, state: ComplianceState) -> str:
        """Build context string from retrieved articles to ground LLM responses."""
        context_parts = []
        for i, article in enumerate(state.retrieved_articles[:5], 1):
            context_parts.append(
                f"Article {article.article_id} - {article.title}\n"
                f"Text: {article.text_snippet[:500]}...\n"
            )
        return "\n".join(context_parts)

    def _format_conflicts(self, conflicts: list) -> str:
        """Format conflicts for LLM prompt."""
        if not conflicts:
            return "No conflicts detected by rule-based system."
        parts = []
        for conflict in conflicts:
            parts.append(
                f"- Article {conflict.article_id} ({conflict.article_title}): "
                f"{conflict.description} [Confidence: {conflict.confidence:.2f}]"
            )
        return "\n".join(parts)

    def _validate_article_citations(self, analysis: Dict[str, object], state: ComplianceState) -> Dict[str, object]:
        """Remove any article citations that don't exist in retrieved articles to prevent hallucinations."""
        valid_article_ids = {article.article_id for article in state.retrieved_articles}
        
        if "articles_involved" in analysis:
            if isinstance(analysis["articles_involved"], list):
                # Filter to only include valid article IDs
                original = analysis["articles_involved"]
                analysis["articles_involved"] = [
                    str(aid) for aid in original if str(aid) in valid_article_ids
                ]
                # If all were invalid, use the first retrieved article
                if not analysis["articles_involved"] and state.retrieved_articles:
                    analysis["articles_involved"] = [state.retrieved_articles[0].article_id]
        
        # Also check reasoning text for invalid article references
        if "reasoning" in analysis and isinstance(analysis["reasoning"], str):
            reasoning = analysis["reasoning"]
            # Remove references to articles not in valid set
            import re
            article_refs = re.findall(r'Article\s+(\d+[A-Z]?)', reasoning)
            for ref in article_refs:
                if ref not in valid_article_ids:
                    reasoning = reasoning.replace(f"Article {ref}", f"[Article {ref} - not in retrieved articles]")
            analysis["reasoning"] = reasoning
        
        return analysis
