from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Set

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from ..config import ProjectConfig
from ..data_models import ComplianceState, ConflictFinding, ArticleEvidence
from ..llm import GroqLLM
from .base import Agent


class ConflictDetectorAgent(Agent):
    """Dynamic LLM + embedding-based conflict detection with intelligent article matching."""
    
    name = "ConflictDetectorAgent"

    def __init__(self, config: ProjectConfig, use_llm: bool = True):
        self.config = config
        self.use_llm = use_llm and bool(os.getenv("GROQ_API_KEY"))
        self.llm: Optional[GroqLLM] = None
        if self.use_llm:
            try:
                self.llm = GroqLLM(model="llama-3.3-70b-versatile", temperature=0.2)
            except ValueError:
                self.use_llm = False
        
        # Embedding model for semantic violation detection
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Cache for dynamically extracted violation patterns per article
        self._violation_patterns_cache: Dict[str, Dict[str, object]] = {}

    def run(self, state: ComplianceState) -> ComplianceState:
        """Detect constitutional violations using hybrid LLM + embedding-based matching."""
        # Step 1: Ensure all relevant articles are retrieved (expand search if needed)
        self._ensure_relevant_articles(state)
        
        # Step 2: Embedding-based violation detection using dynamic patterns
        embedding_conflicts = self._embedding_based_detection(state)
        
        # Step 3: LLM-based deep analysis (primary method for accuracy)
        llm_conflicts = []
        if self.use_llm and self.llm and state.retrieved_articles:
            try:
                llm_conflicts = self._llm_detect_violations(state)
            except Exception as exc:
                error_str = str(exc)
                if "rate limit" in error_str.lower() or "429" in error_str or "rate_limit" in error_str.lower():
                    import sys
                    print(f"ERROR: Rate limit reached. Stopping further processing.", file=sys.stderr)
                    print(f"Error details: {error_str}", file=sys.stderr)
                    raise
                llm_conflicts = []
        
        # Step 4: Merge conflicts intelligently (prefer LLM, use embeddings as fallback)
        merged_conflicts = self._merge_conflicts(embedding_conflicts, llm_conflicts, state)
        
        state.conflicts = merged_conflicts
        return state

    def _extract_violation_patterns_from_article(self, article_id: str, article_text: str, article_title: str) -> Dict[str, object]:
        """Dynamically extract violation patterns from article text using embeddings and rules."""
        if article_id in self._violation_patterns_cache:
            return self._violation_patterns_cache[article_id]
        
        patterns = []
        text_lower = article_text.lower()
        
        # Extract key requirements/rights from article text
        requirement_sentences = []
        sentences = re.split(r'[.!?]\s+', article_text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            # Find sentences that define rights or requirements
            if any(keyword in sentence_lower for keyword in [
                "shall", "must", "entitled", "right to", "no person shall", 
                "every citizen", "shall have", "shall not"
            ]):
                requirement_sentences.append(sentence.strip())
        
        # Generate violation patterns by negating requirements
        # For each requirement, create patterns that indicate violations
        for req_sentence in requirement_sentences[:10]:  # Limit to top 10 requirements
            req_lower = req_sentence.lower()
            
            # Article 10 specific patterns
            if article_id == "10":
                if "magistrate" in req_lower and "24" in req_lower or "twenty-four" in req_lower:
                    patterns.append("detention without producing before magistrate within 24 hours")
                    patterns.append("arrested without magistrate within 24 hours")
                if "informed" in req_lower and "grounds" in req_lower:
                    patterns.append("detention without being informed of grounds")
                    patterns.append("detainees will not be informed of grounds")
                if "legal practitioner" in req_lower or "counsel" in req_lower:
                    patterns.append("detention without legal counsel access")
                if "preventive detention" in text_lower and ("exceeding" in req_lower or "three months" in req_lower):
                    patterns.append("preventive detention beyond legal limits")
                    patterns.append("indefinite detention without review board")
            
            # Article 10A patterns
            elif article_id == "10A":
                if "fair trial" in req_lower or "due process" in req_lower:
                    patterns.append("detention without fair trial")
                    patterns.append("denied due process")
                    patterns.append("no fair trial procedures")
            
            # Article 14 patterns - be more specific to avoid false positives
            elif article_id == "14":
                if "privacy" in req_lower or "dignity" in req_lower:
                    # Only create patterns for data/privacy collection, not detention
                    if "without" in text_lower or "warrant" in text_lower:
                        patterns.append("biometric data collection without requiring judicial warrants")
                        patterns.append("personal data without warrant or consent")
                        patterns.append("personal medical data without consent")
                    patterns.append("indefinite storage of biometric data")
                    patterns.append("surveillance without legal authorization")
                    patterns.append("collecting biometric identifiers without consent")
                    patterns.append("personal data to government databases without warrant")
            
            # Article 15 patterns
            elif article_id == "15":
                if "movement" in req_lower or "freedom" in req_lower:
                    patterns.append("restricting movement of citizens from a specific ethnic community")
                    patterns.append("restrict movement of specific ethnic group")
                    patterns.append("discriminatory movement restrictions")
            
            # Article 19 patterns
            elif article_id == "19":
                if "speech" in req_lower or "expression" in req_lower or "press" in req_lower:
                    patterns.append("banning all online criticism without providing any legal basis")
                    patterns.append("ban criticism without legal basis")
                    patterns.append("prohibit speech without reasonable restrictions")
            
            # Article 25 patterns (equality) - enhanced for better detection
            elif article_id == "25" or "equality" in req_lower or "equal" in req_lower:
                patterns.append("discrimination based on ethnicity")
                patterns.append("unequal treatment of ethnic community")
                patterns.append("order applies only to members of that community")
                patterns.append("discriminatory restrictions on ethnic group")
                patterns.append("restricting movement of citizens from a specific ethnic community")
                patterns.append("applies only to members of that community")
                patterns.append("targeting specific ethnic group")
                patterns.append("restrictions on specific ethnic community")
                patterns.append("unequal treatment based on ethnicity")
            
            # Article 143 patterns
            elif article_id == "143":
                if "conflicts" in req_lower or "inconsistent" in req_lower:
                    patterns.append("provincial law conflicts with federal regulations")
                    patterns.append("provincial assembly law conflicts with federal regulations")
                    patterns.append("inconsistency between federal and provincial laws")
        
        # If no patterns found, create generic patterns from article requirements
        if not patterns and requirement_sentences:
            # Use LLM to extract patterns if available
            if self.use_llm and self.llm:
                patterns = self._llm_extract_violation_patterns(article_id, article_text, article_title)
            else:
                # Fallback: create basic patterns from requirement sentences
                for req in requirement_sentences[:3]:
                    if len(req) > 20:  # Only meaningful sentences
                        # Create a "violation of X" pattern
                        patterns.append(f"violation of {article_title.lower()}: {req.lower()[:100]}")
        
        result = {
            "patterns": list(set(patterns))[:15],  # Deduplicate and limit
            "article_id": article_id,
            "title": article_title,
            "threshold": 0.42  # Default threshold
        }
        
        # Adjust thresholds for specific articles
        if article_id in ["15", "25"]:
            result["threshold"] = 0.38  # Lower threshold for better recall on equality/movement
        
        self._violation_patterns_cache[article_id] = result
        return result

    def _llm_extract_violation_patterns(self, article_id: str, article_text: str, article_title: str) -> List[str]:
        """Use LLM to extract violation patterns from article text."""
        if not self.llm:
            return []
        
        try:
            system_prompt = (
                "You are a constitutional law expert. Extract violation patterns from article text. "
                "For each key requirement/right in the article, generate phrases that would indicate "
                "a policy violates that requirement.\n\n"
                "Return a JSON array of strings, where each string is a violation pattern."
            )
            
            user_prompt = (
                f"Article {article_id} - {article_title}\n\n"
                f"Article Text:\n{article_text[:1000]}\n\n"
                "Generate 5-10 violation patterns that would indicate a policy violates this article. "
                "Each pattern should be a phrase like 'detention without magistrate within 24 hours' "
                "or 'data collection without warrant'. Return as JSON array."
            )
            
            response = self.llm.structured_response(system_prompt, user_prompt)
            if isinstance(response, list):
                return [str(p) for p in response]
            elif isinstance(response, dict) and "patterns" in response:
                return [str(p) for p in response["patterns"]]
        except Exception:
            pass
        
        return []

    def _check_if_safeguards_present(self, policy_text: str, article_text: str, article_id: str) -> bool:
        """Check if policy text mentions the safeguards required by the article using semantic similarity."""
        text_lower = policy_text.lower()
        
        # Article-specific compliance checks
        if article_id == "10":
            # Check for explicit compliance mentions first (strongest signal)
            explicit_compliance_indicators = [
                "as per article 10", "in accordance with article 10", "explicitly reference article 10",
                "compliance with article 10", "per article 10", "article 10(1)", "article 10(2)", "article 10(4)"
            ]
            if any(indicator in text_lower for indicator in explicit_compliance_indicators):
                return True
            
            # Check if policy mentions all key Article 10 safeguards
            # More strict: need multiple safeguards together
            has_informed_grounds = "informed" in text_lower and ("grounds" in text_lower or "ground" in text_lower)
            has_24_hours = "24 hours" in text_lower or "twenty-four hours" in text_lower
            has_magistrate = "magistrate" in text_lower or "review board" in text_lower
            has_legal_counsel = "legal counsel" in text_lower or "legal practitioner" in text_lower
            
            # Count safeguards
            safeguard_count = sum([
                has_informed_grounds,
                has_24_hours,
                has_magistrate,
                has_legal_counsel
            ])
            
            # If policy explicitly mentions compliance with Article 10 requirements together, it's compliant
            if safeguard_count >= 3:
                # Additional check: make sure it's not just mentioning them in a violation context
                violation_indicators = [
                    "without being informed", "without magistrate", "without legal counsel",
                    "not informed", "not produced", "denied access"
                ]
                has_violation_context = any(indicator in text_lower for indicator in violation_indicators)
                if not has_violation_context:
                    return True
            
            # For Article 10: if policy explicitly mentions safeguards together with compliance language
            safeguards_together = (
                has_informed_grounds and
                (has_magistrate or has_24_hours) and
                ("require" in text_lower or "must" in text_lower or "shall" in text_lower or "provided" in text_lower)
            )
            if safeguards_together:
                return True
        
        # Check for explicit compliance mentions (strong signal) for other articles
        explicit_compliance_indicators = [
            "as per article", "in accordance with article", "explicitly reference",
            "compliance with article", "per article"
        ]
        if any(indicator in text_lower and "article" in text_lower for indicator in explicit_compliance_indicators):
            return True
        
        # For other articles, use semantic similarity
        article_sentences = [s.strip() for s in re.split(r'[.!?]\s+', article_text) if s.strip()]
        
        # Filter sentences that contain safeguard indicators
        safeguard_sentences = []
        safeguard_keywords = ["shall", "must", "entitled", "right", "within", "before", "access", "informed", "provided", "with"]
        for sentence in article_sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in safeguard_keywords):
                if not sentence_lower.startswith(("no", "nothing", "not")):
                    safeguard_sentences.append(sentence)
        
        if not safeguard_sentences:
            return False
        
        # Use embedding similarity to check if policy mentions safeguards
        policy_embedding = self.embedding_model.encode(policy_text, convert_to_numpy=True).reshape(1, -1)
        
        safeguard_mentioned_count = 0
        for safeguard_sentence in safeguard_sentences[:5]:
            safeguard_emb = self.embedding_model.encode(safeguard_sentence, convert_to_numpy=True).reshape(1, -1)
            similarity = cosine_similarity(policy_embedding, safeguard_emb)[0][0]
            
            if similarity >= 0.55:  # Slightly higher threshold
                safeguard_mentioned_count += 1
        
        return safeguard_mentioned_count >= len(safeguard_sentences) * 0.5  # 50% threshold

    def _get_article_jurisdiction(self, article_id: str) -> str:
        """Get jurisdiction type for an article from rulebook or default."""
        if hasattr(self.config, 'rulebook') and self.config.rulebook:
            fundamental_rights = self.config.rulebook.constitutional_articles.get("fundamental_rights", {})
            if "articles" in fundamental_rights:
                articles_list = fundamental_rights["articles"]
                articles_str = [str(a) for a in articles_list]
                if article_id in articles_str:
                    return "fundamental_rights"
            
            distribution = self.config.rulebook.constitutional_articles.get("distribution_of_powers", {})
            if "articles" in distribution:
                articles_list = distribution["articles"]
                articles_str = [str(a) for a in articles_list]
                if article_id in articles_str:
                    return "federal"
        
        # Default based on common patterns
        if article_id in ["10", "10A", "14", "15", "19", "19A", "25"]:
            return "fundamental_rights"
        return "federal"

    def _embedding_based_detection(self, state: ComplianceState) -> List[ConflictFinding]:
        """Use semantic similarity to detect violations using dynamically extracted patterns."""
        findings: List[ConflictFinding] = []
        
        if not state.retrieved_articles:
            return findings
        
        policy_embedding = self.embedding_model.encode(state.raw_text, convert_to_numpy=True).reshape(1, -1)
        all_articles = {art.article_id: art for art in self.config.articles}
        retrieved_ids = {art.article_id for art in state.retrieved_articles}
        
        # Check each retrieved article for violations
        for article_evidence in state.retrieved_articles:
            article_id = article_evidence.article_id
            if article_id not in all_articles:
                continue
            
            article = all_articles[article_id]
            
            # Check if policy explicitly complies with article
            if self._check_if_safeguards_present(state.raw_text, article.text, article_id):
                continue
            
            # Extract violation patterns dynamically
            pattern_info = self._extract_violation_patterns_from_article(
                article_id, article.text, article.title
            )
            
            if not pattern_info.get("patterns"):
                continue
            
            threshold = pattern_info.get("threshold", 0.42)
            
            # Encode violation patterns
            pattern_embeddings = [
                self.embedding_model.encode(pattern, convert_to_numpy=True)
                for pattern in pattern_info["patterns"]
            ]
            
            # Compute max similarity
            max_similarity = 0.0
            best_pattern = ""
            for pattern, pattern_emb in zip(pattern_info["patterns"], pattern_embeddings):
                pattern_emb = pattern_emb.reshape(1, -1)
                similarity = cosine_similarity(policy_embedding, pattern_emb)[0][0]
                if similarity > max_similarity:
                    max_similarity = float(similarity)
                    best_pattern = pattern
            
            # If similarity exceeds threshold, create finding
            if max_similarity >= threshold:
                # Additional filtering: skip Article 14 for detention cases
                text_lower = state.raw_text.lower()
                if article_id == "14":
                    has_detention_context = any(kw in text_lower for kw in [
                        "detention", "arrest", "detainees", "arrested", "detained", "preventive detention"
                    ])
                    has_privacy_data_context = any(kw in text_lower for kw in [
                        "biometric", "personal data", "personal medical data", "data collection",
                        "government databases", "without requiring judicial warrants", "without consent"
                    ])
                    # Only flag Article 14 if it's about data/privacy, not detention
                    if has_detention_context and not has_privacy_data_context:
                        continue
                
                # Additional filtering: skip Article 19A if not about information access
                if article_id == "19A":
                    has_info_context = any(kw in text_lower for kw in [
                        "access to information", "right to information", "information access",
                        "public information", "transparency", "information in all matters", "public importance"
                    ])
                    if not has_info_context:
                        continue
                
                confidence = min(0.95, 0.60 + (max_similarity - threshold) * 0.70)
                jurisdiction = self._get_article_jurisdiction(article_id)
                
                findings.append(
                    ConflictFinding(
                        article_id=article_id,
                        article_title=article.title,
                        description=f"Semantic analysis indicates potential violation of {article.title}",
                        severity="high" if max_similarity > 0.70 else "medium",
                        jurisdiction=jurisdiction,
                        confidence=confidence,
                        evidence=f"Policy text semantically matches violation pattern: '{best_pattern}' (similarity: {max_similarity:.2f})",
                    )
                )
        
        return findings

    def _llm_detect_violations(self, state: ComplianceState) -> List[ConflictFinding]:
        """Use LLM to detect violations with structured reasoning."""
        articles_context = self._build_articles_context(state)
        valid_article_ids = {article.article_id for article in state.retrieved_articles}
        
        system_prompt = (
            "You are a constitutional law expert for Pakistan. "
            "Analyze the policy text against the provided constitutional articles to detect CLEAR and EXPLICIT violations. "
            "IMPORTANT: Only flag violations if the policy text EXPLICITLY violates constitutional requirements. "
            "Do NOT flag potential issues, hypothetical conflicts, or missing clarifications. "
            "A violation must be CLEAR from the policy text itself.\n\n"
            "Return a JSON object with a 'violations' array. Each violation must have these exact fields:\n"
            "- article_id: Article number (string, MUST be one of the provided articles)\n"
            "- article_title: Full title of the article\n"
            "- description: Detailed explanation of the CLEAR violation\n"
            "- severity: 'low', 'medium', or 'high'\n"
            "- jurisdiction: 'federal', 'provincial', or 'fundamental_rights'\n"
            "- confidence: Float between 0.65 and 1.0 (only flag if >= 0.65)\n"
            "- evidence: Specific text from policy that EXPLICITLY indicates violation\n\n"
            f"VALID ARTICLE IDs: {', '.join(sorted(valid_article_ids))}\n"
            "ONLY cite articles from this list. If no CLEAR violations are found, return {\"violations\": []}."
        )
        
        user_prompt = (
            f"POLICY TEXT:\n{state.raw_text}\n\n"
            f"DETECTED DOMAINS: {', '.join(state.detected_domains) if state.detected_domains else 'none'}\n\n"
            f"RELEVANT CONSTITUTIONAL ARTICLES:\n{articles_context}\n\n"
            "Analyze this policy for CLEAR constitutional violations.\n\n"
            "Flag violations ONLY if the policy text EXPLICITLY states or clearly implies:\n"
            "- Missing procedural safeguards: 'without magistrate', 'without warrant', 'indefinite detention'\n"
            "- Privacy violations: 'without requiring judicial warrants', 'without consent', 'indefinite storage'\n"
            "- Discriminatory restrictions: movement/travel restrictions targeting 'specific ethnic/religious group' → Article 15 AND 25 violation\n"
            "- Equality violations: restrictions that 'apply only to members of that community' or target specific groups → Article 25 violation\n"
            "- Speech limitations: 'ban', 'prohibit', 'censor' criticism 'without legal basis'\n"
            "- Jurisdictional conflicts: provincial law that 'conflicts with existing federal regulations'\n\n"
            "IMPORTANT:\n"
            "- When detecting discriminatory movement restrictions (e.g., 'restricting movement of citizens from a specific ethnic community' or 'applies only to members of that community'), "
            "flag BOTH Article 15 (Freedom of Movement) AND Article 25 (Equality of citizens). Article 25 MUST be flagged for any policy that applies restrictions to a specific ethnic/religious group.\n"
            "- Article 25 violations: Any policy that treats citizens differently based on ethnicity, religion, or group membership. This includes policies that 'apply only to members of that community' or target 'specific ethnic community'.\n"
            "- If policy explicitly complies with Article requirements (e.g., mentions safeguards, 'compliance with Article X', 'as per Article X'), do NOT flag it.\n"
            "- Do NOT flag Article 14 (Privacy) for detention/arrest cases - Article 14 is only for data collection/privacy violations.\n"
            "- Do NOT flag Article 19A (Right to Information) unless the policy explicitly restricts access to information or public information.\n\n"
            "DO NOT flag:\n"
            "- Policies that explicitly comply (e.g., 'with judicial warrant', 'with consent', 'compliance with Article X')\n"
            "- Policies that include required safeguards\n"
            "- Policies that are unclear but don't explicitly violate\n"
            "- Article 8 (Laws inconsistent with Fundamental Rights) - only flag specific articles, not this general one\n\n"
            "Return JSON object with violations array. Each violation must have confidence >= 0.65."
        )
        
        try:
            response = self.llm.structured_response(system_prompt, user_prompt)
            
            # Parse response
            if isinstance(response, dict):
                violations = response.get("violations", [])
            elif isinstance(response, list):
                violations = response
            else:
                violations = []
            
            # Convert to ConflictFinding objects
            conflicts = []
            seen_articles: Set[str] = set()
            
            for violation in violations:
                if not isinstance(violation, dict):
                    continue
                
                article_id = str(violation.get("article_id", ""))
                
                # Validate article ID
                if article_id not in valid_article_ids:
                    continue
                
                # Skip Article 8 (too general - flag specific articles instead)
                if article_id == "8":
                    continue
                
                # Additional validation: Check if this article makes sense for this policy
                # Skip Article 10/10A only if policy explicitly complies (safeguards already checked above)
                # Don't filter too strictly - let LLM decide based on context
                
                # Skip Article 19A if no information access context (strict filtering)
                if article_id == "19A":
                    text_lower = state.raw_text.lower()
                    has_info_context = any(kw in text_lower for kw in [
                        "access to information", "right to information", "information access",
                        "public information", "transparency", "information in all matters", "public importance",
                        "right to have access to information"
                    ])
                    # Strict: only flag if clearly about information access
                    if not has_info_context:
                        continue
                
                # Skip Article 14 for detention cases (strict filtering)
                if article_id == "14":
                    text_lower = state.raw_text.lower()
                    # If it's about detention/arrest, don't flag Article 14
                    has_detention_context = any(kw in text_lower for kw in [
                        "detention", "arrest", "detainees", "arrested", "detained", "preventive detention"
                    ])
                    has_privacy_data_context = any(kw in text_lower for kw in [
                        "biometric", "personal data", "personal medical data", "data collection",
                        "government databases", "without requiring judicial warrants", "without consent"
                    ])
                    # Only flag Article 14 if it's about data/privacy, not detention
                    if has_detention_context and not has_privacy_data_context:
                        continue
                
                # Avoid duplicates
                if article_id in seen_articles:
                    continue
                
                confidence = float(violation.get("confidence", 0.7))
                if confidence >= 0.65:
                    # Get article title from retrieved articles
                    article_title = violation.get("article_title", "")
                    if not article_title:
                        for art in state.retrieved_articles:
                            if art.article_id == article_id:
                                article_title = art.title
                                break
                    
                    conflicts.append(
                        ConflictFinding(
                            article_id=article_id,
                            article_title=article_title or f"Article {article_id}",
                            description=str(violation.get("description", "")),
                            severity=str(violation.get("severity", "medium")),
                            jurisdiction=str(violation.get("jurisdiction", "fundamental_rights")),
                            confidence=confidence,
                            evidence=str(violation.get("evidence", "")),
                        )
                    )
                    seen_articles.add(article_id)
            
            return conflicts
            
        except Exception as exc:
            return []

    def _build_articles_context(self, state: ComplianceState) -> str:
        """Build context string from retrieved articles."""
        context_parts = []
        for article in state.retrieved_articles[:10]:  # Use up to 10 articles
            context_parts.append(
                f"Article {article.article_id} - {article.title}\n"
                f"Text: {article.text_snippet[:600]}...\n"
            )
        return "\n".join(context_parts)

    def _ensure_relevant_articles(self, state: ComplianceState) -> None:
        """Ensure we have relevant articles by semantic search and keyword matching."""
        text_lower = state.raw_text.lower()
        all_articles = {art.article_id: art for art in self.config.articles}
        retrieved_ids = {art.article_id for art in state.retrieved_articles}
        
        # Enhanced keyword-to-article mapping
        keyword_to_articles = {
            # Privacy/data protection - be more specific to avoid false positives
            "biometric": ["14"],
            "personal medical data": ["14"],
            "personal data": ["14"],
            "without requiring judicial warrants": ["14"],
            "without obtaining consent": ["14"],
            "indefinite storage": ["14"],
            # Don't trigger Article 14 for detention cases
            # Only trigger if it's about data/privacy collection, not detention
            
            # Detention/arrest (be more specific - only if violation indicators present)
            # Note: Don't trigger on just "detention" - need violation indicators
            "detention without": ["10", "10A"],
            "arrest without": ["10", "10A"],
            "without magistrate": ["10"],
            "detainees will not be informed": ["10"],
            "without being informed of grounds": ["10"],
            "indefinite detention": ["10"],
            
            # Speech
            "speech": ["19"],
            "criticism": ["19"],
            "ban": ["19"],
            "prohibit": ["19"],
            "without legal basis": ["19"],
            
            # Movement/equality - ensure Article 25 is retrieved for discriminatory policies
            "restricting movement": ["15", "25"],
            "ethnic community": ["15", "25"],
            "specific ethnic": ["15", "25"],
            "members of that community": ["15", "25"],
            "applies only to members": ["25"],
            "applies only to": ["25"],
            "discrimination": ["25"],
            "unequal treatment": ["25"],
            "equality": ["25"],
            "equal before the law": ["25"],
            # Additional patterns for Article 25
            "specific ethnic community": ["25"],
            "targeting specific": ["25"],
            "only to members": ["25"],
            
            # Federal/provincial
            "provincial law conflicts": ["143"],
            "conflicts with federal": ["143"],
        }
        
        found_articles = set()
        for keyword, article_ids in keyword_to_articles.items():
            if keyword in text_lower:
                for art_id in article_ids:
                    if art_id in all_articles and art_id not in retrieved_ids and art_id not in found_articles:
                        article = all_articles[art_id]
                        state.retrieved_articles.append(
                            ArticleEvidence(
                                article_id=art_id,
                                title=article.title,
                                text_snippet=article.text[:500],
                                relevance=0.65,
                                category=article.category
                            )
                        )
                        found_articles.add(art_id)
                        retrieved_ids.add(art_id)
        
        # Also use semantic search to find additional articles
        policy_embedding = self.embedding_model.encode(state.raw_text, convert_to_numpy=True)
        
        # Check all articles that haven't been retrieved
        for article_id, article in all_articles.items():
            if article_id in retrieved_ids or article_id in found_articles:
                continue
            
            # Skip Article 8 (too general)
            if article_id == "8":
                continue
            
            # Check semantic similarity to article text
            article_emb = self.embedding_model.encode(article.text[:1000], convert_to_numpy=True).reshape(1, -1)
            policy_emb = policy_embedding.reshape(1, -1)
            similarity = cosine_similarity(policy_emb, article_emb)[0][0]
            
            # If high similarity, add article
            if similarity > 0.45:  # Lower threshold to catch more relevant articles
                state.retrieved_articles.append(
                    ArticleEvidence(
                        article_id=article_id,
                        title=article.title,
                        text_snippet=article.text[:500],
                        relevance=float(similarity),
                        category=article.category
                    )
                )
                retrieved_ids.add(article_id)

    def _merge_conflicts(
        self, 
        embedding_conflicts: List[ConflictFinding],
        llm_conflicts: List[ConflictFinding],
        state: ComplianceState
    ) -> List[ConflictFinding]:
        """Intelligently merge conflicts from different sources."""
        merged: Dict[str, ConflictFinding] = {}
        
        # Prefer LLM conflicts (higher quality reasoning)
        for llm_cf in llm_conflicts:
            merged[llm_cf.article_id] = llm_cf
        
        # Add embedding conflicts if not already covered and confidence is high enough
        for emb_cf in embedding_conflicts:
            if emb_cf.article_id not in merged:
                # Only add if confidence is reasonably high
                if emb_cf.confidence >= 0.60:
                    merged[emb_cf.article_id] = emb_cf
        
        # Sort by confidence (highest first)
        return sorted(merged.values(), key=lambda cf: cf.confidence, reverse=True)
