from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple

from ..config import Rulebook
from ..data_models import ComplianceState
from .base import Agent


class DomainClassifierAgent(Agent):
    """Embedding-based domain classification using semantic similarity."""
    
    name = "DomainClassifierAgent"

    def __init__(self, rulebook: Rulebook, threshold: float = 0.3, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.rulebook = rulebook
        self.threshold = threshold
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.domain_embeddings = self._embed_domains()

    def _embed_domains(self) -> Dict[str, np.ndarray]:
        """Create embeddings for each domain using comprehensive domain descriptions."""
        domain_descriptions = {
            "cyber_security": (
                "cybersecurity, network security, information systems, national cyber defense, "
                "CERT, cyber attacks, digital infrastructure, internet security, malware, "
                "information security, network protection, cyber threats"
            ),
            "data_protection": (
                "personal data, privacy, data processing, consent, PII, data retention, "
                "digital rights, information privacy, biometric data, personal information, "
                "data collection, surveillance, data sharing"
            ),
            "healthcare": (
                "health, medical, hospital, primary care, clinical service, health authority, "
                "public health, medical licensing, patient care, healthcare services, "
                "medical facilities, health regulations"
            ),
            "land_use": (
                "land, plot, CDA, ground rent, building control, zoning, urban development, "
                "property, real estate, construction, building permits, land regulations"
            ),
        }
        
        embeddings = {}
        for domain, description in domain_descriptions.items():
            embeddings[domain] = self.embedding_model.encode(description, convert_to_numpy=True)
        return embeddings

    def run(self, state: ComplianceState) -> ComplianceState:
        """Classify policy domain using semantic similarity."""
        # Embed policy text
        policy_embedding = self.embedding_model.encode(state.raw_text, convert_to_numpy=True)
        policy_embedding = policy_embedding.reshape(1, -1)
        
        # Compute cosine similarity with each domain
        scores: Dict[str, float] = {}
        for domain, domain_emb in self.domain_embeddings.items():
            domain_emb = domain_emb.reshape(1, -1)
            similarity = cosine_similarity(policy_embedding, domain_emb)[0][0]
            scores[domain] = float(similarity)
        
        # Sort by score and filter by threshold
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        detected = [domain for domain, score in sorted_scores if score >= self.threshold]
        state.detected_domains = detected or ["general"]
        
        # Store scores for each segment
        for segment in state.segments:
            segment.domain_scores = scores
        
        return state
