from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pdfplumber

from ..data_models import ComplianceState, PolicySegment
from ..utils.text_utils import normalize_text, segment_sentences
from .base import Agent


class IngestionAgent(Agent):
    name = "IngestionAgent"

    def __init__(self, max_segment_length: int = 600):
        self.max_segment_length = max_segment_length

    def run(self, state: ComplianceState) -> ComplianceState:
        text = self._read_source(state.source)
        state.raw_text = text
        state.segments = [
            PolicySegment(text=segment) for segment in segment_sentences(text, self.max_segment_length)
        ]
        return state

    def _read_source(self, source: str) -> str:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Policy file not found: {source}")
        if path.suffix.lower() == ".pdf":
            return self._read_pdf(path)
        return normalize_text(path.read_text(encoding="utf-8"))

    def _read_pdf(self, path: Path) -> str:
        text_chunks: Iterable[str] = []
        with pdfplumber.open(path) as pdf:
            text_chunks = [page.extract_text() or "" for page in pdf.pages]
        return normalize_text(" ".join(text_chunks))

