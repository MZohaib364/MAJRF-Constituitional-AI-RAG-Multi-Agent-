from __future__ import annotations

import re
from typing import List

SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").strip()
    return re.sub(r"\s+", " ", text)


def segment_sentences(text: str, max_length: int = 600) -> List[str]:
    normalized = normalize_text(text)
    sentences = SENTENCE_SPLIT_REGEX.split(normalized)
    segments: List[str] = []
    buffer: List[str] = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if current_length + len(sentence) > max_length and buffer:
            segments.append(" ".join(buffer))
            buffer = [sentence]
            current_length = len(sentence)
        else:
            buffer.append(sentence)
            current_length += len(sentence)

    if buffer:
        segments.append(" ".join(buffer))
    return segments

