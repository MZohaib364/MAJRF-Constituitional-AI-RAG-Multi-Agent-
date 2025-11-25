from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import chromadb
from chromadb.utils import embedding_functions

from .config import Article
from .data_models import ArticleEvidence

COLLECTION_NAME = "constitution_articles"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class ConstitutionKnowledgeBase:
    """Chroma-based semantic search over constitutional articles."""

    def __init__(self, articles: Iterable[Article], persist_dir: str | Path = "vector_store/constitution"):
        self.articles: List[Article] = list(articles)
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self._create_collection()
        self._populate_collection()

    def _create_collection(self):
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        return self.client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fn,
        )

    def _populate_collection(self) -> None:
        if not self.articles:
            return
        ids = [article.article_id for article in self.articles]
        documents = [article.text for article in self.articles]
        metadatas = [{"title": article.title, "category": article.category} for article in self.articles]
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def search(self, query: str, top_k: int = 5) -> List[ArticleEvidence]:
        if not query.strip():
            return []
        results = self.collection.query(query_texts=[query], n_results=top_k)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]

        evidences: List[ArticleEvidence] = []
        for idx, doc in enumerate(documents):
            if not doc:
                continue
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            article_id = ids[idx] if idx < len(ids) else f"article_{idx}"
            distance = distances[idx] if idx < len(distances) else 1.0
            relevance = max(0.0, 1.0 - float(distance))
            snippet = doc[:400]
            evidences.append(
                ArticleEvidence(
                    article_id=str(article_id),
                    title=str(metadata.get("title", "")),
                    text_snippet=snippet,
                    relevance=relevance,
                    category=str(metadata.get("category", "")),
                )
            )
        return evidences


def build_knowledge_base(
    articles: Iterable[Article],
    persist_dir: str | Path = "vector_store/constitution",
) -> ConstitutionKnowledgeBase:
    return ConstitutionKnowledgeBase(articles, persist_dir=persist_dir)

