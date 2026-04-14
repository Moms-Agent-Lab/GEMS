"""
SkillRAG — Embedding-based skill retrieval for ComfyClaw.

Replaces the keyword-based ``detect_relevant_skills`` with semantic
similarity over skill descriptions. Falls back to keyword matching
when no embedding model is available.

Architecture:
- Encodes all skill descriptions into a vector store at init time.
- On query, encodes the prompt + verifier feedback and retrieves the
  top-k most semantically similar skills.
- Optionally combines with verifier fix_strategy keywords for boosted
  precision.

Embedding backends (in priority order):
1. litellm embedding API (remote, high quality)
2. sentence-transformers (local, fast)
3. Keyword fallback (no model needed)

Usage::

    rag = SkillRAG(skill_manager=sm, backend="litellm")
    relevant = rag.retrieve("a cat wearing armor", top_k=3)
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from .skill_manager import SkillManager

log = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SkillRAG:
    """Embedding-based skill retrieval system.

    Parameters
    ----------
    skill_manager : The SkillManager instance to read skills from.
    backend       : "litellm", "sentence-transformers", or "keyword".
    embedding_model : Model name for the embedding backend.
    api_key       : API key for remote embedding models.
    cache_embeddings : Whether to cache embeddings to disk.
    """

    def __init__(
        self,
        skill_manager: SkillManager,
        backend: str = "auto",
        embedding_model: str = "text-embedding-3-small",
        api_key: str = "",
        cache_embeddings: bool = True,
    ) -> None:
        self.skill_manager = skill_manager
        self.embedding_model = embedding_model
        self.api_key = api_key
        self.cache_embeddings = cache_embeddings

        self._backend = self._resolve_backend(backend)
        self._skill_embeddings: dict[str, list[float]] = {}
        self._build_index()

    def _resolve_backend(self, backend: str) -> str:
        """Determine the best available embedding backend."""
        if backend != "auto":
            return backend

        try:
            import litellm  # noqa: F401
            return "litellm"
        except ImportError:
            pass

        try:
            import sentence_transformers  # noqa: F401
            return "sentence-transformers"
        except ImportError:
            pass

        log.info("No embedding model available, falling back to keyword matching")
        return "keyword"

    def _build_index(self) -> None:
        """Pre-compute embeddings for all skill descriptions."""
        if self._backend == "keyword":
            return

        skill_texts: dict[str, str] = {}
        for name in self.skill_manager.skill_names:
            props = self.skill_manager.get_properties(name)
            skill_texts[name] = f"{props.name}: {props.description}"

        if not skill_texts:
            return

        try:
            embeddings = self._batch_embed(list(skill_texts.values()))
            for (name, _), emb in zip(skill_texts.items(), embeddings):
                self._skill_embeddings[name] = emb
            log.info(
                "Built skill RAG index with %d skills (backend=%s)",
                len(self._skill_embeddings), self._backend,
            )
        except Exception as exc:
            log.warning("Failed to build embedding index: %s. Falling back to keyword.", exc)
            self._backend = "keyword"

    def _batch_embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using the configured backend."""
        if self._backend == "litellm":
            return self._embed_litellm(texts)
        elif self._backend == "sentence-transformers":
            return self._embed_sentence_transformers(texts)
        else:
            raise ValueError(f"Unknown embedding backend: {self._backend}")

    def _embed_litellm(self, texts: list[str]) -> list[list[float]]:
        import litellm
        response = litellm.embedding(
            model=self.embedding_model,
            input=texts,
        )
        return [item["embedding"] for item in response.data]

    def _embed_sentence_transformers(self, texts: list[str]) -> list[list[float]]:
        from sentence_transformers import SentenceTransformer
        if not hasattr(self, "_st_model"):
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = self._st_model.encode(texts)
        return [emb.tolist() for emb in embeddings]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        verifier_feedback: str | None = None,
        min_similarity: float = 0.1,
    ) -> list[dict[str, Any]]:
        """Retrieve the most relevant skills for a query.

        Parameters
        ----------
        query : The user's image generation prompt.
        top_k : Maximum number of skills to return.
        verifier_feedback : Optional verifier feedback to augment the query.
        min_similarity : Minimum cosine similarity to include a result.

        Returns
        -------
        List of dicts with keys: name, description, similarity, source.
        """
        if self._backend == "keyword":
            return self._keyword_retrieve(query, verifier_feedback, top_k)

        # Build augmented query from prompt + feedback
        augmented = query
        if verifier_feedback:
            strategies = re.findall(
                r"fix_strategy['\"]?\s*:\s*['\"]?(\w+)", verifier_feedback
            )
            if strategies:
                augmented += " " + " ".join(strategies)
            # Also include keywords from feedback
            augmented += " " + verifier_feedback[:200]

        try:
            query_emb = self._batch_embed([augmented])[0]
        except Exception as exc:
            log.warning("Embedding query failed: %s. Using keyword fallback.", exc)
            return self._keyword_retrieve(query, verifier_feedback, top_k)

        results: list[tuple[float, str]] = []
        for name, skill_emb in self._skill_embeddings.items():
            sim = _cosine_similarity(query_emb, skill_emb)
            if sim >= min_similarity:
                results.append((sim, name))

        results.sort(key=lambda x: x[0], reverse=True)

        output: list[dict[str, Any]] = []
        for sim, name in results[:top_k]:
            props = self.skill_manager.get_properties(name)
            output.append({
                "name": name,
                "description": props.description,
                "similarity": round(sim, 4),
                "source": "embedding",
            })

        return output

    def _keyword_retrieve(
        self,
        query: str,
        verifier_feedback: str | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Fallback keyword-based retrieval."""
        combined = query
        if verifier_feedback:
            combined += " " + verifier_feedback

        matched = self.skill_manager.detect_relevant_skills(combined)
        output: list[dict[str, Any]] = []
        for name in matched[:top_k]:
            props = self.skill_manager.get_properties(name)
            output.append({
                "name": name,
                "description": props.description,
                "similarity": 1.0,
                "source": "keyword",
            })
        return output

    def suggest_skills_xml(
        self,
        query: str,
        verifier_feedback: str | None = None,
        top_k: int = 3,
    ) -> str:
        """Generate an XML block of suggested skills for the agent prompt.

        Returns an ``<suggested_skills>`` XML block that can be injected
        into the agent's system or user message.
        """
        skills = self.retrieve(query, top_k=top_k, verifier_feedback=verifier_feedback)
        if not skills:
            return ""

        lines = ["<suggested_skills>"]
        for s in skills:
            lines.extend([
                f"<skill name=\"{s['name']}\" similarity=\"{s['similarity']}\">",
                f"  {s['description']}",
                "</skill>",
            ])
        lines.append("</suggested_skills>")
        return "\n".join(lines)
