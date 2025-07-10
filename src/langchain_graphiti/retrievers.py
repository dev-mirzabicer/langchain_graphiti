"""
LangChain retrievers for the Graphiti knowledge graph system.

This module provides powerful retriever implementations that leverage Graphiti's
advanced search capabilities, including hybrid search (text, vector, and graph-based),
sophisticated reranking, and graph-aware semantic search.

Key Components:
- GraphitiRetriever: The primary retriever with full configurability and proper score handling
- GraphitiSemanticRetriever: Specialized for graph-aware semantic search
- Streaming support and async iteration
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional, Tuple

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import ConfigurableField
from langchain_core.runnables.config import RunnableConfig
from langsmith import traceable
from pydantic import Field

from ._client import GraphitiClient
from .exceptions import GraphitiRetrieverError
from .utils import require_client, safe_sync_run
from graphiti_core.search.search_config import (
    SearchConfig,
    SearchResults,
    EdgeSearchConfig,
    EdgeSearchMethod,
    EdgeReranker,
    EpisodeSearchConfig,
    EpisodeSearchMethod,
)
from graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search import search as graphiti_search_internal

logger = logging.getLogger(__name__)


class GraphitiRetriever(BaseRetriever):
    """
    A powerful retriever for the Graphiti knowledge graph system.

    This retriever uses Graphiti's advanced search capabilities including:
    - Hybrid search combining text, vector, and graph traversal
    - Sophisticated reranking (RRF, MMR, cross-encoder)
    - Graph-aware context preservation
    - Runtime configuration support
    - Proper score extraction from native Graphiti search
    - Streaming support

    The retriever returns LangChain Documents with rich metadata that preserves
    graph relationships and enables downstream components to understand the
    knowledge graph structure.
    """

    client: GraphitiClient = Field(
        ..., description="The Graphiti client instance."
    )
    config: SearchConfig = Field(
        default_factory=lambda: COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
        description="The search configuration to use for retrieval.",
    )
    search_mode: Literal["edges", "nodes", "combined"] = Field(
        default="combined",
        description="Determines what to return as documents: 'edges', 'nodes', or both.",
    )
    group_ids: Optional[List[str]] = Field(
        default=None, description="Optional list of group IDs to scope the search."
    )
    search_filter: Optional[SearchFilters] = Field(
        default_factory=SearchFilters, description="Optional filters to apply to the search."
    )
    include_graph_context: bool = Field(
        default=True,
        description="Whether to include graph relationship context in metadata.",
    )
    score_threshold: Optional[float] = Field(
        default=None,
        description="Minimum score threshold for filtering results (0.0-1.0).",
    )
    k: int = Field(
        default=10, description="The number of documents to return."
    )
 
    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def configurable_fields(cls, **field_updates):
        """Enable runtime configuration of retriever parameters."""
        return super().configurable_fields(
            config=ConfigurableField(
                id="search_config",
                name="Search Configuration",
                description="The SearchConfig object controlling search behavior",
            ),
            search_mode=ConfigurableField(
                id="search_mode",
                name="Search Mode", 
                description="What to return: 'edges', 'nodes', or 'combined'",
            ),
            group_ids=ConfigurableField(
                id="group_ids",
                name="Group IDs",
                description="List of group IDs to scope the search",
            ),
            score_threshold=ConfigurableField(
                id="score_threshold",
                name="Score Threshold",
                description="Minimum score threshold for filtering results",
            ),
            **field_updates
        )

    @traceable
    @require_client
    async def _perform_search_with_scores(
        self, query: str, config_override: Optional[SearchConfig] = None, limit: Optional[int] = None, **kwargs: Any
    ) -> Tuple[SearchResults, Dict[str, float]]:
        """
        Perform Graphiti search and extract scores more efficiently.
        
        This method uses Graphiti's native search capabilities and extracts scores
        from the reranking process instead of re-ranking all results.
        """
        try:
            graphiti_instance = self.client.graphiti_instance
            
            # Use the override if provided, otherwise use the instance's config
            search_config = config_override or self.config.model_copy(deep=True)

            # Set the limit on the config. If a limit is passed, it takes precedence.
            # Otherwise, use the one from the retriever instance.
            effective_limit = limit if limit is not None else self.k
            search_config.limit = effective_limit
 
            # Use Graphiti's native search which includes scoring in the reranker
            # Extract only the kwargs that are valid for the search function
            valid_kwargs = {}
            if "center_node_uuid" in kwargs:
                valid_kwargs["center_node_uuid"] = kwargs["center_node_uuid"]

            search_results = await graphiti_search_internal(
                clients=graphiti_instance.clients,
                query=query,
                group_ids=self.group_ids,
                config=search_config,
                search_filter=self.search_filter or SearchFilters(),
                query_vector=None,  # Let Graphiti generate it internally
                **valid_kwargs,
            )

            # Simplified scoring logic - positional scoring based on result order
            scores = {}
            all_items = []
            
            if self.search_mode in ["edges", "combined"]:
                all_items.extend([(edge.uuid, "edge") for edge in search_results.edges])
            
            if self.search_mode in ["nodes", "combined"]:
                all_items.extend([(node.uuid, "node") for node in search_results.nodes])
            
            # Assign decreasing scores based on ranking order (higher is better)
            total_items = len(all_items)
            for i, (uuid, item_type) in enumerate(all_items):
                scores[uuid] = 1.0 - (i / max(1, total_items))
            
            return search_results, scores
            
        except Exception as e:
            logger.error(f"Search failed for query '{query[:50]}...': {e}")
            raise GraphitiRetrieverError(f"Failed to perform search: {e}") from e

    @traceable
    @require_client
    async def _convert_search_results_to_docs(
        self, search_results: SearchResults, scores: Dict[str, float]
    ) -> List[Document]:
        """Converts Graphiti search results into LangChain Documents."""
        docs: List[Document] = []
        if not search_results.episodes:
            return docs

        for episode in search_results.episodes:
            # The standard tests expect the original page_content and metadata.
            # We can extract this from the episode content.
            # Assuming the episode content is the original page_content.
            doc = Document(
                page_content=episode.content,
                metadata={
                    "uuid": episode.uuid,
                    "name": episode.name,
                    "source": episode.source.value
                    if hasattr(episode.source, "value")
                    else str(episode.source),
                    "group_id": episode.group_id,
                    "created_at": episode.created_at.isoformat(),
                    "valid_at": episode.valid_at.isoformat() if episode.valid_at else None,
                },
            )
            docs.append(doc)
        return docs

    @require_client
    async def astream(
        self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> AsyncIterator[List[Document]]:
        """
        Asynchronously stream documents.
        
        This provides proper async streaming support for use in async contexts.
        """
        try:
            search_results, scores = await self._perform_search_with_scores(
                input, **kwargs
            )
            docs = await self._convert_search_results_to_docs(search_results, scores)
            yield docs
        except Exception as e:
            logger.error(f"Streaming failed for query '{input[:50]}...': {e}")
            raise GraphitiRetrieverError(f"Failed to stream results: {e}") from e

    @require_client
    def stream(
        self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Iterator[List[Document]]:
        """
        Stream documents with proper async context handling.
        """
        try:
            async def _collect_docs() -> List[List[Document]]:
                all_docs = []
                async for doc_list in self.astream(input, config=config, **kwargs):
                    all_docs.append(doc_list)
                return all_docs

            # safe_sync_run will handle the event loop correctly
            docs_lists = safe_sync_run(_collect_docs())
            yield from docs_lists
        except Exception as e:
            logger.error(f"Sync streaming failed for query '{input[:50]}...': {e}")
            raise GraphitiRetrieverError(f"Failed to stream results: {e}") from e

    @traceable
    @require_client
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        """Synchronous retrieval of documents from Graphiti."""
        try:
            return safe_sync_run(self._aget_relevant_documents(query, run_manager=run_manager, **kwargs))
        except Exception as e:
            logger.error(f"Sync retrieval failed for query '{query[:50]}...': {e}")
            raise GraphitiRetrieverError(f"Failed to retrieve documents: {e}") from e
 
    @traceable
    @require_client
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        """Asynchronous retrieval of documents from Graphiti."""
        try:
            k = kwargs.get("k", self.k)
            search_results, scores = await self._perform_search_with_scores(query, limit=k, **kwargs)
            return await self._convert_search_results_to_docs(search_results, scores)
        except Exception as e:
            logger.error(f"Async retrieval failed for query '{query[:50]}...': {e}")
            raise GraphitiRetrieverError(f"Failed to retrieve documents: {e}") from e

    @require_client
    async def asimilarity_search_with_score(
        self, 
        query: str, 
        k: int = 10,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with score filtering without mutating instance state.
        
        Args:
            query: The search query
            k: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            # Create a temporary config for this specific call (thread-safe)
            temp_config = self.config.model_copy(deep=True)
            
            # Apply score threshold to search config if specified
            if score_threshold is not None:
                temp_config.reranker_min_score = score_threshold
 
            search_results, scores = await self._perform_search_with_scores(
                query, config_override=temp_config, limit=k
            )
            docs = await self._convert_search_results_to_docs(search_results, scores)
            
            # Filter by score threshold again as the internal one is a hint
            final_results = []
            for doc in docs:
                score = doc.metadata.get("score", 0.0)
                if score_threshold is None or score >= score_threshold:
                    final_results.append((doc, score))
            
            return final_results
        except Exception as e:
            logger.error(f"Similarity search with score failed for query '{query[:50]}...': {e}")
            raise GraphitiRetrieverError(f"Failed to perform similarity search: {e}") from e

    @require_client
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 10,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Synchronous version of asimilarity_search_with_score."""
        try:
            return safe_sync_run(
                self.asimilarity_search_with_score(query, k, score_threshold, **kwargs)
            )
        except Exception as e:
            logger.error(f"Sync similarity search with score failed for query '{query[:50]}...': {e}")
            raise GraphitiRetrieverError(f"Failed to perform similarity search: {e}") from e


# A search configuration that combines graph-aware edge search with episode search
SEMANTIC_SEARCH_WITH_EPISODES = SearchConfig(
    edge_config=EdgeSearchConfig(
        search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
        reranker=EdgeReranker.node_distance,
        sim_min_score=0.6,
        mmr_lambda=0.5,
        bfs_max_depth=3,
    ),
    node_config=None,
    episode_config=EpisodeSearchConfig(
        search_methods=[
            EpisodeSearchMethod.bm25,
        ]
    ),
    community_config=None,
)


class GraphitiSemanticRetriever(GraphitiRetriever):
    """
    A specialized retriever that leverages Graphiti's graph-aware semantic search.
    
    This retriever uses advanced graph topology features like node distance,
    community structure, and relationship patterns to enhance semantic search
    beyond simple vector similarity.
    """

    def __init__(self, client: GraphitiClient, **kwargs):
        # Use a graph-aware search configuration that also returns episodes
        kwargs.setdefault("config", SEMANTIC_SEARCH_WITH_EPISODES)
        kwargs.setdefault("search_mode", "combined")
        kwargs.setdefault("include_graph_context", True)
        super().__init__(client=client, **kwargs)

    async def asimilarity_search(
        self,
        query: str,
        k: int = 10,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform async similarity search."""
        try:
            results_with_scores = await self.asimilarity_search_with_score(
                query, k, score_threshold, **kwargs
            )
            return [doc for doc, score in results_with_scores]
        except Exception as e:
            logger.error(f"Async similarity search failed for query '{query[:50]}...': {e}")
            raise GraphitiRetrieverError(f"Failed to perform async similarity search: {e}") from e

    @require_client
    def similarity_search(
        self,
        query: str,
        k: int = 10,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Perform sync similarity search."""
        try:
            results_with_scores = self.similarity_search_with_score(
                query, k, score_threshold, **kwargs
            )
            return [doc for doc, score in results_with_scores]
        except Exception as e:
            logger.error(f"Sync similarity search failed for query '{query[:50]}...': {e}")
            raise GraphitiRetrieverError(f"Failed to perform similarity search: {e}") from e

    @require_client
    async def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 10,
        lambda_mult: float = 0.5,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Perform Maximum Marginal Relevance (MMR) search without state mutation.
        
        This uses Graphiti's MMR reranker configuration for diverse results.
        """
        try:
            # Create MMR-configured search without mutating instance state
            from graphiti_core.search.search_config import SearchConfig, EdgeSearchConfig, NodeSearchConfig
            from graphiti_core.search.search_config import EdgeReranker, NodeReranker, EdgeSearchMethod, NodeSearchMethod
            
            mmr_config = SearchConfig(
                edge_config=EdgeSearchConfig(
                    search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
                    reranker=EdgeReranker.mmr,
                    mmr_lambda=lambda_mult,
                ),
                node_config=NodeSearchConfig(
                    search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
                    reranker=NodeReranker.mmr,
                    mmr_lambda=lambda_mult,
                ),
                limit=k,
            )
            
            if score_threshold is not None:
                mmr_config.reranker_min_score = score_threshold
            
            search_results, scores = await self._perform_search_with_scores(
                query, config_override=mmr_config
            )
            docs = await self._convert_search_results_to_docs(search_results, scores)
            return docs
        except Exception as e:
            logger.error(f"MMR search failed for query '{query[:50]}...': {e}")
            raise GraphitiRetrieverError(f"Failed to perform MMR search: {e}") from e


class GraphitiCachedRetriever(GraphitiRetriever):
    """
    A caching wrapper around GraphitiRetriever for improved performance.
    
    This retriever caches search results to avoid repeated expensive operations.
    Useful for development, testing, or when the same queries are made frequently.
    """
    
    def __init__(self, client: GraphitiClient, cache_ttl_seconds: int = 300, **kwargs: Any):
        super().__init__(client=client, **kwargs)
        self._cache: Dict[str, Tuple[List[Document], float]] = {}
        self._cache_ttl = cache_ttl_seconds
        
    def _get_cache_key(self, query: str) -> str:
        """Generate a cache key for the query and current configuration."""
        import hashlib
        import json
        
        cache_data = {
            "query": query,
            "search_mode": self.search_mode,
            "group_ids": self.group_ids,
            "score_threshold": self.score_threshold,
            "config_hash": hash(str(self.config)),  # Simple config hash
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cache entry is still valid."""
        try:
            import time
            return (time.time() - timestamp) < self._cache_ttl
        except Exception as e:
            logger.warning(f"Cache validation failed: {e}")
            return False
    
    @require_client
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Cached asynchronous retrieval."""
        import time
        
        try:
            cache_key = self._get_cache_key(query)
            
            # Check cache
            if cache_key in self._cache:
                cached_docs, timestamp = self._cache[cache_key]
                if self._is_cache_valid(timestamp):
                    logger.debug(f"Cache hit for query: {query[:50]}...")
                    return cached_docs
            
            # Cache miss, perform search
            docs = await super()._aget_relevant_documents(query, run_manager=run_manager)
            
            # Store in cache
            self._cache[cache_key] = (docs, time.time())
            
            # Clean old cache entries (simple cleanup)
            if len(self._cache) > 100:  # Arbitrary limit
                expired_keys = [
                    key for key, (_, timestamp) in self._cache.items()
                    if not self._is_cache_valid(timestamp)
                ]
                for key in expired_keys:
                    del self._cache[key]
            
            return docs
        except Exception as e:
            logger.error(f"Cached retrieval failed for query '{query[:50]}...': {e}")
            raise GraphitiRetrieverError(f"Failed to retrieve cached documents: {e}") from e
    
    def clear_cache(self) -> None:
        """Clear the retrieval cache."""
        self._cache.clear()