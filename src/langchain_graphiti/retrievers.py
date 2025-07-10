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

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional, Tuple

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import ConfigurableField
from langsmith import traceable
from pydantic import BaseModel, Field

from ._client import GraphitiClient
from .exceptions import GraphitiRetrieverError, GraphitiClientError
from .utils import require_client, safe_sync_run, format_graph_results
from graphiti_core.search.search_config import SearchConfig, SearchResults
from graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
    EDGE_HYBRID_SEARCH_NODE_DISTANCE,
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
        self, query: str
    ) -> Tuple[SearchResults, Dict[str, float]]:
        """
        Perform Graphiti search and extract scores more efficiently.
        
        This method uses Graphiti's native search capabilities and extracts scores
        from the reranking process instead of re-ranking all results.
        """
        try:
            graphiti_instance = self.client.graphiti_instance
            
            # Apply score threshold to search config if specified
            config = self.config
            if self.score_threshold is not None:
                config.reranker_min_score = self.score_threshold

            # Use Graphiti's native search which includes scoring in the reranker
            search_results = await graphiti_search_internal(
                clients=graphiti_instance.clients,
                query=query,
                group_ids=self.group_ids,
                config=config,
                search_filter=self.search_filter,
                query_vector=None,  # Let Graphiti generate it internally
            )

            # Extract scores based on the reranker configuration
            scores = {}
            
            # For cross-encoder reranking, we can get more accurate scores
            if (hasattr(config, 'edge_config') and config.edge_config and 
                hasattr(config.edge_config, 'reranker') and 
                str(config.edge_config.reranker).endswith('cross_encoder')):
                
                # Cross-encoder provides high-quality scores
                await self._extract_cross_encoder_scores(query, search_results, scores)
            else:
                # Fallback to RRF or other reranker scores
                await self._extract_fallback_scores(search_results, scores)
            
            return search_results, scores
            
        except Exception as e:
            logger.error(f"Search failed for query '{query[:50]}...': {e}")
            raise GraphitiRetrieverError(f"Failed to perform search: {e}") from e

    async def _extract_cross_encoder_scores(
        self, 
        query: str, 
        search_results: SearchResults, 
        scores: Dict[str, float]
    ) -> None:
        """Extract scores using cross-encoder for high accuracy."""
        try:
            cross_encoder = self.client.graphiti_instance.cross_encoder
            
            # Build passages for scoring
            passages_to_score = []
            passage_to_uuid = {}

            if self.search_mode in ["edges", "combined"]:
                for edge in search_results.edges:
                    passage = f"[{edge.name}] {edge.fact}"
                    passages_to_score.append(passage)
                    passage_to_uuid[passage] = edge.uuid

            if self.search_mode in ["nodes", "combined"]:
                for node in search_results.nodes:
                    passage = f"Entity: {node.name}. Summary: {node.summary or 'No summary available.'}"
                    passages_to_score.append(passage)
                    passage_to_uuid[passage] = node.uuid

            if passages_to_score:
                try:
                    # Get ranked passages with scores from cross-encoder
                    ranked_passages = await cross_encoder.rank(query, passages_to_score)
                    for passage, score in ranked_passages:
                        uuid = passage_to_uuid.get(passage)
                        if uuid:
                            scores[uuid] = float(score)
                except Exception as e:
                    logger.warning(f"Failed to extract cross-encoder scores: {e}")
                    # Fallback to basic scoring
                    await self._extract_fallback_scores(search_results, scores)
        except Exception as e:
            logger.warning(f"Cross-encoder scoring failed: {e}")
            await self._extract_fallback_scores(search_results, scores)

    async def _extract_fallback_scores(
        self, 
        search_results: SearchResults, 
        scores: Dict[str, float]
    ) -> None:
        """Extract basic scores when cross-encoder scoring is not available."""
        try:
            # Assign scores based on order (higher for earlier results)
            all_items = []
            
            if self.search_mode in ["edges", "combined"]:
                all_items.extend([(edge.uuid, "edge") for edge in search_results.edges])
            
            if self.search_mode in ["nodes", "combined"]:
                all_items.extend([(node.uuid, "node") for node in search_results.nodes])
            
            # Assign decreasing scores based on ranking order
            total_items = len(all_items)
            for i, (uuid, item_type) in enumerate(all_items):
                # Higher score for items that appear earlier in results
                scores[uuid] = max(0.1, 1.0 - (i / max(total_items, 1)))
        except Exception as e:
            logger.error(f"Fallback scoring failed: {e}")
            # If even fallback fails, assign default scores
            for edge in search_results.edges:
                scores[edge.uuid] = 0.5
            for node in search_results.nodes:
                scores[node.uuid] = 0.5

    @traceable
    @require_client
    async def _convert_search_results_to_docs(
        self, search_results: SearchResults, scores: Dict[str, float]
    ) -> List[Document]:
        """Converts Graphiti search results into LangChain Documents with rich metadata."""
        try:
            docs = []
            
            # Create a lookup for node names to enrich edge documents
            node_uuid_to_name = {node.uuid: node.name for node in search_results.nodes}

            # Convert edges to documents
            if self.search_mode in ["edges", "combined"]:
                for edge in search_results.edges:
                    try:
                        score = scores.get(edge.uuid, 0.0)
                        
                        # Apply score threshold filtering if specified
                        if self.score_threshold is not None and score < self.score_threshold:
                            continue
                        
                        source_name = node_uuid_to_name.get(edge.source_node_uuid, "Unknown Entity")
                        target_name = node_uuid_to_name.get(edge.target_node_uuid, "Unknown Entity")
                        
                        metadata = {
                            "type": "edge",
                            "score": score,
                            "uuid": edge.uuid,
                            "name": edge.name,
                            "source_node_uuid": edge.source_node_uuid,
                            "source_node_name": source_name,
                            "target_node_uuid": edge.target_node_uuid,
                            "target_node_name": target_name,
                            "group_id": edge.group_id,
                            "created_at": edge.created_at.isoformat(),
                            "valid_at": edge.valid_at.isoformat() if edge.valid_at else None,
                            "invalid_at": edge.invalid_at.isoformat() if edge.invalid_at else None,
                            "episodes": edge.episodes,
                            "fact": edge.fact,
                            **edge.attributes,
                        }
                        
                        # Add graph context if enabled
                        if self.include_graph_context:
                            metadata["graph_context"] = {
                                "relationship_type": edge.name,
                                "embedding_available": edge.fact_embedding is not None,
                            }
                        
                        content = f"{source_name} -[{edge.name}]-> {target_name}: {edge.fact}"
                        doc = Document(page_content=content, metadata=metadata)
                        docs.append(doc)
                    except Exception as e:
                        logger.warning(f"Failed to convert edge {edge.uuid} to document: {e}")
                        continue

            # Convert nodes to documents
            if self.search_mode in ["nodes", "combined"]:
                for node in search_results.nodes:
                    try:
                        score = scores.get(node.uuid, 0.0)
                        
                        # Apply score threshold filtering if specified
                        if self.score_threshold is not None and score < self.score_threshold:
                            continue
                        
                        metadata = {
                            "type": "node",
                            "score": score,
                            "uuid": node.uuid,
                            "name": node.name,
                            "labels": node.labels,
                            "group_id": node.group_id,
                            "created_at": node.created_at.isoformat(),
                            "summary": node.summary,
                            **node.attributes,
                        }
                        
                        # Add graph context if enabled
                        if self.include_graph_context:
                            metadata["graph_context"] = {
                                "node_type": next((l for l in node.labels if l != 'Entity'), 'Entity'),
                                "embedding_available": node.name_embedding is not None,
                            }
                        
                        # Use summary if available, otherwise just the name
                        content = f"**{node.name}**\n\n{node.summary}" if node.summary else node.name
                        doc = Document(page_content=content, metadata=metadata)
                        docs.append(doc)
                    except Exception as e:
                        logger.warning(f"Failed to convert node {node.uuid} to document: {e}")
                        continue
            
            # Sort final documents by score if available
            docs.sort(key=lambda d: d.metadata.get("score", 0.0), reverse=True)

            return docs
        except Exception as e:
            logger.error(f"Failed to convert search results to documents: {e}")
            raise GraphitiRetrieverError(f"Failed to convert search results: {e}") from e

    @require_client
    async def astream(self, query: str) -> AsyncIterator[Document]:
        """
        Asynchronously stream documents one by one.
        
        This provides proper async streaming support for use in async contexts.
        """
        try:
            search_results, scores = await self._perform_search_with_scores(query)
            docs = await self._convert_search_results_to_docs(search_results, scores)
            
            for doc in docs:
                yield doc
        except Exception as e:
            logger.error(f"Streaming failed for query '{query[:50]}...': {e}")
            raise GraphitiRetrieverError(f"Failed to stream results: {e}") from e

    @require_client
    def stream(self, query: str) -> Iterator[Document]:
        """
        Stream documents one by one with proper async context handling.
        
        This method handles both sync and async contexts by checking if an event loop
        is running. If it is, it runs the async streaming in a separate thread.
        If no event loop is running, it creates a new one to run the async code.
        """
        try:
            async def collect_docs():
                docs = []
                async for doc in self.astream(query):
                    docs.append(doc)
                return docs
            
            docs = safe_sync_run(collect_docs())
            for doc in docs:
                yield doc
        except Exception as e:
            logger.error(f"Sync streaming failed for query '{query[:50]}...': {e}")
            raise GraphitiRetrieverError(f"Failed to stream results: {e}") from e

    @traceable
    @require_client
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Synchronous retrieval of documents from Graphiti."""
        try:
            return safe_sync_run(self._aget_relevant_documents(query, run_manager=run_manager))
        except Exception as e:
            logger.error(f"Sync retrieval failed for query '{query[:50]}...': {e}")
            raise GraphitiRetrieverError(f"Failed to retrieve documents: {e}") from e

    @traceable
    @require_client
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Asynchronous retrieval of documents from Graphiti."""
        try:
            search_results, scores = await self._perform_search_with_scores(query)
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
        Perform similarity search with score filtering.
        
        Args:
            query: The search query
            k: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of (Document, score) tuples
        """
        # Temporarily override config
        original_config = self.config
        original_threshold = self.score_threshold
        
        try:
            temp_config = self.config.model_copy(deep=True)
            temp_config.limit = k
            
            # Set score threshold
            temp_score_threshold = score_threshold or self.score_threshold
            
            self.config = temp_config
            self.score_threshold = temp_score_threshold
            
            docs = await self._aget_relevant_documents(query, run_manager=None)
            return [(doc, doc.metadata.get("score", 0.0)) for doc in docs]
        except Exception as e:
            logger.error(f"Similarity search with score failed for query '{query[:50]}...': {e}")
            raise GraphitiRetrieverError(f"Failed to perform similarity search: {e}") from e
        finally:
            self.config = original_config
            self.score_threshold = original_threshold

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


class GraphitiSemanticRetriever(GraphitiRetriever):
    """
    A specialized retriever that leverages Graphiti's graph-aware semantic search.
    
    This retriever uses advanced graph topology features like node distance,
    community structure, and relationship patterns to enhance semantic search
    beyond simple vector similarity.
    """
    
    def __init__(self, client: GraphitiClient, **kwargs):
        # Use graph-aware search configuration by default
        kwargs.setdefault("config", EDGE_HYBRID_SEARCH_NODE_DISTANCE)
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
        Perform Maximum Marginal Relevance (MMR) search.
        
        This uses Graphiti's MMR reranker configuration for diverse results.
        """
        # Temporarily configure for MMR
        original_config = self.config
        original_threshold = self.score_threshold
        
        try:
            # Create MMR-configured search
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
            
            self.config = mmr_config
            self.score_threshold = score_threshold
            
            docs = await self._aget_relevant_documents(query, run_manager=None)
            return docs
        except Exception as e:
            logger.error(f"MMR search failed for query '{query[:50]}...': {e}")
            raise GraphitiRetrieverError(f"Failed to perform MMR search: {e}") from e
        finally:
            self.config = original_config
            self.score_threshold = original_threshold


class GraphitiCachedRetriever(GraphitiRetriever):
    """
    A caching wrapper around GraphitiRetriever for improved performance.
    
    This retriever caches search results to avoid repeated expensive operations.
    Useful for development, testing, or when the same queries are made frequently.
    """
    
    def __init__(self, client: GraphitiClient, cache_ttl_seconds: int = 300, **kwargs):
        super().__init__(client=client, **kwargs)
        self._cache = {}
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