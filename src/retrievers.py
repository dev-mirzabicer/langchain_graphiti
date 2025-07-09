"""
LangChain retrievers for the Graphiti knowledge graph system.

This module provides powerful retriever implementations that leverage Graphiti's
advanced search capabilities, including hybrid search (text, vector, and graph-based),
sophisticated reranking, and graph-aware semantic search.

Key Components:
- GraphitiRetriever: The primary retriever with full configurability
- GraphitiSemanticRetriever: Specialized for graph-aware semantic search
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple

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
from graphiti_core.search.search_config import SearchConfig, SearchResults
from graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
    EDGE_HYBRID_SEARCH_NODE_DISTANCE,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search import search as graphiti_search_internal

class GraphitiRetriever(BaseRetriever):
    """
    A powerful retriever for the Graphiti knowledge graph system.

    This retriever uses Graphiti's advanced search capabilities including:
    - Hybrid search combining text, vector, and graph traversal
    - Sophisticated reranking (RRF, MMR, cross-encoder)
    - Graph-aware context preservation
    - Runtime configuration support

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
            **field_updates
        )

    @traceable
    async def _search_and_get_results_with_scores(
        self, query: str
    ) -> Tuple[SearchResults, Dict[str, float]]:
        """
        Internal method to run Graphiti search and reconstruct scores.
        This is a workaround since the core Graphiti search function doesn't
        propagate scores directly in its return object.
        """
        # Graphiti's internal search function
        graphiti_instance = self.client.graphiti_instance
        
        # We need to generate the query vector once
        query_vector = await graphiti_instance.embedder.create(input_data=[query.replace('\n', ' ')])

        # The search function in graphiti.py calls lower-level functions in search.py
        # which do have scores. We will call the main search function and then
        # re-rank the top results with the cross-encoder to get reliable scores.
        # This is the most robust way to get scores for the final ranked list.
        
        search_results = await graphiti_search_internal(
            clients=graphiti_instance.clients,
            query=query,
            group_ids=self.group_ids,
            config=self.config,
            search_filter=self.search_filter,
            query_vector=query_vector,
        )

        scores = {}
        
        # To get meaningful scores, we can re-rank the top results using the cross-encoder
        # This provides a unified scoring mechanism for the final document list.
        passages_to_rank = []
        passage_to_uuid = {}

        if self.search_mode in ["edges", "combined"]:
            for edge in search_results.edges:
                passage = f"[{edge.name}] {edge.fact}"
                passages_to_rank.append(passage)
                passage_to_uuid[passage] = edge.uuid

        if self.search_mode in ["nodes", "combined"]:
            for node in search_results.nodes:
                passage = f"Entity: {node.name}. Summary: {node.summary}"
                passages_to_rank.append(passage)
                passage_to_uuid[passage] = node.uuid

        if passages_to_rank:
            ranked_passages = await graphiti_instance.cross_encoder.rank(query, passages_to_rank)
            for passage, score in ranked_passages:
                uuid = passage_to_uuid.get(passage)
                if uuid:
                    scores[uuid] = score
        
        return search_results, scores

    @traceable
    async def _convert_search_results_to_docs(
        self, search_results: SearchResults, scores: Dict[str, float]
    ) -> List[Document]:
        """Converts Graphiti search results into LangChain Documents with rich metadata."""
        docs = []
        
        # Create a lookup for node names to enrich edge documents
        node_uuid_to_name = {node.uuid: node.name for node in search_results.nodes}

        # Convert edges to documents
        if self.search_mode in ["edges", "combined"]:
            for edge in search_results.edges:
                source_name = node_uuid_to_name.get(edge.source_node_uuid, "Unknown Entity")
                target_name = node_uuid_to_name.get(edge.target_node_uuid, "Unknown Entity")
                
                metadata = {
                    "type": "edge",
                    "score": scores.get(edge.uuid),
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

        # Convert nodes to documents
        if self.search_mode in ["nodes", "combined"]:
            for node in search_results.nodes:
                metadata = {
                    "type": "node",
                    "score": scores.get(node.uuid),
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
        
        # Sort final documents by score if available
        docs.sort(key=lambda d: d.metadata.get("score", 0.0), reverse=True)

        return docs

    def stream(self, query: str) -> Iterator[Document]:
        """
        Stream documents one by one. 
        
        Note: Graphiti's backend doesn't stream natively, so this provides
        a streaming interface over the complete results.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        # Ensure the new event loop is running
        if loop.is_running():
            # If we're in an async context, we need to use a different approach
            # TODO: This is a limitation, ideally we'd use astream in async contexts
            raise RuntimeError(
                "Cannot use stream() in an async context. Use aget_relevant_documents() instead."
            )
        
        search_results, scores = loop.run_until_complete(
            self._search_and_get_results_with_scores(query)
        )
        
        docs = loop.run_until_complete(self._convert_search_results_to_docs(search_results, scores))
        
        for doc in docs:
            yield doc

    @traceable
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Synchronous retrieval of documents from Graphiti."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(self._aget_relevant_documents(query, run_manager=run_manager))

    @traceable
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Asynchronous retrieval of documents from Graphiti."""
        search_results, scores = await self._search_and_get_results_with_scores(query)
        return await self._convert_search_results_to_docs(search_results, scores)


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

    @traceable
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
        # Create a temporary config with the specified limit and score threshold
        original_config = self.config
        temp_config = self.config.copy(deep=True)
        temp_config.limit = k
        if score_threshold is not None:
            temp_config.reranker_min_score = score_threshold
        
        self.config = temp_config
        
        try:
            docs = await self._aget_relevant_documents(query, run_manager=None)
            return [(doc, doc.metadata.get("score", 0.0)) for doc in docs]
        finally:
            self.config = original_config

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 10,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Synchronous version of asimilarity_search_with_score."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.asimilarity_search_with_score(query, k, score_threshold, **kwargs)
        )