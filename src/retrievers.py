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
from typing import Any, Dict, Iterator, List, Literal, Optional, Union

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
        default=None, description="Optional filters to apply to the search."
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
    async def _convert_search_results_to_docs(
        self, search_results: SearchResults
    ) -> List[Document]:
        """Converts Graphiti search results into LangChain Documents with rich metadata."""
        docs = []
        
        # Convert edges to documents
        if self.search_mode in ["edges", "combined"]:
            for edge in search_results.edges:
                # Build rich metadata with graph context
                metadata = {
                    "type": "edge",
                    "uuid": edge.uuid,
                    "name": edge.name,
                    "source_node_uuid": edge.source_node_uuid,
                    "target_node_uuid": edge.target_node_uuid,
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
                        "confidence_score": getattr(edge, 'score', None),
                        "embedding_available": edge.fact_embedding is not None,
                    }
                
                # Use the fact as the main content, with additional context
                content = edge.fact
                if edge.name and edge.name != edge.fact:
                    content = f"[{edge.name}] {edge.fact}"
                
                doc = Document(page_content=content, metadata=metadata)
                docs.append(doc)

        # Convert nodes to documents
        if self.search_mode in ["nodes", "combined"]:
            for node in search_results.nodes:
                metadata = {
                    "type": "node",
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
                        "node_type": node.labels[0] if node.labels else "Entity",
                        "embedding_available": node.name_embedding is not None,
                        # Could add degree, centrality, etc. here if available
                    }
                
                # Use summary as content, with name as title
                content = f"**{node.name}**\n\n{node.summary}" if node.summary else node.name
                
                doc = Document(page_content=content, metadata=metadata)
                docs.append(doc)
                
        # Convert episodes to documents (if any in results)
        for episode in search_results.episodes:
            metadata = {
                "type": "episode",
                "uuid": episode.uuid,
                "name": episode.name,
                "group_id": episode.group_id,
                "created_at": episode.created_at.isoformat(),
                "source_description": episode.source_description,
                "content": episode.content,
            }
            
            doc = Document(
                page_content=f"**{episode.name}**\n\n{episode.content}", 
                metadata=metadata
            )
            docs.append(doc)
            
        # Convert communities to documents (if any in results)
        for community in search_results.communities:
            metadata = {
                "type": "community",
                "uuid": community.uuid,
                "name": community.name,
                "group_id": community.group_id,
                "created_at": community.created_at.isoformat(),
                "summary": community.summary,
            }
            
            content = f"**Community: {community.name}**\n\n{community.summary}"
            doc = Document(page_content=content, metadata=metadata)
            docs.append(doc)

        return docs

    def stream(self, query: str) -> Iterator[Document]:
        """
        Stream documents one by one. 
        
        Note: Graphiti's backend doesn't stream natively, so this provides
        a streaming interface over the complete results.
        """
        # Run the async search in a new event loop if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        if loop.is_running():
            # If we're in an async context, we need to use a different approach
            # This is a limitation - ideally we'd use astream in async contexts
            raise RuntimeError(
                "Cannot use stream() in an async context. Use aget_relevant_documents() instead."
            )
        
        search_results = loop.run_until_complete(
            self.client.graphiti_instance.search_(
                query=query,
                config=self.config,
                group_ids=self.group_ids,
                search_filter=self.search_filter,
            )
        )
        
        docs = loop.run_until_complete(self._convert_search_results_to_docs(search_results))
        
        # Stream documents one by one
        for doc in docs:
            yield doc

    @traceable
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Synchronous retrieval of documents from Graphiti.
        
        Args:
            query: The query string.
            run_manager: The callback manager for the run.

        Returns:
            A list of relevant documents.
        """
        # Use asyncio to run the async method
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
        """
        Asynchronous retrieval of documents from Graphiti.

        Args:
            query: The query string.
            run_manager: The callback manager for the run.

        Returns:
            A list of relevant documents.
        """
        search_results = await self.client.graphiti_instance.search_(
            query=query,
            config=self.config,
            group_ids=self.group_ids,
            search_filter=self.search_filter,
        )
        return await self._convert_search_results_to_docs(search_results)


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
    async def similarity_search_with_score_threshold(
        self, 
        query: str, 
        score_threshold: float = 0.5,
        k: int = 10
    ) -> List[tuple[Document, float]]:
        """
        Perform similarity search with score filtering.
        
        Args:
            query: The search query
            score_threshold: Minimum similarity score threshold
            k: Maximum number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        # Create a custom config with score threshold
        custom_config = SearchConfig(
            edge_config=self.config.edge_config,
            node_config=self.config.node_config,
            episode_config=self.config.episode_config,
            community_config=self.config.community_config,
            limit=k,
            reranker_min_score=score_threshold,
        )
        
        # Temporarily use the custom config
        original_config = self.config
        self.config = custom_config
        
        try:
            docs = await self._aget_relevant_documents(query, run_manager=None)
            # Return with dummy scores since Graphiti doesn't expose raw scores
            # In a real implementation, we'd extract scores from the search results
            return [(doc, 1.0) for doc in docs]
        finally:
            self.config = original_config
