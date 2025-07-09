"""
LangChain VectorStore implementation for Graphiti.

This module provides a VectorStore adapter that allows Graphiti to be used
as a drop-in replacement for other vector stores in the LangChain ecosystem.
While the GraphitiRetriever provides more advanced functionality, this
VectorStore offers maximum compatibility with existing LangChain code.

Key Features:
- Standard VectorStore interface for maximum compatibility
- Graph-aware similarity search with relationship context
- Automatic entity extraction and graph integration
- Metadata preservation and enhancement
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langsmith import traceable
from pydantic import Field

from ._client import GraphitiClient
from graphiti_core.nodes import EntityNode, EpisodeType
from graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
    NODE_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.datetime_utils import utc_now


class GraphitiVectorStore(VectorStore):
    """
    A VectorStore implementation backed by Graphiti's knowledge graph.

    This class provides a familiar VectorStore interface while leveraging
    Graphiti's advanced knowledge graph capabilities. Unlike simple vector
    stores, this implementation:

    - Automatically extracts entities and relationships from added documents
    - Uses graph structure to enhance similarity search
    - Preserves document metadata and adds graph-derived context
    - Supports both vector similarity and graph-aware search methods

    Note: This implementation is designed for compatibility. For advanced
    graph operations, use GraphitiRetriever directly.
    """

    def __init__(
        self,
        client: GraphitiClient,
        embeddings: Optional[Embeddings] = None,
        group_id: str = "",
        **kwargs: Any,
    ):
        """
        Initialize the GraphitiVectorStore.

        Args:
            client: The GraphitiClient instance to use.
            embeddings: Optional embeddings instance (not used directly,
                       as Graphiti handles embeddings internally).
            group_id: Default group ID for document isolation.
            **kwargs: Additional arguments.
        """
        self.client = client
        self.embeddings = embeddings  # For interface compatibility
        self.group_id = group_id
        super().__init__(**kwargs)

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Return the embeddings instance (for interface compatibility)."""
        return self._embeddings

    @embeddings.setter
    def embeddings(self, value: Optional[Embeddings]) -> None:
        """Set the embeddings instance."""
        self._embeddings = value

    @traceable
    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        group_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add texts to the vector store asynchronously.

        Args:
            texts: Iterable of text strings to add.
            metadatas: Optional list of metadata dictionaries.
            group_id: Optional group ID to override default.
            **kwargs: Additional arguments.

        Returns:
            List of document IDs (episode UUIDs in Graphiti).
        """
        texts_list = list(texts)
        metadatas = metadatas or [{}] * len(texts_list)
        group_id = group_id or self.group_id

        episode_ids = []

        for i, text in enumerate(texts_list):
            metadata = metadatas[i] if i < len(metadatas) else {}
            
            # Extract name from metadata or generate from text
            name = metadata.get("title", metadata.get("name", f"Document {i+1}"))
            source_description = metadata.get(
                "source_description", 
                f"Added to vector store at {utc_now().isoformat()}"
            )

            try:
                # Add as an episode to Graphiti
                result = await self.client.graphiti_instance.add_episode(
                    name=name,
                    episode_body=text,
                    source_description=source_description,
                    reference_time=utc_now(),
                    source=EpisodeType.TEXT,
                    group_id=group_id,
                    update_communities=False,  # Batch update later if needed
                )
                
                episode_ids.append(result.episode.uuid)
                
            except Exception as e:
                # Log error but continue with other documents
                print(f"Warning: Failed to add text {i}: {e}")
                episode_ids.append(f"error_{i}")

        return episode_ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add texts to the vector store synchronously.

        Args:
            texts: Iterable of text strings to add.
            metadatas: Optional list of metadata dictionaries.
            **kwargs: Additional arguments.

        Returns:
            List of document IDs.
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(self.aadd_texts(texts, metadatas, **kwargs))

    @traceable
    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        group_ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Perform similarity search asynchronously.

        Args:
            query: Query string.
            k: Number of results to return.
            filter: Optional metadata filter (not implemented yet).
            group_ids: Optional list of group IDs to search within.
            **kwargs: Additional arguments.

        Returns:
            List of similar documents.
        """
        # Use a simplified search config focused on nodes
        search_config = NODE_HYBRID_SEARCH_RRF
        search_config.limit = k

        group_ids = group_ids or ([self.group_id] if self.group_id else None)

        search_results = await self.client.graphiti_instance.search_(
            query=query,
            config=search_config,
            group_ids=group_ids,
            search_filter=SearchFilters(),  # TODO: Convert filter dict to SearchFilters
        )

        documents = []

        # Convert nodes to documents
        for node in search_results.nodes:
            metadata = {
                "source": "graphiti_node",
                "node_uuid": node.uuid,
                "node_name": node.name,
                "node_labels": node.labels,
                "group_id": node.group_id,
                "created_at": node.created_at.isoformat(),
                **node.attributes,
            }
            
            content = f"**{node.name}**\n\n{node.summary}" if node.summary else node.name
            documents.append(Document(page_content=content, metadata=metadata))

        # Convert edges to documents if no nodes found
        if not documents:
            for edge in search_results.edges[:k]:
                metadata = {
                    "source": "graphiti_edge",
                    "edge_uuid": edge.uuid,
                    "edge_name": edge.name,
                    "source_node_uuid": edge.source_node_uuid,
                    "target_node_uuid": edge.target_node_uuid,
                    "group_id": edge.group_id,
                    "created_at": edge.created_at.isoformat(),
                    **edge.attributes,
                }
                
                content = f"[{edge.name}] {edge.fact}"
                documents.append(Document(page_content=content, metadata=metadata))

        return documents[:k]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Perform similarity search synchronously.

        Args:
            query: Query string.
            k: Number of results to return.
            filter: Optional metadata filter.
            **kwargs: Additional arguments.

        Returns:
            List of similar documents.
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(self.asimilarity_search(query, k, filter, **kwargs))

    @traceable
    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with scores asynchronously.

        Args:
            query: Query string.
            k: Number of results to return.
            filter: Optional metadata filter.
            **kwargs: Additional arguments.

        Returns:
            List of (document, score) tuples.
        """
        documents = await self.asimilarity_search(query, k, filter, **kwargs)
        
        # Graphiti doesn't expose raw similarity scores in the current API
        # So we return a dummy score. In a real implementation, this would
        # require extending Graphiti's search API to return scores.
        return [(doc, 1.0) for doc in documents]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with scores synchronously.

        Args:
            query: Query string.
            k: Number of results to return.
            filter: Optional metadata filter.
            **kwargs: Additional arguments.

        Returns:
            List of (document, score) tuples.
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(
            self.asimilarity_search_with_score(query, k, filter, **kwargs)
        )

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        client: Optional[GraphitiClient] = None,
        **kwargs: Any,
    ) -> GraphitiVectorStore:
        """
        Create a GraphitiVectorStore from a list of texts.

        Args:
            texts: List of text documents.
            embedding: Embeddings instance (for compatibility).
            metadatas: Optional list of metadata dictionaries.
            client: GraphitiClient instance.
            **kwargs: Additional arguments.

        Returns:
            A new GraphitiVectorStore instance with texts added.
        """
        if client is None:
            raise ValueError("GraphitiClient must be provided")

        store = cls(client=client, embeddings=embedding, **kwargs)
        store.add_texts(texts, metadatas)
        return store

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """
        Delete documents by IDs.

        Args:
            ids: List of document IDs (episode UUIDs) to delete.
            **kwargs: Additional arguments.

        Returns:
            True if successful, False otherwise.
        """
        if not ids:
            return False

        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        async def _delete():
            try:
                for episode_id in ids:
                    await self.client.graphiti_instance.remove_episode(episode_id)
                return True
            except Exception:
                return False
                
        return loop.run_until_complete(_delete())