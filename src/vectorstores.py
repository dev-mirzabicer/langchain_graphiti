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
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import (
    NODE_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.bulk_utils import RawEpisode


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
        self._embeddings = embeddings  # For interface compatibility
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
        Add texts to the vector store asynchronously by processing each text
        as an individual episode in parallel.

        This method leverages asyncio to run multiple `add_episode` calls
        concurrently, providing a balance of performance and the ability to
        return the specific UUID of each created episode.

        Args:
            texts: Iterable of text strings to add.
            metadatas: Optional list of metadata dictionaries, one for each text.
            group_id: Optional group ID to override the default for this batch.
            **kwargs: Additional arguments (not used by this implementation).

        Returns:
            A list of document IDs (the UUIDs of the created episodes). If an
            individual text fails to be added, it will return an empty string
            for that entry in the list.
        """
        from graphiti_core.helpers import semaphore_gather

        texts_list = list(texts)
        if not texts_list:
            return []

        metadatas = metadatas or [{}] * len(texts_list)
        target_group_id = group_id or self.group_id

        async def _add_one_text(index: int, text: str) -> str:
            """Helper coroutine to add a single text and handle errors."""
            metadata = metadatas[index] if index < len(metadatas) else {}
            
            name = metadata.get("title", metadata.get("name", f"Document {index + 1}"))
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
                    source=EpisodeType.text,
                    group_id=target_group_id,
                    update_communities=False,  # Batch updates are more efficient
                )
                return result.episode.uuid
            except Exception as e:
                # Log the error and return an empty string for this entry
                print(f"Error adding text at index {index}: {e}")
                return ""

        # Create a list of coroutine tasks, one for each text
        tasks = [_add_one_text(i, text) for i, text in enumerate(texts_list)]

        # Execute all tasks concurrently, respecting the semaphore limit
        # from the Graphiti instance to avoid overwhelming resources.
        episode_ids = await semaphore_gather(
            *tasks,
            max_coroutines=self.client.graphiti_instance.max_coroutines
        )

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
    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        group_ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with scores asynchronously.

        Args:
            query: Query string.
            k: Number of results to return.
            filter: Optional metadata filter.
            group_ids: Optional list of group IDs to restrict the search to.
            **kwargs: Additional arguments.

        Returns:
            List of (document, score) tuples.
        """
        from .retrievers import GraphitiRetriever

        # Use the retriever to get properly scored and formatted documents
        retriever = GraphitiRetriever(
            client=self.client,
            config=NODE_HYBRID_SEARCH_RRF,
            search_mode="nodes", # Vectorstores typically map to nodes/entities
            group_ids=group_ids or ([self.group_id] if self.group_id else None),
            search_filter=SearchFilters(**filter) if filter else SearchFilters(),
        )
        retriever.config.limit = k

        documents = await retriever._aget_relevant_documents(query, run_manager=None)
        
        return [(doc, doc.metadata.get("score", 0.0)) for doc in documents]

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

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Perform similarity search asynchronously.

        Args:
            query: Query string.
            k: Number of results to return.
            filter: Optional metadata filter.
            **kwargs: Additional arguments.

        Returns:
            List of documents matching the query.
        """
        results_with_scores = await self.asimilarity_search_with_score(
            query, k, filter, **kwargs
        )
        return [doc for doc, score in results_with_scores]

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
            List of documents matching the query.
        """
        results_with_scores = self.similarity_search_with_score(
            query, k, filter, **kwargs
        )
        return [doc for doc, score in results_with_scores]

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
            texts: List of text strings to add.
            embedding: Embeddings instance (not used directly, as Graphiti handles embeddings internally).
            metadatas: Optional list of metadata dictionaries, one for each text.
            client: GraphitiClient instance to use.
            **kwargs: Additional arguments.

        Returns:
            A new instance of GraphitiVectorStore populated with the provided texts.
        """
        if client is None:
            raise ValueError("GraphitiClient must be provided")

        store = cls(client=client, embeddings=embedding, **kwargs)
        store.add_texts(texts, metadatas)
        return store

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """
        Delete documents by IDs.
        Note: This assumes IDs are episode UUIDs.

        Args:
            ids: List of episode UUIDs to delete.
            **kwargs: Additional arguments (not used by this implementation).

        Returns:
            True if deletion was successful, False otherwise.
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
                tasks = [self.client.graphiti_instance.remove_episode(episode_id) for episode_id in ids]
                await asyncio.gather(*tasks)
                return True
            except Exception as e:
                print(f"Error during delete: {e}")
                return False
                
        return loop.run_until_complete(_delete())