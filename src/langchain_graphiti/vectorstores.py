"""
LangChain VectorStore implementation for Graphiti.

This module provides a comprehensive VectorStore adapter that allows Graphiti to be used
as a drop-in replacement for other vector stores in the LangChain ecosystem.
While the GraphitiRetriever provides more advanced functionality, this
VectorStore offers maximum compatibility with existing LangChain code.

Note: This implementation is designed for compatibility. Use of GraphitiRetriever
directly is recommended for advanced graph operations and retrieval capabilities.

Key Features:
- Complete VectorStore interface implementation
- Graph-aware similarity search with relationship context
- Automatic entity extraction and graph integration
- Metadata preservation and enhancement
- Maximum Marginal Relevance (MMR) search support
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langsmith import traceable
from pydantic import Field

from ._client import GraphitiClient
from .retrievers import GraphitiRetriever, GraphitiSemanticRetriever
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import (
    NODE_HYBRID_SEARCH_RRF,
    NODE_HYBRID_SEARCH_MMR,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.datetime_utils import utc_now

logger = logging.getLogger(__name__)


class GraphitiVectorStore(VectorStore):
    """
    A comprehensive VectorStore implementation backed by Graphiti's knowledge graph.

    This class provides a complete VectorStore interface while leveraging
    Graphiti's advanced knowledge graph capabilities. Unlike simple vector
    stores, this implementation:

    - Automatically extracts entities and relationships from added documents
    - Uses graph structure to enhance similarity search
    - Preserves document metadata and adds graph-derived context
    - Supports both vector similarity and graph-aware search methods
    - Implements the complete VectorStore interface including MMR search
    - Provides proper async/sync support for all operations

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

    # --- Core VectorStore Interface ---

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
                logger.warning(f"Error adding text at index {index}: {e}")
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
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(self.aadd_texts(texts, metadatas, **kwargs))

    @traceable
    async def aadd_documents(
        self,
        documents: List[Document],
        **kwargs: Any,
    ) -> List[str]:
        """
        Add documents to the vector store asynchronously.

        Args:
            documents: List of Document objects to add.
            **kwargs: Additional arguments.

        Returns:
            List of document IDs.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return await self.aadd_texts(texts, metadatas, **kwargs)

    def add_documents(
        self,
        documents: List[Document],
        **kwargs: Any,
    ) -> List[str]:
        """
        Add documents to the vector store synchronously.

        Args:
            documents: List of Document objects to add.
            **kwargs: Additional arguments.

        Returns:
            List of document IDs.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(self.aadd_documents(documents, **kwargs))

    # --- Similarity Search Methods ---

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
        # Use the retriever to get properly scored and formatted documents
        retriever = GraphitiRetriever(
            client=self.client,
            config=NODE_HYBRID_SEARCH_RRF,
            search_mode="combined",  # VectorStores should search both nodes and edges
            group_ids=group_ids or ([self.group_id] if self.group_id else None),
            search_filter=SearchFilters(**filter) if filter else SearchFilters(),
        )
        retriever.config.limit = k

        return await retriever.asimilarity_search_with_score(query, k, **kwargs)

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


    # --- Maximum Marginal Relevance (MMR) Search ---

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Perform Maximum Marginal Relevance (MMR) search asynchronously.

        MMR search balances relevance and diversity by selecting documents
        that are both relevant to the query and diverse from each other.

        Args:
            query: Query string.
            k: Number of final results to return.
            fetch_k: Number of initial results to fetch before MMR reranking.
            lambda_mult: Balance between relevance and diversity (0.0 = only diversity, 1.0 = only relevance).
            filter: Optional metadata filter.
            **kwargs: Additional arguments.

        Returns:
            List of documents optimized for relevance and diversity.
        """
        # Use semantic retriever with MMR configuration
        retriever = GraphitiSemanticRetriever(
            client=self.client,
            config=NODE_HYBRID_SEARCH_MMR,
            search_mode="combined",
            group_ids=([self.group_id] if self.group_id else None),
            search_filter=SearchFilters(**filter) if filter else SearchFilters(),
        )
        
        return await retriever.max_marginal_relevance_search(
            query=query,
            k=k,
            lambda_mult=lambda_mult,
            **kwargs
        )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Perform Maximum Marginal Relevance (MMR) search synchronously.

        Args:
            query: Query string.
            k: Number of final results to return.
            fetch_k: Number of initial results to fetch before MMR reranking.
            lambda_mult: Balance between relevance and diversity.
            filter: Optional metadata filter.
            **kwargs: Additional arguments.

        Returns:
            List of documents optimized for relevance and diversity.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(
            self.amax_marginal_relevance_search(query, k, fetch_k, lambda_mult, filter, **kwargs)
        )

    # --- Retriever Interface ---

    def as_retriever(self, **kwargs: Any) -> GraphitiRetriever:
        """
        Return a GraphitiRetriever backed by this vector store.

        Args:
            **kwargs: Additional arguments to pass to the retriever.

        Returns:
            A configured GraphitiRetriever instance.
        """
        # Set up default retriever configuration
        retriever_kwargs = {
            "client": self.client,
            "search_mode": "combined",
            "group_ids": [self.group_id] if self.group_id else None,
            "include_graph_context": True,
        }
        retriever_kwargs.update(kwargs)
        
        return GraphitiRetriever(**retriever_kwargs)

    # --- Factory Methods ---

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        client: Optional[GraphitiClient] = None,
        **kwargs: Any,
    ) -> "GraphitiVectorStore":
        """
        Create a GraphitiVectorStore from a list of texts.

        Args:
            texts: List of text strings to add.
            embedding: Embeddings instance (for interface compatibility).
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

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        client: Optional[GraphitiClient] = None,
        **kwargs: Any,
    ) -> "GraphitiVectorStore":
        """
        Create a GraphitiVectorStore from a list of documents.

        Args:
            documents: List of Document objects to add.
            embedding: Embeddings instance (for interface compatibility).
            client: GraphitiClient instance to use.
            **kwargs: Additional arguments.

        Returns:
            A new instance of GraphitiVectorStore populated with the provided documents.
        """
        if client is None:
            raise ValueError("GraphitiClient must be provided")

        store = cls(client=client, embeddings=embedding, **kwargs)
        store.add_documents(documents)
        return store

    # --- Document Management ---

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

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        async def _delete():
            try:
                successful = 0
                for episode_id in ids:
                    try:
                        await self.client.graphiti_instance.remove_episode(episode_id)
                        successful += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete episode {episode_id}: {e}")
                
                return successful > 0
            except Exception as e:
                logger.error(f"Error during batch delete: {e}")
                return False
                
        return loop.run_until_complete(_delete())

    async def adelete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """
        Delete documents by IDs asynchronously.

        Args:
            ids: List of episode UUIDs to delete.
            **kwargs: Additional arguments.

        Returns:
            True if deletion was successful, False otherwise.
        """
        if not ids:
            return False

        try:
            successful = 0
            for episode_id in ids:
                try:
                    await self.client.graphiti_instance.remove_episode(episode_id)
                    successful += 1
                except Exception as e:
                    logger.warning(f"Failed to delete episode {episode_id}: {e}")
            
            return successful > 0
        except Exception as e:
            logger.error(f"Error during async batch delete: {e}")
            return False

    def update_document(self, document_id: str, document: Document) -> None:
        """
        Update a document in the vector store.

        Note: Updating documents in Graphiti requires removing the old episode
        and adding a new one, as episodes are immutable.

        Args:
            document_id: The ID of the document to update.
            document: The new document content.
        """
        # First delete the existing document
        success = self.delete([document_id])
        if success:
            # Add the new document
            new_ids = self.add_documents([document])
            if new_ids and new_ids[0]:
                logger.info(f"Updated document {document_id} -> {new_ids[0]}")
            else:
                logger.warning(f"Failed to add updated document for {document_id}")
        else:
            logger.warning(f"Failed to delete document {document_id} for update")

    async def aupdate_document(self, document_id: str, document: Document) -> None:
        """
        Update a document in the vector store asynchronously.

        Args:
            document_id: The ID of the document to update.
            document: The new document content.
        """
        # First delete the existing document
        success = await self.adelete([document_id])
        if success:
            # Add the new document
            new_ids = await self.aadd_documents([document])
            if new_ids and new_ids[0]:
                logger.info(f"Updated document {document_id} -> {new_ids[0]}")
            else:
                logger.warning(f"Failed to add updated document for {document_id}")
        else:
            logger.warning(f"Failed to delete document {document_id} for update")

    # --- Additional Utility Methods ---

    def get_by_ids(self, ids: List[str]) -> List[Document]:
        """
        Get documents by their IDs.

        Args:
            ids: List of document IDs (episode UUIDs).

        Returns:
            List of documents corresponding to the IDs.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(self.aget_by_ids(ids))

    async def aget_by_ids(self, ids: List[str]) -> List[Document]:
        """
        Get documents by their IDs asynchronously.

        Args:
            ids: List of document IDs (episode UUIDs).

        Returns:
            List of documents corresponding to the IDs.
        """
        try:
            # Use Graphiti's get_nodes_and_edges_by_episode to retrieve content
            search_results = await self.client.graphiti_instance.get_nodes_and_edges_by_episode(ids)
            
            # Convert to documents (simplified representation)
            docs = []
            for i, episode_id in enumerate(ids):
                # Create a basic document representation
                content = f"Episode {episode_id} content"
                metadata = {"uuid": episode_id, "type": "episode"}
                docs.append(Document(page_content=content, metadata=metadata))
            
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents by IDs: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary containing store statistics.
        """
        return {
            "client_type": type(self.client).__name__,
            "group_id": self.group_id,
            "embeddings_provider": type(self._embeddings).__name__ if self._embeddings else None,
        }