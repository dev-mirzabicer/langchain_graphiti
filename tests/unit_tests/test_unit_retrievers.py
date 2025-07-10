"""Unit tests for Graphiti retrievers."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta
import time

from langchain_core.documents import Document

from langchain_graphiti.retrievers import (
    GraphitiRetriever,
    GraphitiCachedRetriever,
    GraphitiRetrieverError,
)
from langchain_graphiti._client import GraphitiClient
from graphiti_core.search.search_config import SearchResults
from graphiti_core.nodes import EpisodicNode, EpisodeType, EntityNode
from graphiti_core.edges import EntityEdge


@pytest.fixture
def mock_graphiti_client() -> GraphitiClient:
    """Fixture for a mocked GraphitiClient."""
    client = MagicMock(spec=GraphitiClient)
    mock_graphiti_instance = MagicMock()

    async def mock_search(query: str, **kwargs):
        mock_episode = EpisodicNode(
            uuid="test-uuid-1",
            name=f"Episode for {query}",
            content=f"This is the content for query: {query}",
            source=EpisodeType.text,
            source_description="mock source",
            valid_at=datetime.now(),
            group_id="test-group",
        )
        return SearchResults(
            episodes=[mock_episode],
            nodes=[],
            edges=[],
            communities=[],
            query=query,
            query_vector=[0.1, 0.2, 0.3],
        )

    mock_graphiti_instance.search_ = AsyncMock(side_effect=mock_search)
    client.graphiti_instance = mock_graphiti_instance

    return client


class TestGraphitiRetrieverUnit:
    """Unit tests for GraphitiRetriever."""

    @pytest.mark.asyncio
    async def test_get_relevant_documents(self, mock_graphiti_client: GraphitiClient):
        """Test the retriever's get_relevant_documents method."""
        retriever = GraphitiRetriever(client=mock_graphiti_client)
        query = "What is the meaning of life?"

        async def mock_perform_search(query_text, **kwargs):
            mock_episode = EpisodicNode(
                uuid="test-uuid-2",
                name=f"Test Episode for '{query_text}'",
                content=f"Content related to '{query_text}'.",
                source=EpisodeType.message,
                source_description="mock source",
                valid_at=datetime.now(),
                group_id="unit-test",
            )
            search_results = SearchResults(
                episodes=[mock_episode], nodes=[], edges=[], communities=[]
            )
            scores = {"test-uuid-2": 0.95}
            return search_results, scores

        retriever._perform_search_with_scores = AsyncMock(
            side_effect=mock_perform_search
        )

        docs = await retriever.aget_relevant_documents(query)

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].page_content == f"Content related to '{query}'."
        assert docs[0].metadata["uuid"] == "test-uuid-2"
        assert docs[0].metadata["name"] == f"Test Episode for '{query}'"
        assert docs[0].metadata["group_id"] == "unit-test"

    @pytest.mark.asyncio
    async def test_stream_and_astream(self, mock_graphiti_client: GraphitiClient):
        """Test the stream and astream methods."""
        retriever = GraphitiRetriever(client=mock_graphiti_client)
        query = "test stream"

        mock_episode1 = EpisodicNode(
            uuid="s-uuid-1",
            name="Stream 1",
            content="First stream doc",
            group_id="test",
            source=EpisodeType.text,
            source_description="test",
            valid_at=datetime.now(),
        )
        mock_episode2 = EpisodicNode(
            uuid="s-uuid-2",
            name="Stream 2",
            content="Second stream doc",
            group_id="test",
            source=EpisodeType.text,
            source_description="test",
            valid_at=datetime.now(),
        )
        search_results = SearchResults(
            episodes=[mock_episode1, mock_episode2], nodes=[], edges=[], communities=[]
        )
        scores = {"s-uuid-1": 0.9, "s-uuid-2": 0.8}
        retriever._perform_search_with_scores = AsyncMock(
            return_value=(search_results, scores)
        )

        # Test astream
        astream_docs = [doc async for doc in retriever.astream(query)]
        assert len(astream_docs) == 2
        assert astream_docs[0].page_content == "First stream doc"

        # Test stream
        stream_docs = list(retriever.stream(query))
        assert len(stream_docs) == 2
        assert stream_docs[1].page_content == "Second stream doc"

    @pytest.mark.asyncio
    async def test_similarity_search_with_score(
        self, mock_graphiti_client: GraphitiClient
    ):
        """Test similarity search with score filtering."""
        retriever = GraphitiRetriever(client=mock_graphiti_client)
        query = "find similar things"

        mock_episode1 = EpisodicNode(
            uuid="sim-1",
            name="sim1",
            content="sim content 1",
            group_id="test",
            source=EpisodeType.text,
            source_description="test",
            valid_at=datetime.now(),
        )
        mock_episode2 = EpisodicNode(
            uuid="sim-2",
            name="sim2",
            content="sim content 2",
            group_id="test",
            source=EpisodeType.text,
            source_description="test",
            valid_at=datetime.now(),
        )
        search_results = SearchResults(
            episodes=[mock_episode1, mock_episode2], nodes=[], edges=[], communities=[]
        )
        scores = {"sim-1": 0.95, "sim-2": 0.75}
        retriever._perform_search_with_scores = AsyncMock(
            return_value=(search_results, scores)
        )

        # Mock _convert_search_results_to_docs to add scores to metadata
        async def mock_convert(results, scores_map):
            docs = []
            for episode in results.episodes:
                doc = Document(
                    page_content=episode.content,
                    metadata={"uuid": episode.uuid, "score": scores_map.get(episode.uuid)},
                )
                docs.append(doc)
            return docs

        retriever._convert_search_results_to_docs = AsyncMock(side_effect=mock_convert)

        results_with_scores = await retriever.asimilarity_search_with_score(
            query, k=2, score_threshold=0.8
        )

        assert len(results_with_scores) == 1
        doc, score = results_with_scores[0]
        assert doc.metadata["uuid"] == "sim-1"
        assert score == 0.95

    @pytest.mark.asyncio
    async def test_retriever_error_handling(self, mock_graphiti_client: GraphitiClient):
        """Test that retriever methods raise GraphitiRetrieverError on failure."""
        retriever = GraphitiRetriever(client=mock_graphiti_client)
        query = "this will fail"

        retriever._perform_search_with_scores = AsyncMock(
            side_effect=Exception("Internal search failed")
        )

        with pytest.raises(GraphitiRetrieverError, match="Failed to retrieve documents"):
            await retriever.aget_relevant_documents(query)

        with pytest.raises(GraphitiRetrieverError, match="Failed to stream results"):
            await retriever.astream(query).__anext__()

        with pytest.raises(
            GraphitiRetrieverError, match="Failed to perform similarity search"
        ):
            await retriever.asimilarity_search_with_score(query)


class TestGraphitiCachedRetrieverUnit:
    """Unit tests for the GraphitiCachedRetriever."""

    @pytest.fixture
    def mock_retriever(self) -> GraphitiCachedRetriever:
        """Fixture for a mocked GraphitiCachedRetriever."""
        client = MagicMock(spec=GraphitiClient)
        retriever = GraphitiCachedRetriever(client=client, cache_ttl_seconds=1)
        # Mock the underlying search method
        retriever._perform_search_with_scores = AsyncMock()
        return retriever

    @pytest.mark.asyncio
    async def test_cache_hit_and_miss(self, mock_retriever: GraphitiCachedRetriever):
        """Test that the cache is used on subsequent calls."""
        query = "test cache"
        mock_episode = EpisodicNode(
            uuid="cache-1",
            name="c1",
            content="cached content",
            group_id="test",
            source=EpisodeType.text,
            source_description="test",
            valid_at=datetime.now(),
        )
        search_results = SearchResults(episodes=[mock_episode], nodes=[], edges=[], communities=[])
        scores = {"cache-1": 1.0}
        mock_retriever._perform_search_with_scores.return_value = (
            search_results,
            scores,
        )

        # First call (cache miss)
        docs1 = await mock_retriever.aget_relevant_documents(query)
        assert mock_retriever._perform_search_with_scores.call_count == 1
        assert len(docs1) == 1
        assert docs1[0].page_content == "cached content"

        # Second call (cache hit)
        docs2 = await mock_retriever.aget_relevant_documents(query)
        assert mock_retriever._perform_search_with_scores.call_count == 1  # No new call
        assert len(docs2) == 1
        assert docs2[0].page_content == "cached content"

    @pytest.mark.asyncio
    async def test_cache_expiration(self, mock_retriever: GraphitiCachedRetriever):
        """Test that the cache expires after the TTL."""
        query = "test expiration"
        mock_retriever._perform_search_with_scores.return_value = (
            SearchResults(
                episodes=[
                    EpisodicNode(
                        uuid="exp-1",
                        name="e1",
                        content="exp",
                        group_id="test",
                        source=EpisodeType.text,
                        source_description="test",
                        valid_at=datetime.now(),
                    )
                ],
                nodes=[],
                edges=[],
                communities=[],
            ),
            {"exp-1": 1.0},
        )

        # First call
        await mock_retriever.aget_relevant_documents(query)
        assert mock_retriever._perform_search_with_scores.call_count == 1

        # Wait for TTL to expire
        time.sleep(1.1)

        # Second call (should be a miss again)
        await mock_retriever.aget_relevant_documents(query)
        assert mock_retriever._perform_search_with_scores.call_count == 2

    def test_clear_cache(self, mock_retriever: GraphitiCachedRetriever):
        """Test clearing the cache."""
        mock_retriever._cache["some_key"] = ("some_data", time.time())
        assert len(mock_retriever._cache) == 1

        mock_retriever.clear_cache()
        assert len(mock_retriever._cache) == 0