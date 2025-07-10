"""Integration tests for GraphitiCachedRetriever."""

import pytest
import time
from datetime import datetime
from unittest.mock import patch
from langchain_graphiti.retrievers import GraphitiCachedRetriever
from langchain_graphiti._client import GraphitiClient


@pytest.mark.asyncio
async def test_graphiti_cached_retriever_integration(client_for_integration: GraphitiClient):
    """
    Test the GraphitiCachedRetriever against a live database.
    
    This test performs the following steps:
    1. Creates a GraphitiClient to connect to the database.
    2. Cleans up any old data and sets up indices.
    3. Adds a test document (episode) to the graph.
    4. Creates a GraphitiCachedRetriever instance.
    5. Makes an initial query to populate the cache.
    6. Makes a second query to verify that the cache is used.
    7. Clears the cache and verifies that a new query hits the database again.
    8. Cleans up the test data.
    """
    client = client_for_integration
    test_group_id = "cached-retriever-integration-test"
    test_content = "This is a test for the cached retriever."
    test_doc_name = "Test Document for Cached Retriever"

    try:
        # 1. Setup
        await client.graphiti_instance.build_indices_and_constraints(delete_existing=True)
        await client.graphiti_instance.add_episode(
            name=test_doc_name,
            episode_body=test_content,
            source_description="Integration test",
            group_id=test_group_id,
            reference_time=datetime.now(),
        )

        # 2. Create the retriever
        retriever = GraphitiCachedRetriever(
            client=client,
            group_ids=[test_group_id],
            cache_ttl_seconds=10,
        )

        # 3. First call (cache miss)
        with patch.object(retriever, '_perform_search_with_scores', wraps=retriever._perform_search_with_scores) as mock_search:
            docs1 = await retriever.aget_relevant_documents("cached retriever")
            assert mock_search.call_count == 1
            assert len(docs1) > 0
            assert docs1[0].page_content == test_content

            # 4. Second call (cache hit)
            docs2 = await retriever.aget_relevant_documents("cached retriever")
            assert mock_search.call_count == 1  # Should not be called again
            assert len(docs2) == 1
            assert docs2[0].page_content == test_content

        # 5. Clear cache and verify
        retriever.clear_cache()
        with patch.object(retriever, '_perform_search_with_scores', wraps=retriever._perform_search_with_scores) as mock_search:
            docs3 = await retriever.aget_relevant_documents("cached retriever")
            assert mock_search.call_count == 1  # Should be called again
            assert len(docs3) > 0

    finally:
        # 6. Teardown
        await client.graphiti_instance.driver.execute_query(
            "MATCH (n {group_id: $group_id}) DETACH DELETE n",
            group_id=test_group_id,
        )