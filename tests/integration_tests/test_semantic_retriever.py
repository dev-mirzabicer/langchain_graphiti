"""Integration tests for GraphitiSemanticRetriever."""

import pytest
import asyncio
from datetime import datetime
from langchain_core.documents import Document
from langchain_graphiti import create_graphiti_client
from langchain_graphiti.retrievers import GraphitiSemanticRetriever


@pytest.mark.asyncio
async def test_graphiti_semantic_retriever_integration():
    """
    Test the GraphitiSemanticRetriever against a live database.
    
    This test performs the following steps:
    1. Creates a GraphitiClient to connect to the database.
    2. Cleans up any old data and sets up indices.
    3. Adds a test document (episode) to the graph.
    4. Creates a GraphitiSemanticRetriever instance.
    5. Uses the retriever to search for the test document.
    6. Asserts that the correct document is returned.
    7. Cleans up the test data.
    """
    try:
        client = create_graphiti_client()
    except Exception as e:
        pytest.skip(f"Could not create Graphiti client, skipping integration test: {e}")

    test_group_id = "semantic-retriever-integration-test"
    test_content = "The quick brown fox jumps over the lazy dog."
    test_doc_name = "Test Document for Semantic Retriever"

    try:
        # 1. Setup: Clean old data and build indices
        await client.graphiti_instance.build_indices_and_constraints(delete_existing=True)
        
        # 2. Add a test document
        add_result = await client.graphiti_instance.add_episode(
            name=test_doc_name,
            episode_body=test_content,
            source_description="Integration test",
            group_id=test_group_id,
            reference_time=datetime.now(),
        )
        center_node_uuid = add_result.nodes[0].uuid

        # Add a small delay to allow for indexing
        await asyncio.sleep(5)

        # 3. Create the retriever
        retriever = GraphitiSemanticRetriever(
            client=client,
            group_ids=[test_group_id],
        )
        # 4. Retrieve documents
        query = "A fast, dark-colored canine leaps over a sleepy canine."
        docs = await retriever.ainvoke(query, center_node_uuid=center_node_uuid)

        # 5. Assertions
        assert len(docs) > 0
        assert isinstance(docs[0], Document)
        
        found_doc = None
        for doc in docs:
            if doc.metadata.get("name") == test_doc_name:
                found_doc = doc
                break
        
        assert found_doc is not None, "Test document not found in retrieval results"
        assert found_doc.page_content == test_content

    finally:
        # 6. Teardown: Clean up the test data
        if 'client' in locals():
            await client.graphiti_instance.driver.execute_query(
                "MATCH (n {group_id: $group_id}) DETACH DELETE n",
                group_id=test_group_id,
            )
            await client.close()