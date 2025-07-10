"""Integration tests for Graphiti retrievers."""

import pytest
from datetime import datetime
from langchain_core.documents import Document
from langchain_graphiti import create_graphiti_client
from langchain_graphiti.retrievers import GraphitiRetriever


@pytest.mark.asyncio
async def test_graphiti_retriever_integration():
    """
    Test the GraphitiRetriever against a live database.
    
    This test performs the following steps:
    1. Creates a GraphitiClient to connect to the database.
    2. Cleans up any old data and sets up indices.
    3. Adds a test document (episode) to the graph.
    4. Creates a GraphitiRetriever instance.
    5. Uses the retriever to search for the test document.
    6. Asserts that the correct document is returned.
    7. Cleans up the test data.
    """
    try:
        client = create_graphiti_client()
    except Exception as e:
        pytest.skip(f"Could not create Graphiti client, skipping integration test: {e}")

    test_group_id = "retriever-integration-test"
    test_content = "LangChain Graphiti is a cool new library for knowledge graphs."
    test_doc_name = "Test Document for Retriever"

    try:
        # 1. Setup: Clean old data and build indices
        await client.graphiti_instance.build_indices_and_constraints(delete_existing=True)
        
        # 2. Add a test document
        await client.graphiti_instance.add_episode(
            name=test_doc_name,
            episode_body=test_content,
            source_description="Integration test",
            group_id=test_group_id,
            reference_time=datetime.now(),
        )

        # 3. Create the retriever
        retriever = GraphitiRetriever(
            client=client,
            group_ids=[test_group_id],
        )

        # 4. Retrieve documents
        query = "knowledge graphs"
        docs = await retriever.ainvoke(query)

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