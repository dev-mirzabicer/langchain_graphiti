"""Integration tests for Graphiti retrievers."""

import pytest
import os
from datetime import datetime
from langchain_core.documents import Document
from langchain_graphiti import (
    GraphitiClientFactory,
    LLMProvider,
    DriverProvider,
    GeminiConfig,
    Neo4jConfig,
)
from langchain_graphiti.retrievers import GraphitiRetriever


def create_test_client():
    """Helper to create a client for testing from environment variables."""
    gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("TEST_GEMINI_API_KEY")
    neo4j_uri = os.getenv("NEO4J_URI") or os.getenv("TEST_URI")
    neo4j_user = os.getenv("NEO4J_USER") or os.getenv("TEST_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD") or os.getenv("TEST_PASSWORD")

    if not all([gemini_api_key, neo4j_uri, neo4j_user, neo4j_password]):
        raise ConnectionError("Missing env vars for integration tests: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, GEMINI_API_KEY")

    llm_config = GeminiConfig(api_key=gemini_api_key)
    driver_config = Neo4jConfig(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)

    return GraphitiClientFactory.create(
        llm_provider=LLMProvider.GEMINI,
        driver_provider=DriverProvider.NEO4J,
        llm_config=llm_config,
        driver_config=driver_config,
    )


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
        client = create_test_client()
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