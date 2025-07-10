"""Integration tests for Graphiti tools."""

import pytest
import os
from langchain_graphiti import (
    GraphitiClientFactory,
    LLMProvider,
    DriverProvider,
    GeminiConfig,
    Neo4jConfig,
)
from langchain_graphiti.tools import AddEpisodeTool, SearchGraphTool


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
async def test_add_and_search_tools_integration():
    """
    Test the AddEpisodeTool and SearchGraphTool in an integrated workflow.
    
    This test performs the following steps:
    1. Creates a GraphitiClient to connect to the database.
    2. Cleans up any old data and sets up indices.
    3. Instantiates the AddEpisodeTool and SearchGraphTool.
    4. Uses the AddEpisodeTool to add a new episode.
    5. Uses the SearchGraphTool to find the newly added episode.
    6. Asserts that the search results contain the expected information.
    7. Cleans up the test data.
    """
    try:
        client = create_test_client()
    except Exception as e:
        pytest.skip(f"Could not create Graphiti client, skipping integration test: {e}")

    test_group_id = "tools-integration-test"
    episode_name = "Project Phoenix Status Update"
    episode_body = "The project is on track for a Q4 launch. Key dependencies are secure."

    try:
        # 1. Setup
        await client.graphiti_instance.build_indices_and_constraints(delete_existing=True)

        # 2. Instantiate tools
        add_tool = AddEpisodeTool(client=client)
        search_tool = SearchGraphTool(client=client)

        # 3. Add an episode
        add_params = {
            "name": episode_name,
            "episode_body": episode_body,
            "source_description": "Integration test",
            "group_id": test_group_id,
        }
        add_result = await add_tool._arun(**add_params)
        assert "✅ Successfully added episode" in add_result

        # 4. Search for the episode
        search_params = {
            "query": "Project Phoenix",
            "group_ids": [test_group_id],
        }
        search_result = await search_tool._arun(**search_params)

        # 5. Assertions
        assert "📊 Search Results Summary:" in search_result
        assert f"Query: '{search_params['query']}'" in search_result
        # Check that the content of the episode is in the search result summary
        assert "The project is on track for a Q4 launch" in search_result

    finally:
        # 6. Teardown
        if 'client' in locals():
            await client.graphiti_instance.driver.execute_query(
                "MATCH (n {group_id: $group_id}) DETACH DELETE n",
                group_id=test_group_id,
            )
            await client.close()