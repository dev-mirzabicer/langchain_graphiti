"""
End-to-end integration test for the entire Graphiti tool workflow.

This test simulates a real-world usage pattern by calling the tools in a
logical sequence to ensure they work together correctly with a live database.
"""

import pytest
from typing import Generator

from langchain_graphiti import create_graphiti_client
from langchain_graphiti.tools import (
    AddEpisodeTool,
    SearchGraphTool,
    RemoveEpisodeTool,
    AddTripletTool,
    GetNodesAndEdgesByEpisodeTool,
    BuildCommunitiesTool,
    BuildIndicesAndConstraintsTool,
)
from langchain_graphiti._client import GraphitiClient
from langchain_graphiti.utils import safe_sync_run

TEST_GROUP_ID = "tool-workflow-integration-test"

@pytest.fixture(scope="module")
def client_for_workflow() -> Generator[GraphitiClient, None, None]:
    """Module-level fixture for the workflow test."""
    try:
        client = create_graphiti_client()
    except Exception as e:
        pytest.skip(f"Could not create Graphiti client, skipping workflow test: {e}")
    
    yield client
    
    # Final teardown: clean up all test data
    try:
        safe_sync_run(
            client.graphiti_instance.driver.execute_query(
                "MATCH (n {group_id: $group_id}) DETACH DELETE n",
                group_id=TEST_GROUP_ID,
            )
        )
        safe_sync_run(client.close())
    except Exception:
        pass # Suppress errors during teardown

@pytest.mark.integration
@pytest.mark.asyncio
class TestToolWorkflow:
    """A class to test the end-to-end tool workflow."""

    async def test_full_workflow(self, client_for_workflow: GraphitiClient):
        """
        Tests the full data lifecycle: setup -> create -> verify -> maintain -> cleanup.
        """
        # --- 1. Setup ---
        build_indices_tool = BuildIndicesAndConstraintsTool(client=client_for_workflow)
        result = await build_indices_tool._arun(delete_existing=True)
        assert "✅ Successfully rebuilt" in result

        # --- 2. Create ---
        add_episode_tool = AddEpisodeTool(client=client_for_workflow)
        add_triplet_tool = AddTripletTool(client=client_for_workflow)

        # Add an episode
        episode_params = {
            "name": "Workflow Test Episode",
            "episode_body": "LangChain Graphiti provides powerful tools for knowledge management.",
            "source_description": "Workflow test",
            "group_id": TEST_GROUP_ID,
        }
        add_result = await add_episode_tool._arun(**episode_params)
        assert "✅ Successfully added episode" in add_result
        episode_uuid = add_result.split("'")[1] # Extract UUID from the success message

        # Add a direct triplet
        triplet_params = {
            "source_node_name": "LangChain Graphiti",
            "edge_name": "HAS_FEATURE",
            "edge_fact": "It includes an end-to-end workflow test.",
            "target_node_name": "Workflow Test",
            "group_id": TEST_GROUP_ID,
            "source_node_labels": ["Software Library"],
            "target_node_labels": ["Test"],
        }
        triplet_result = await add_triplet_tool._arun(**triplet_params)
        assert "✅ Successfully added triplet" in triplet_result

        # --- 3. Verify ---
        search_tool = SearchGraphTool(client=client_for_workflow)
        get_nodes_tool = GetNodesAndEdgesByEpisodeTool(client=client_for_workflow)

        # Verify with search
        search_result = await search_tool._arun(query="knowledge management tools", group_ids=[TEST_GROUP_ID])
        assert "📊 Search Results Summary:" in search_result
        assert "LangChain Graphiti" in search_result

        # Verify with get_nodes_and_edges
        get_nodes_result = await get_nodes_tool._arun(episode_uuids=[episode_uuid])
        assert "📊 Graph elements extracted" in get_nodes_result
        assert "LangChain Graphiti" in get_nodes_result # Check if entity was extracted

        # --- 4. Maintain ---
        build_communities_tool = BuildCommunitiesTool(client=client_for_workflow)
        community_result = await build_communities_tool._arun(group_ids=[TEST_GROUP_ID])
        assert "✅ Successfully rebuilt communities" in community_result

        # --- 5. Cleanup ---
        remove_tool = RemoveEpisodeTool(client=client_for_workflow)
        remove_result = await remove_tool._arun(episode_uuids=[episode_uuid])
        assert "✅ Successfully removed 1 episodes" in remove_result

        # Verify removal with search
        search_after_delete = await search_tool._arun(query="knowledge management tools", group_ids=[TEST_GROUP_ID])
        # This is a soft check; depending on graph structure, related nodes might still be found.
        # A better check is that the original episode content is gone.
        assert "powerful tools for knowledge management" not in search_after_delete