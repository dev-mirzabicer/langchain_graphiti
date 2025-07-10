"""Unit tests for Graphiti tools."""

import pytest
from typing import Type, List
from unittest.mock import MagicMock, AsyncMock

from datetime import datetime
from langchain_core.tools import BaseTool
from langchain_graphiti.tools import (
    AddEpisodeTool,
    SearchGraphTool,
    BuildCommunitiesTool,
    RemoveEpisodeTool,
    AddTripletTool,
    GetNodesAndEdgesByEpisodeTool,
    BuildIndicesAndConstraintsTool,
    create_basic_agent_tools,
    create_advanced_agent_tools,
    AddEpisodeSchema,
    SearchGraphSchema,
    GraphitiToolError,
)
from langchain_graphiti._client import GraphitiClient
from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType
from graphiti_core.edges import EntityEdge
from graphiti_core.search.search_config import SearchResults
from graphiti_core.errors import GraphitiError


@pytest.fixture
def mock_graphiti_client() -> GraphitiClient:
    """Fixture for a mocked GraphitiClient."""
    client = MagicMock(spec=GraphitiClient)
    mock_instance = MagicMock()

    # Mock methods that will be used across tools
    mock_instance.add_episode = AsyncMock(
        return_value=MagicMock(
            episode=MagicMock(uuid="new-episode-uuid"),
            nodes=[MagicMock()],
            edges=[MagicMock()],
        )
    )
    mock_instance.search_ = AsyncMock(
        return_value=SearchResults(episodes=[], nodes=[], edges=[], communities=[])
    )
    mock_instance.build_communities = AsyncMock()
    mock_instance.remove_episode = AsyncMock()
    mock_instance.add_triplet = AsyncMock()
    mock_instance.get_nodes_and_edges_by_episode = AsyncMock(
        return_value=SearchResults(nodes=[], edges=[], episodes=[], communities=[])
    )
    mock_instance.build_indices_and_constraints = AsyncMock()
    mock_instance.max_coroutines = 10

    client.graphiti_instance = mock_instance
    return client


class TestAddEpisodeTool:
    """Unit tests for the AddEpisodeTool."""

    @pytest.mark.asyncio
    async def test_add_episode_success(self, mock_graphiti_client: GraphitiClient):
        """Test successful episode addition."""
        tool = AddEpisodeTool(client=mock_graphiti_client)
        result = await tool._arun(
            name="Test", episode_body="Test body", source_description="test"
        )
        assert "✅ Successfully added episode" in result
        mock_graphiti_client.graphiti_instance.add_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_episode_invalid_source(
        self, mock_graphiti_client: GraphitiClient
    ):
        """Test tool with an invalid source type."""
        tool = AddEpisodeTool(client=mock_graphiti_client)
        result = await tool._arun(
            name="Test",
            episode_body="Test body",
            source_description="test",
            source="invalid_source",
        )
        assert "Error: Invalid source type 'invalid_source'" in result

    @pytest.mark.asyncio
    async def test_add_episode_graphiti_error(
        self, mock_graphiti_client: GraphitiClient
    ):
        """Test handling of GraphitiError during episode addition."""
        mock_graphiti_client.graphiti_instance.add_episode.side_effect = GraphitiError(
            "DB connection failed"
        )
        tool = AddEpisodeTool(client=mock_graphiti_client)
        with pytest.raises(GraphitiToolError, match="Graphiti Error: DB connection failed"):
            await tool._arun(
                name="Test", episode_body="Test body", source_description="test"
            )

    @pytest.mark.asyncio
    async def test_add_episode_client_closed(self, mock_graphiti_client: GraphitiClient):
        """Test that the tool raises an error if the client is closed."""
        mock_graphiti_client._is_closed = True
        tool = AddEpisodeTool(client=mock_graphiti_client)
        with pytest.raises(ValueError, match="GraphitiClient has been closed"):
            await tool._arun(
                name="Test", episode_body="Test body", source_description="test"
            )


class TestSearchGraphTool:
    """Unit tests for the SearchGraphTool."""

    @pytest.mark.asyncio
    async def test_search_with_results(self, mock_graphiti_client: GraphitiClient):
        """Test search when results are found."""
        mock_graphiti_client.graphiti_instance.search_.return_value = SearchResults(
            nodes=[EntityNode(name="Test Node", group_id="test")],
            edges=[],
            episodes=[],
            communities=[],
        )
        tool = SearchGraphTool(client=mock_graphiti_client)
        result = await tool._arun(query="test query")
        assert "📊 Search Results Summary:" in result
        assert "• Found 1 entities" in result

    @pytest.mark.asyncio
    async def test_search_no_results(self, mock_graphiti_client: GraphitiClient):
        """Test search when no results are found."""
        tool = SearchGraphTool(client=mock_graphiti_client)
        result = await tool._arun(query="another query")
        assert "🔍 No relevant information found" in result

    @pytest.mark.asyncio
    async def test_search_empty_query(self, mock_graphiti_client: GraphitiClient):
        """Test search with an empty query string."""
        tool = SearchGraphTool(client=mock_graphiti_client)
        result = await tool._arun(query="  ")
        assert "⚠️ Empty query provided" in result

    @pytest.mark.asyncio
    async def test_search_graphiti_error(self, mock_graphiti_client: GraphitiClient):
        """Test that the tool raises GraphitiToolError on search failure."""
        mock_graphiti_client.graphiti_instance.search_.side_effect = GraphitiError(
            "Search index not found"
        )
        tool = SearchGraphTool(client=mock_graphiti_client)
        with pytest.raises(GraphitiToolError, match="Graphiti Error during search: Search index not found"):
            await tool._arun(query="test query")


class TestBuildCommunitiesTool:
    """Unit tests for the BuildCommunitiesTool."""

    @pytest.mark.asyncio
    async def test_build_communities_success(
        self, mock_graphiti_client: GraphitiClient
    ):
        """Test successful community building."""
        tool = BuildCommunitiesTool(client=mock_graphiti_client)
        result = await tool._arun()
        assert "✅ Successfully rebuilt communities" in result
        mock_graphiti_client.graphiti_instance.build_communities.assert_called_once()

    @pytest.mark.asyncio
    async def test_build_communities_with_group_ids(
        self, mock_graphiti_client: GraphitiClient
    ):
        """Test community building with group IDs."""
        tool = BuildCommunitiesTool(client=mock_graphiti_client)
        group_ids = ["group1", "group2"]
        result = await tool._arun(group_ids=group_ids)
        assert f"for groups {group_ids}" in result
        mock_graphiti_client.graphiti_instance.build_communities.assert_called_with(
            group_ids=group_ids
        )

    @pytest.mark.asyncio
    async def test_build_communities_graphiti_error(
        self, mock_graphiti_client: GraphitiClient
    ):
        """Test that the tool raises GraphitiToolError on build failure."""
        mock_graphiti_client.graphiti_instance.build_communities.side_effect = GraphitiError(
            "Community detection algorithm failed"
        )
        tool = BuildCommunitiesTool(client=mock_graphiti_client)
        with pytest.raises(GraphitiToolError, match="Graphiti Error building communities: Community detection algorithm failed"):
            await tool._arun()


class TestRemoveEpisodeTool:
    """Unit tests for the RemoveEpisodeTool."""

    @pytest.mark.asyncio
    async def test_remove_episode_success(self, mock_graphiti_client: GraphitiClient):
        """Test successful episode removal."""
        tool = RemoveEpisodeTool(client=mock_graphiti_client)
        uuids = ["uuid1", "uuid2"]
        result = await tool._arun(episode_uuids=uuids)
        assert "✅ Successfully removed 2 episodes" in result
        assert mock_graphiti_client.graphiti_instance.remove_episode.call_count == 2

    @pytest.mark.asyncio
    async def test_remove_episode_partial_failure(
        self, mock_graphiti_client: GraphitiClient
    ):
        """Test partial failure during episode removal."""
        mock_graphiti_client.graphiti_instance.remove_episode.side_effect = [
            None,
            GraphitiError("Cannot find uuid3"),
        ]
        tool = RemoveEpisodeTool(client=mock_graphiti_client)
        uuids = ["uuid1", "uuid3"]
        result = await tool._arun(episode_uuids=uuids)
        assert "✅ Successfully removed 1 episodes" in result
        assert "❌ Failed to remove 1 episodes" in result
        assert "Cannot find uuid3" in result

    @pytest.mark.asyncio
    async def test_remove_episode_no_uuids(self, mock_graphiti_client: GraphitiClient):
        """Test calling remove tool with no UUIDs."""
        tool = RemoveEpisodeTool(client=mock_graphiti_client)
        result = await tool._arun(episode_uuids=[])
        assert "⚠️ No episode UUIDs provided" in result

    @pytest.mark.asyncio
    async def test_remove_episode_graphiti_error(
        self, mock_graphiti_client: GraphitiClient
    ):
        """Test that the tool returns an error message on removal failure."""
        mock_graphiti_client.graphiti_instance.remove_episode.side_effect = GraphitiError(
            "DB constraint violation"
        )
        tool = RemoveEpisodeTool(client=mock_graphiti_client)
        result = await tool._arun(episode_uuids=["uuid1"])
        assert "❌ Failed to remove 1 episodes" in result
        assert "DB constraint violation" in result


class TestAddTripletTool:
    """Unit tests for the AddTripletTool."""

    @pytest.mark.asyncio
    async def test_add_triplet_success(self, mock_graphiti_client: GraphitiClient):
        """Test successful triplet addition."""
        tool = AddTripletTool(client=mock_graphiti_client)
        result = await tool._arun(
            source_node_name="Company A",
            edge_name="ACQUIRED",
            edge_fact="Acquired for $1B",
            target_node_name="Company B",
            source_node_labels=["Company"],
            target_node_labels=["Company"],
        )
        assert "✅ Successfully added triplet" in result
        mock_graphiti_client.graphiti_instance.add_triplet.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_triplet_graphiti_error(self, mock_graphiti_client: GraphitiClient):
        """Test that the tool raises GraphitiToolError on triplet addition failure."""
        mock_graphiti_client.graphiti_instance.add_triplet.side_effect = GraphitiError(
            "Invalid node labels"
        )
        tool = AddTripletTool(client=mock_graphiti_client)
        with pytest.raises(GraphitiToolError, match="Graphiti Error adding triplet: Invalid node labels"):
            await tool._arun(
                source_node_name="Company A",
                edge_name="ACQUIRED",
                edge_fact="Acquired for $1B",
                target_node_name="Company B",
                source_node_labels=["Company"],
                target_node_labels=["Company"],
            )


class TestGetNodesAndEdgesByEpisodeTool:
    """Unit tests for the GetNodesAndEdgesByEpisodeTool."""

    @pytest.mark.asyncio
    async def test_get_nodes_and_edges_success(
        self, mock_graphiti_client: GraphitiClient
    ):
        """Test successful retrieval of nodes and edges."""
        mock_graphiti_client.graphiti_instance.get_nodes_and_edges_by_episode.return_value = SearchResults(
            nodes=[EntityNode(name="Test Node", group_id="test")],
            edges=[
                EntityEdge(
                    name="RELATED_TO",
                    fact="is related",
                    source_node_uuid="n1",
                    target_node_uuid="n2",
                    group_id="test",
                    created_at=datetime.now(),
                )
            ],
            episodes=[],
            communities=[],
        )
        tool = GetNodesAndEdgesByEpisodeTool(client=mock_graphiti_client)
        result = await tool._arun(episode_uuids=["uuid1"])
        assert "📊 Graph elements extracted" in result
        assert "🔵 Found 1 entities" in result
        assert "• Test Node: No summary..." in result
        assert "📊 Found 1 relationships" in result
        assert "• RELATED_TO: is related..." in result

    @pytest.mark.asyncio
    async def test_get_nodes_and_edges_graphiti_error(
        self, mock_graphiti_client: GraphitiClient
    ):
        """Test that the tool raises GraphitiToolError on retrieval failure."""
        mock_graphiti_client.graphiti_instance.get_nodes_and_edges_by_episode.side_effect = GraphitiError(
            "Episode not found"
        )
        tool = GetNodesAndEdgesByEpisodeTool(client=mock_graphiti_client)
        with pytest.raises(GraphitiToolError, match="Graphiti Error retrieving nodes and edges: Episode not found"):
            await tool._arun(episode_uuids=["uuid1"])


class TestBuildIndicesAndConstraintsTool:
    """Unit tests for the BuildIndicesAndConstraintsTool."""

    @pytest.mark.asyncio
    async def test_build_indices_success(self, mock_graphiti_client: GraphitiClient):
        """Test successful index building."""
        tool = BuildIndicesAndConstraintsTool(client=mock_graphiti_client)
        result = await tool._arun()
        assert "✅ Successfully built database indices" in result
        mock_graphiti_client.graphiti_instance.build_indices_and_constraints.assert_called_with(
            False
        )

    @pytest.mark.asyncio
    async def test_build_indices_with_delete(
        self, mock_graphiti_client: GraphitiClient
    ):
        """Test index building with delete_existing=True."""
        tool = BuildIndicesAndConstraintsTool(client=mock_graphiti_client)
        result = await tool._arun(delete_existing=True)
        assert "✅ Successfully rebuilt database indices" in result
        mock_graphiti_client.graphiti_instance.build_indices_and_constraints.assert_called_with(
            True
        )

    @pytest.mark.asyncio
    async def test_build_indices_graphiti_error(
        self, mock_graphiti_client: GraphitiClient
    ):
        """Test that the tool raises GraphitiToolError on build failure."""
        mock_graphiti_client.graphiti_instance.build_indices_and_constraints.side_effect = GraphitiError(
            "Insufficient permissions"
        )
        tool = BuildIndicesAndConstraintsTool(client=mock_graphiti_client)
        with pytest.raises(GraphitiToolError, match="Graphiti Error building indices: Insufficient permissions"):
            await tool._arun()


class TestToolFactories:
    """Unit tests for the tool factory functions."""

    def test_create_basic_agent_tools(self, mock_graphiti_client: GraphitiClient):
        """Test the create_basic_agent_tools factory function."""
        tools = create_basic_agent_tools(client=mock_graphiti_client)
        assert isinstance(tools, list)
        assert len(tools) == 3
        tool_types = [type(tool) for tool in tools]
        assert AddEpisodeTool in tool_types
        assert SearchGraphTool in tool_types
        assert BuildCommunitiesTool in tool_types

    def test_create_advanced_agent_tools(self, mock_graphiti_client: GraphitiClient):
        """Test the create_advanced_agent_tools factory function."""
        tools = create_advanced_agent_tools(client=mock_graphiti_client)
        assert isinstance(tools, list)
        assert len(tools) == 7
        tool_types = [type(tool) for tool in tools]
        assert AddEpisodeTool in tool_types
        assert SearchGraphTool in tool_types
        assert BuildCommunitiesTool in tool_types
        assert RemoveEpisodeTool in tool_types
        assert AddTripletTool in tool_types
        assert GetNodesAndEdgesByEpisodeTool in tool_types
        assert BuildIndicesAndConstraintsTool in tool_types