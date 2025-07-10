"""
LangChain tools for interacting with a Graphiti knowledge graph.

This module provides a comprehensive set of tools that agents can use
to read from and write to a Graphiti instance. These tools are designed
to support both synchronous and asynchronous execution, follow modern 
LangChain best practices, and provide rich error handling and observability.

Key Components:
- AddEpisodeTool: Add new information to the knowledge graph
- SearchGraphTool: Query the knowledge graph for relevant information
- BuildCommunitiesTool: Trigger community detection and organization
- RemoveEpisodeTool: Remove specific episodes from the graph
- AddTripletTool: Directly add node-edge-node relationships
- GetNodesAndEdgesByEpisodeTool: Retrieve graph data for specific episodes
- BuildIndicesAndConstraintsTool: Set up database indices and constraints
"""

from __future__ import annotations

import asyncio
from typing import List, Optional, Type, Annotated, Dict, Any, Tuple

from langchain_core.tools import BaseTool, InjectedToolArg
from langsmith import traceable
from pydantic import BaseModel, Field

from ._client import GraphitiClient
from .exceptions import GraphitiToolError, GraphitiClientError
from .utils import safe_sync_run, format_graph_results, require_client
from graphiti_core.nodes import EpisodeType, EntityNode
from graphiti_core.edges import EntityEdge
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.search.search_helpers import search_results_to_context_string
from graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
)
from graphiti_core.errors import (
    GroupIdValidationError, 
    EntityTypeValidationError,
    GraphitiError,
)
from graphiti_core.helpers import semaphore_gather


# --- Base Tool Class ---


class GraphitiBaseTool(BaseTool):
    """Base class for Graphiti tools to handle sync/async execution."""
    
    client: Annotated[
        GraphitiClient,
        InjectedToolArg(
            description="The GraphitiClient instance to use for graph operations.",
        ),
    ]

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """Generic synchronous wrapper for the async method."""
        try:
            return safe_sync_run(self._arun(*args, **kwargs))
        except Exception as e:
            error_msg = f"Failed to execute tool '{self.name}': {e}"
            raise GraphitiToolError(error_msg, tool_name=self.name) from e

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        """Override this method in subclasses with the actual async logic."""
        raise NotImplementedError("Subclasses must implement _arun method")


# --- AddEpisodeTool ---


class AddEpisodeSchema(BaseModel):
    """Input schema for the AddEpisodeTool."""

    name: str = Field(
        ...,
        description="A descriptive name for the episode or event being recorded.",
    )
    episode_body: str = Field(
        ...,
        description=(
            "The main content of the episode, such as a conversation transcript, "
            "a document's text, or a JSON data string."
        ),
    )
    source_description: str = Field(
        ...,
        description=(
            "A brief description of the information's origin "
            "(e.g., 'User conversation on 2024-08-15', 'Q2 2025 Financial Report')."
        ),
    )
    group_id: str = Field(
        default="",
        description=(
            "An optional ID to partition the graph. All related information should "
            "share the same group_id for multi-tenant applications."
        ),
    )
    source: str = Field(
        default="message",
        description=(
            "The type of the episode content. Must be one of 'message', 'text', "
            "or 'json'."
        ),
    )
    update_communities: bool = Field(
        default=False,
        description="Whether to update community structures after adding this episode.",
    )


class AddEpisodeTool(GraphitiBaseTool):
    """
    A tool to add a new episode (a piece of information) to the Graphiti
    knowledge graph. 
    
    Episodes are the fundamental unit of information in Graphiti - they represent
    discrete pieces of content that get processed into structured knowledge.
    When an episode is added, Graphiti extracts entities, relationships, and
    integrates them into the existing knowledge graph.
    """

    name: str = "add_episode_to_knowledge_graph"
    description: str = (
        "Adds a new piece of information (an episode) to the Graphiti knowledge graph. "
        "This is the primary way to teach the system new things. Use this to record "
        "new facts, events, conversations, or any structured information. The system "
        "will automatically extract entities and relationships from the content."
    )
    args_schema: Type[BaseModel] = AddEpisodeSchema

    @traceable
    @require_client
    async def _arun(
        self,
        name: str,
        episode_body: str,
        source_description: str,
        group_id: str = "",
        source: str = "message",
        update_communities: bool = False,
        **kwargs,
    ) -> str:
        """Use the tool asynchronously."""
        try:
            episode_source = EpisodeType.from_str(source)
        except (NotImplementedError, ValueError) as e:
            return (
                f"Error: Invalid source type '{source}'. Must be one of 'message', "
                f"'text', or 'json'. Details: {str(e)}"
            )

        try:
            results = await self.client.graphiti_instance.add_episode(
                name=name,
                episode_body=episode_body,
                source_description=source_description,
                reference_time=utc_now(),
                source=episode_source,
                group_id=group_id,
                update_communities=update_communities,
            )
            
            node_count = len(results.nodes)
            edge_count = len(results.edges)
            episode_uuid = results.episode.uuid
            
            return (
                f"✅ Successfully added episode '{episode_uuid}'. "
                f"Graph updated with {node_count} entities and {edge_count} relationships. "
                f"{'Communities were updated. ' if update_communities else ''}"
                f"The knowledge graph now contains this new information and can be queried."
            )
        except (GroupIdValidationError, EntityTypeValidationError) as e:
            # Catch specific validation errors from Graphiti
            return f"❌ Validation Error: {str(e)}. Please correct the input and try again."
        except GraphitiError as e:
            # Catch Graphiti core errors
            return f"❌ Graphiti Error: {str(e)}. Please check the episode content and try again."
        except Exception as e:
            # General fallback for other errors
            error_msg = (
                f"❌ Failed to add episode to knowledge graph. "
                f"Error: {type(e).__name__}: {str(e)}. "
                f"Please check the episode content and parameters and try again."
            )
            raise GraphitiToolError(error_msg, tool_name=self.name) from e


# --- SearchGraphTool ---


class SearchGraphSchema(BaseModel):
    """Input schema for the SearchGraphTool."""

    query: str = Field(
        ...,
        description="The natural language query to search for in the knowledge graph.",
    )
    group_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of group IDs to scope the search.",
    )
    max_results: int = Field(
        default=10,
        description="Maximum number of results to return (1-50).",
        ge=1,
        le=50,
    )


class SearchGraphTool(GraphitiBaseTool):
    """
    A tool to search the Graphiti knowledge graph for information relevant
    to a query. 
    
    This tool performs advanced hybrid search combining text matching,
    semantic similarity, and graph traversal to find the most relevant
    information. Results are returned as a structured summary suitable
    for use in generating responses.
    """

    name: str = "search_knowledge_graph"
    description: str = (
        "Searches the Graphiti knowledge graph for information relevant to a query. "
        "Uses advanced hybrid search (text + semantic + graph) to find the most "
        "relevant facts, entities, and relationships. Returns a condensed summary "
        "of findings that can be used to answer questions or provide context."
    )
    args_schema: Type[BaseModel] = SearchGraphSchema

    @traceable
    @require_client
    async def _arun(
        self, 
        query: str, 
        group_ids: Optional[List[str]] = None, 
        max_results: int = 10,
        **kwargs
    ) -> str:
        """Use the tool asynchronously."""
        if not query.strip():
            return "⚠️ Empty query provided. Please provide a specific question or topic to search for."
            
        try:
            # Create a search config with the specified limit
            search_config = COMBINED_HYBRID_SEARCH_CROSS_ENCODER
            search_config.limit = max_results
            
            search_results = await self.client.graphiti_instance.search_(
                query=query,
                config=search_config,
                group_ids=group_ids,
            )
            
            # Check if we found anything
            total_results = (
                len(search_results.edges) + 
                len(search_results.nodes) + 
                len(search_results.episodes) + 
                len(search_results.communities)
            )
            
            if total_results == 0:
                return (
                    f"🔍 No relevant information found in the knowledge graph for query: '{query}'. "
                    f"The knowledge graph may not contain information about this topic yet. "
                    f"Consider adding relevant information first using the add_episode tool."
                )

            # Generate context string using Graphiti's helper
            context_summary = search_results_to_context_string(search_results)
            
            # Add metadata about the search
            metadata = (
                f"\n\n📊 Search Results Summary:\n"
                f"• Found {len(search_results.edges)} relationships\n"
                f"• Found {len(search_results.nodes)} entities\n"
                f"• Found {len(search_results.episodes)} episodes\n"
                f"• Found {len(search_results.communities)} communities\n"
                f"• Query: '{query}'"
            )
            
            return context_summary + metadata
            
        except GraphitiError as e:
            return f"❌ Graphiti Error during search: {str(e)}. Please try rephrasing your query."
        except Exception as e:
            error_msg = (
                f"❌ Error occurred during knowledge graph search: {type(e).__name__}: {str(e)}. "
                f"Please try rephrasing your query or check the search parameters."
            )
            raise GraphitiToolError(error_msg, tool_name=self.name) from e


# --- BuildCommunitiesTool ---


class BuildCommunitiesSchema(BaseModel):
    """Input schema for the BuildCommunitiesTool."""

    group_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of group IDs to scope the community building.",
    )


class BuildCommunitiesTool(GraphitiBaseTool):
    """
    A tool to trigger community detection and organization in the Graphiti
    knowledge graph.
    
    Communities represent clusters of closely related entities and relationships
    in the graph. This tool analyzes the graph structure to identify and organize
    these communities, which can improve search relevance and knowledge organization.
    """

    name: str = "build_communities"
    description: str = (
        "Triggers community detection and organization in the knowledge graph. "
        "This analyzes the graph structure to identify clusters of related entities "
        "and relationships, improving knowledge organization and search relevance. "
        "Use this after adding significant amounts of new information."
    )
    args_schema: Type[BaseModel] = BuildCommunitiesSchema

    @traceable
    @require_client
    async def _arun(
        self, 
        group_ids: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Use the tool asynchronously."""
        try:
            await self.client.graphiti_instance.build_communities(group_ids=group_ids)
            
            scope_desc = f" for groups {group_ids}" if group_ids else " across the entire graph"
            return (
                f"✅ Successfully rebuilt communities{scope_desc}. "
                f"The knowledge graph organization has been updated to reflect "
                f"current entity relationships and clustering patterns. "
                f"This should improve search relevance and knowledge discovery."
            )
            
        except GraphitiError as e:
            return f"❌ Graphiti Error building communities: {str(e)}. This may be due to insufficient data in the graph."
        except Exception as e:
            error_msg = (
                f"❌ Failed to build communities: {type(e).__name__}: {str(e)}. "
                f"This may be due to insufficient data in the graph or processing issues."
            )
            raise GraphitiToolError(error_msg, tool_name=self.name) from e


# --- RemoveEpisodeTool ---


class RemoveEpisodeSchema(BaseModel):
    """Input schema for the RemoveEpisodeTool."""

    episode_uuids: List[str] = Field(
        ...,
        description="List of episode UUIDs to remove from the knowledge graph.",
    )


class RemoveEpisodeTool(GraphitiBaseTool):
    """
    A tool to remove specific episodes from the Graphiti knowledge graph.
    
    This tool allows for selective removal of information from the graph.
    When episodes are removed, associated entities and relationships that
    are no longer supported by other episodes may also be cleaned up.
    """

    name: str = "remove_episodes"
    description: str = (
        "Removes specific episodes from the knowledge graph by their UUIDs. "
        "This also cleans up any entities or relationships that are no longer "
        "supported by remaining episodes. Use with caution as this permanently "
        "removes information from the graph."
    )
    args_schema: Type[BaseModel] = RemoveEpisodeSchema

    @traceable
    @require_client
    async def _arun(
        self, 
        episode_uuids: List[str],
        **kwargs
    ) -> str:
        """Use the tool asynchronously with parallel processing."""
        if not episode_uuids:
            return "⚠️ No episode UUIDs provided. Please specify which episodes to remove."
            
        try:
            async def _remove_one(uuid: str) -> Tuple[str, Optional[str]]:
                """Helper to remove a single episode and return result."""
                try:
                    await self.client.graphiti_instance.remove_episode(uuid)
                    return uuid, None
                except Exception as e:
                    return uuid, str(e)

            # Use semaphore_gather for controlled concurrency
            results = await semaphore_gather(
                *[_remove_one(uuid) for uuid in episode_uuids],
                max_coroutines=self.client.graphiti_instance.max_coroutines
            )

            successful_removals = [uuid for uuid, err in results if err is None]
            failed_removals = [(uuid, err) for uuid, err in results if err is not None]
            
            # Build detailed response
            response_parts = []
            
            if successful_removals:
                response_parts.append(
                    f"✅ Successfully removed {len(successful_removals)} episodes: "
                    f"{', '.join(successful_removals[:3])}"
                    f"{'...' if len(successful_removals) > 3 else ''}"
                )
            
            if failed_removals:
                response_parts.append(
                    f"❌ Failed to remove {len(failed_removals)} episodes. "
                    f"First error: {failed_removals[0][1]}"
                )
                
            if successful_removals:
                response_parts.append(
                    "The knowledge graph has been updated and associated entities "
                    "and relationships have been cleaned up as needed."
                )
            
            return " ".join(response_parts)
            
        except Exception as e:
            error_msg = (
                f"❌ Error during episode removal: {type(e).__name__}: {str(e)}. "
                f"Please check the episode UUIDs and try again."
            )
            raise GraphitiToolError(error_msg, tool_name=self.name) from e


# --- AddTripletTool (NEW) ---


class AddTripletSchema(BaseModel):
    """Input schema for the AddTripletTool."""

    source_node_name: str = Field(
        ...,
        description="Name of the source entity node.",
    )
    source_node_labels: List[str] = Field(
        default=["Entity"],
        description="Labels for the source node (default: ['Entity']).",
    )
    edge_name: str = Field(
        ...,
        description="Name/type of the relationship edge.",
    )
    edge_fact: str = Field(
        ...,
        description="Descriptive fact about the relationship.",
    )
    target_node_name: str = Field(
        ...,
        description="Name of the target entity node.",
    )
    target_node_labels: List[str] = Field(
        default=["Entity"],
        description="Labels for the target node (default: ['Entity']).",
    )
    group_id: str = Field(
        default="",
        description="Optional group ID to partition the graph.",
    )


class AddTripletTool(GraphitiBaseTool):
    """
    A tool to directly add a triplet (source_node -> edge -> target_node) 
    to the Graphiti knowledge graph.
    
    This provides a way to add structured knowledge directly without going 
    through episode processing. Useful for adding facts, relationships, or 
    structured data when you know the exact graph structure you want.
    """

    name: str = "add_triplet"
    description: str = (
        "Directly adds a structured triplet (source_node -> relationship -> target_node) "
        "to the knowledge graph. Use this when you have explicit structured knowledge "
        "to add, such as facts, relationships, or when importing from structured data sources."
    )
    args_schema: Type[BaseModel] = AddTripletSchema

    @traceable
    @require_client
    async def _arun(
        self,
        source_node_name: str,
        source_node_labels: List[str],
        edge_name: str,
        edge_fact: str,
        target_node_name: str,
        target_node_labels: List[str],
        group_id: str = "",
        **kwargs,
    ) -> str:
        """Use the tool asynchronously."""
        try:
            # Create nodes and edge using Graphiti's data structures
            source_node = EntityNode(
                name=source_node_name,
                labels=source_node_labels,
                group_id=group_id,
                created_at=utc_now(),
            )
            
            target_node = EntityNode(
                name=target_node_name,
                labels=target_node_labels,
                group_id=group_id,
                created_at=utc_now(),
            )
            
            edge = EntityEdge(
                source_node_uuid=source_node.uuid,
                target_node_uuid=target_node.uuid,
                name=edge_name,
                fact=edge_fact,
                group_id=group_id,
                created_at=utc_now(),
                valid_at=utc_now(),
            )
            
            # Add the triplet using Graphiti's add_triplet method
            await self.client.graphiti_instance.add_triplet(source_node, edge, target_node)
            
            return (
                f"✅ Successfully added triplet: {source_node_name} -[{edge_name}]-> {target_node_name}. "
                f"Fact: {edge_fact}. The knowledge graph has been updated with this structured relationship."
            )
            
        except GraphitiError as e:
            return f"❌ Graphiti Error adding triplet: {str(e)}. Please check the node and edge parameters."
        except Exception as e:
            error_msg = (
                f"❌ Failed to add triplet: {type(e).__name__}: {str(e)}. "
                f"Please check the node and edge parameters and try again."
            )
            raise GraphitiToolError(error_msg, tool_name=self.name) from e


# --- GetNodesAndEdgesByEpisodeTool (NEW) ---


class GetNodesAndEdgesByEpisodeSchema(BaseModel):
    """Input schema for the GetNodesAndEdgesByEpisodeTool."""

    episode_uuids: List[str] = Field(
        ...,
        description="List of episode UUIDs to retrieve nodes and edges for.",
    )


class GetNodesAndEdgesByEpisodeTool(GraphitiBaseTool):
    """
    A tool to retrieve all nodes and edges associated with specific episodes.
    
    This tool allows you to see what structured knowledge (entities and relationships)
    was extracted from particular episodes, useful for understanding how episodes
    were processed or for debugging knowledge extraction.
    """

    name: str = "get_nodes_and_edges_by_episode"
    description: str = (
        "Retrieves all nodes (entities) and edges (relationships) that were extracted "
        "from specific episodes. Use this to understand what structured knowledge was "
        "derived from particular episodes or to debug knowledge extraction results."
    )
    args_schema: Type[BaseModel] = GetNodesAndEdgesByEpisodeSchema

    @traceable
    @require_client
    async def _arun(
        self, 
        episode_uuids: List[str],
        **kwargs
    ) -> str:
        """Use the tool asynchronously."""
        if not episode_uuids:
            return "⚠️ No episode UUIDs provided. Please specify which episodes to analyze."
            
        try:
            # Use Graphiti's get_nodes_and_edges_by_episode method
            search_results = await self.client.graphiti_instance.get_nodes_and_edges_by_episode(episode_uuids)
            
            # Use the utility function to format results
            formatted_results = format_graph_results(search_results)
            
            result_header = f"📊 Graph elements extracted from {len(episode_uuids)} episode(s):"
            return f"{result_header}\n{formatted_results}"
            
        except GraphitiError as e:
            return f"❌ Graphiti Error retrieving nodes and edges: {str(e)}. Please check the episode UUIDs."
        except Exception as e:
            error_msg = (
                f"❌ Error retrieving nodes and edges: {type(e).__name__}: {str(e)}. "
                f"Please check the episode UUIDs and try again."
            )
            raise GraphitiToolError(error_msg, tool_name=self.name) from e


# --- BuildIndicesAndConstraintsTool (NEW) ---


class BuildIndicesAndConstraintsSchema(BaseModel):
    """Input schema for the BuildIndicesAndConstraintsTool."""

    delete_existing: bool = Field(
        default=False,
        description="Whether to delete existing indices before creating new ones.",
    )


class BuildIndicesAndConstraintsTool(GraphitiBaseTool):
    """
    A tool to build database indices and constraints for optimal performance.
    
    This tool sets up the necessary database indices and constraints to optimize
    query performance for the knowledge graph. Should typically be run once during
    initial setup or when updating the database schema.
    """

    name: str = "build_indices_and_constraints"
    description: str = (
        "Builds database indices and constraints to optimize knowledge graph performance. "
        "This sets up the necessary database schema optimizations for efficient querying. "
        "Should typically be run once during initial setup or after schema changes."
    )
    args_schema: Type[BaseModel] = BuildIndicesAndConstraintsSchema

    @traceable
    @require_client
    async def _arun(
        self, 
        delete_existing: bool = False,
        **kwargs
    ) -> str:
        """Use the tool asynchronously."""
        try:
            await self.client.graphiti_instance.build_indices_and_constraints(delete_existing)
            
            action = "rebuilt" if delete_existing else "built"
            return (
                f"✅ Successfully {action} database indices and constraints. "
                f"The knowledge graph database is now optimized for efficient querying. "
                f"{'Existing indices were deleted first. ' if delete_existing else ''}"
                f"Query performance should be significantly improved."
            )
            
        except GraphitiError as e:
            return f"❌ Graphiti Error building indices: {str(e)}. This operation requires appropriate database permissions."
        except Exception as e:
            error_msg = (
                f"❌ Failed to build indices and constraints: {type(e).__name__}: {str(e)}. "
                f"This operation requires appropriate database permissions and may take time on large graphs."
            )
            raise GraphitiToolError(error_msg, tool_name=self.name) from e


def create_agent_tools(client: GraphitiClient) -> list:
    """
    Create a comprehensive set of Graphiti tools for agent use.
    
    Args:
        client: GraphitiClient instance
        
    Returns:
        List of configured tools ready for agent use
        
    Example:
        ```python
        client = create_graphiti_client(...)
        tools = create_agent_tools(client)
        
        # Use with LangGraph
        from langgraph.prebuilt import create_react_agent
        agent = create_react_agent(llm, tools)
        ```
    """
    return [
        AddEpisodeTool(client=client),
        SearchGraphTool(client=client),
        BuildCommunitiesTool(client=client),
        RemoveEpisodeTool(client=client),
        AddTripletTool(client=client),
        GetNodesAndEdgesByEpisodeTool(client=client),
        BuildIndicesAndConstraintsTool(client=client),
    ]


def create_basic_agent_tools(client: GraphitiClient) -> list:
    """
    Create a basic set of Graphiti tools for simple agent use cases.
    
    Args:
        client: GraphitiClient instance
        
    Returns:
        List of essential tools for basic knowledge graph operations
    """
    return [
        AddEpisodeTool(client=client),
        SearchGraphTool(client=client),
        BuildCommunitiesTool(client=client),
    ]


def create_advanced_agent_tools(client: GraphitiClient) -> list:
    """
    Create an advanced set of Graphiti tools for complex agent use cases.
    
    Args:
        client: GraphitiClient instance
        
    Returns:
        List of all available tools for advanced knowledge graph operations
    """
    return create_agent_tools(client)  # Returns all tools