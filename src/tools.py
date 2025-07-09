"""
LangChain tools for interacting with a Graphiti knowledge graph.

This module provides a comprehensive set of tools that agents can use
to read from and write to a Graphiti instance. These tools are designed
to be asynchronous, follow modern LangChain best practices, and provide
rich error handling and observability.

Key Components:
- AddEpisodeTool: Add new information to the knowledge graph
- SearchGraphTool: Query the knowledge graph for relevant information
- BuildCommunitiesTool: Trigger community detection and organization
- RemoveEpisodeTool: Remove specific episodes from the graph
"""

from __future__ import annotations

from typing import List, Optional, Type, Annotated

from langchain_core.tools import BaseTool, InjectedToolArg
from langsmith import traceable
from pydantic import BaseModel, Field

from ._client import GraphitiClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.search.search_helpers import search_results_to_context_string
from graphiti_core.search.search_config_recipes import (
    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
)


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


class AddEpisodeTool(BaseTool):
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
    
    # Use InjectedToolArg for proper dependency injection
    client: Annotated[
        GraphitiClient,
        InjectedToolArg(
            description="The GraphitiClient instance to use for graph operations.",
        ),
    ]

    @traceable
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
            
            # Provide rich feedback about what was added
            node_count = len(results.nodes)
            edge_count = len(results.edges)
            episode_uuid = results.episode.uuid
            
            return (
                f"✅ Successfully added episode '{episode_uuid}'. "
                f"Graph updated with {node_count} entities and {edge_count} relationships. "
                f"{'Communities were updated. ' if update_communities else ''}"
                f"The knowledge graph now contains this new information and can be queried."
            )
        except Exception as e:
            # Provide helpful error information for debugging
            return (
                f"❌ Failed to add episode to knowledge graph. "
                f"Error: {type(e).__name__}: {str(e)}. "
                f"Please check the episode content format and try again."
            )


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


class SearchGraphTool(BaseTool):
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
    
    client: Annotated[
        GraphitiClient,
        InjectedToolArg(
            description="The GraphitiClient instance to use for graph operations.",
        ),
    ]

    @traceable
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
            
        except Exception as e:
            return (
                f"❌ Error occurred during knowledge graph search: {type(e).__name__}: {str(e)}. "
                f"Please try rephrasing your query or check the search parameters."
            )


# --- BuildCommunitiesTool ---


class BuildCommunitiesSchema(BaseModel):
    """Input schema for the BuildCommunitiesTool."""

    group_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of group IDs to scope the community building.",
    )


class BuildCommunitiesTool(BaseTool):
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
    
    client: Annotated[
        GraphitiClient,
        InjectedToolArg(
            description="The GraphitiClient instance to use for graph operations.",
        ),
    ]

    @traceable
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
            
        except Exception as e:
            return (
                f"❌ Failed to build communities: {type(e).__name__}: {str(e)}. "
                f"This may be due to insufficient data in the graph or processing issues."
            )


# --- RemoveEpisodeTool ---


class RemoveEpisodeSchema(BaseModel):
    """Input schema for the RemoveEpisodeTool."""

    episode_uuids: List[str] = Field(
        ...,
        description="List of episode UUIDs to remove from the knowledge graph.",
    )


class RemoveEpisodeTool(BaseTool):
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
    
    client: Annotated[
        GraphitiClient,
        InjectedToolArg(
            description="The GraphitiClient instance to use for graph operations.",
        ),
    ]

    @traceable
    async def _arun(
        self, 
        episode_uuids: List[str],
        **kwargs
    ) -> str:
        """Use the tool asynchronously."""
        if not episode_uuids:
            return "⚠️ No episode UUIDs provided. Please specify which episodes to remove."
            
        try:
            successful_removals = []
            failed_removals = []
            
            for episode_uuid in episode_uuids:
                try:
                    await self.client.graphiti_instance.remove_episode(episode_uuid)
                    successful_removals.append(episode_uuid)
                except Exception as e:
                    failed_removals.append((episode_uuid, str(e)))
            
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
            return (
                f"❌ Error during episode removal: {type(e).__name__}: {str(e)}. "
                f"Please check the episode UUIDs and try again."
            )
