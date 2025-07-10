"""
LangChain integration for Graphiti knowledge graph system.

This package provides seamless integration between Graphiti's powerful knowledge
graph capabilities and the LangChain ecosystem, enabling sophisticated RAG
applications, agentic systems, and knowledge management workflows.
"""

# Version information
from ._version import __version__

# Core Client and Utilities
from ._client import (
    GraphitiClient,
    GraphitiClientFactory,
    graphiti_client_context,
)
from .config import (
    LLMProvider,
    DriverProvider,
    OpenAIConfig,
    AzureOpenAIConfig,
    GeminiConfig,
    AnthropicConfig,
    GroqConfig,
    Neo4jConfig,
    FalkorDBConfig,
)

# Custom Exceptions
from .exceptions import (
    GraphitiClientError,
    GraphitiConnectionError,
    GraphitiConfigurationError,
    GraphitiOperationError,
    GraphitiRetrieverError,
    GraphitiToolError,
)

# Retrievers
from .retrievers import (
    GraphitiRetriever,
    GraphitiSemanticRetriever,
    GraphitiCachedRetriever,
)


# Tools
from .tools import (
    AddEpisodeTool,
    SearchGraphTool,
    BuildCommunitiesTool,
    RemoveEpisodeTool,
    AddTripletTool,
    GetNodesAndEdgesByEpisodeTool,
    BuildIndicesAndConstraintsTool,
    create_agent_tools,
    create_basic_agent_tools,
    create_advanced_agent_tools,
)

# Organized __all__ for better discoverability
__all__ = [
    # Version
    "__version__",

    # Client
    "GraphitiClient",
    "GraphitiClientFactory",
    "graphiti_client_context",

    # Config
    "LLMProvider",
    "DriverProvider",
    "OpenAIConfig",
    "AzureOpenAIConfig",
    "GeminiConfig",
    "AnthropicConfig",
    "GroqConfig",
    "Neo4jConfig",
    "FalkorDBConfig",

    # Exceptions
    "GraphitiClientError",
    "GraphitiConnectionError",
    "GraphitiConfigurationError",
    "GraphitiOperationError",
    "GraphitiRetrieverError",
    "GraphitiToolError",

    # Retrievers
    "GraphitiRetriever",
    "GraphitiSemanticRetriever",
    "GraphitiCachedRetriever",


    # Tools
    "AddEpisodeTool",
    "SearchGraphTool",
    "BuildCommunitiesTool",
    "RemoveEpisodeTool",
    "AddTripletTool",
    "GetNodesAndEdgesByEpisodeTool",
    "BuildIndicesAndConstraintsTool",
    "create_agent_tools",
    "create_basic_agent_tools",
    "create_advanced_agent_tools",
]

# Utility functions for backward compatibility and convenience
def get_version() -> str:
    """Get the current version of langchain-graphiti."""
    return __version__

def list_available_tools() -> list[str]:
    """Get a list of all available tool classes."""
    return [
        "AddEpisodeTool",
        "SearchGraphTool",
        "BuildCommunitiesTool",
        "RemoveEpisodeTool",
        "AddTripletTool",
        "GetNodesAndEdgesByEpisodeTool",
        "BuildIndicesAndConstraintsTool",
    ]

def get_feature_summary() -> dict[str, list[str]]:
    """Get a summary of available features by category."""
    return {
        "client_features": [
            "Enhanced error handling with custom exceptions",
            "Health monitoring and connection management",
            "Async context manager support",
            "Automatic retry logic",
            "Resource lifecycle management",
        ],
        "retriever_features": [
            "Proper score extraction from Graphiti search",
            "Enhanced streaming with async support",
            "Semantic search with graph topology",
            "Caching support for performance",
            "Score threshold filtering",
        ],
        "tools_features": [
            "Full sync/async support for all tools",
            "Direct graph manipulation (add_triplet)",
            "Episode content retrieval",
            "Database optimization (indices/constraints)",
            "Comprehensive error handling",
            "Tool factory functions for easy setup",
        ],
    }


