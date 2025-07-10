"""
LangChain integration for Graphiti knowledge graph system.

This package provides seamless integration between Graphiti's powerful knowledge
graph capabilities and the LangChain ecosystem, enabling sophisticated RAG
applications, agentic systems, and knowledge management workflows.

Core Components:
- GraphitiClient: Manages connections to Graphiti with enhanced error handling
- GraphitiRetriever: Advanced graph-aware retrieval with proper score handling
- GraphitiVectorStore: Complete VectorStore interface for compatibility
- Tools: Comprehensive set of tools for agent use (now with sync/async support)

Enhanced Features in this version:
- Full sync/async support for all components
- Comprehensive error handling and health monitoring  
- Complete VectorStore interface implementation
- Advanced retrievers with caching and semantic search
- New tools for direct graph manipulation
- Proper streaming support
- Connection lifecycle management

Example:
    ```python
    import os
    from langchain_graphiti import (
        GraphitiClient, 
        GraphitiRetriever, 
        GraphitiVectorStore,
        AddEpisodeTool,
        SearchGraphTool,
        create_agent_tools
    )
    
    # Initialize client with automatic defaults
    client = create_graphiti_client()
    
    # Or with specific components
    from graphiti_core.driver.neo4j_driver import Neo4jDriver
    from graphiti_core.llm_client import OpenAIClient
    from graphiti_core.embedder import OpenAIEmbedder
    from graphiti_core.cross_encoder import OpenAIRerankerClient

    client = GraphitiClient.from_connections(
        driver=Neo4jDriver(
            uri=os.environ["NEO4J_URI"],
            user=os.environ["NEO4J_USER"], 
            password=os.environ["NEO4J_PASSWORD"]
        ),
        llm_client=OpenAIClient(),
        embedder=OpenAIEmbedder(),
        cross_encoder=OpenAIRerankerClient(),
    )

    # Use as enhanced retriever with proper score handling
    retriever = GraphitiRetriever(client=client)
    docs = await retriever.aget_relevant_documents("What is machine learning?")

    # Use as complete VectorStore
    vectorstore = GraphitiVectorStore(client=client)
    docs = vectorstore.similarity_search("machine learning", k=5)
    
    # Use comprehensive tool set for agents
    tools = create_agent_tools(client)
    
    # Use with async context management
    async with GraphitiClient.from_connections(...) as client:
        # Client automatically cleaned up
        pass
    ```
"""

# Version information
__version__ = "0.2.0"  # Updated for enhanced features
__author__ = "Mirza Bicer"

# Core client and utilities
from ._client import (
    GraphitiClient, 
    create_graphiti_client,
    graphiti_client_context,
    # Exceptions
    GraphitiClientError,
    GraphitiConnectionError,
    GraphitiConfigurationError,
    GraphitiOperationError,
)

# Enhanced retrievers
from .retrievers import (
    GraphitiRetriever, 
    GraphitiSemanticRetriever,
    GraphitiCachedRetriever,
)

# Comprehensive tool set
from .tools import (
    # Core tools
    AddEpisodeTool,
    SearchGraphTool, 
    BuildCommunitiesTool,
    RemoveEpisodeTool,
    
    # New advanced tools
    AddTripletTool,
    GetNodesAndEdgesByEpisodeTool,
    BuildIndicesAndConstraintsTool,
    
    # Tool factory functions
    create_agent_tools,
    create_basic_agent_tools,
    create_advanced_agent_tools,
)

# Complete VectorStore implementation
from .vectorstores import GraphitiVectorStore

# Public API - all available components
__all__ = [
    # Core client and utilities
    "GraphitiClient",
    "create_graphiti_client", 
    "graphiti_client_context",
    
    # Client exceptions
    "GraphitiClientError",
    "GraphitiConnectionError", 
    "GraphitiConfigurationError",
    "GraphitiOperationError",
    
    # Enhanced retrievers
    "GraphitiRetriever", 
    "GraphitiSemanticRetriever",
    "GraphitiCachedRetriever",
    
    # Comprehensive tool set
    "AddEpisodeTool",
    "SearchGraphTool", 
    "BuildCommunitiesTool",
    "RemoveEpisodeTool",
    "AddTripletTool",
    "GetNodesAndEdgesByEpisodeTool", 
    "BuildIndicesAndConstraintsTool",
    
    # Tool utilities
    "create_agent_tools",
    "create_basic_agent_tools", 
    "create_advanced_agent_tools",
    
    # Complete VectorStore
    "GraphitiVectorStore",
    
    # Version info
    "__version__",
]


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
        "vectorstore_features": [
            "Complete VectorStore interface implementation",
            "Maximum Marginal Relevance (MMR) search",
            "Document management (add/update/delete)",
            "Bulk operations support", 
            "Factory methods for easy creation",
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
