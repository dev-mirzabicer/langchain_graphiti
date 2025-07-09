"""
LangChain integration for Graphiti knowledge graph system.

This package provides seamless integration between Graphiti's powerful knowledge
graph capabilities and the LangChain ecosystem, enabling sophisticated RAG
applications, agentic systems, and knowledge management workflows.

Core Components:
- GraphitiClient: Manages connections to Graphiti
- GraphitiRetriever: Advanced graph-aware retrieval
- GraphitiVectorStore: VectorStore interface for compatibility
- Tools: AddEpisodeTool, SearchGraphTool, etc. for agent use

Example:
    ```python
    import os
    from langchain_graphiti import (
        GraphitiClient, 
        GraphitiRetriever, 
        AddEpisodeTool,
        SearchGraphTool
    )
    from graphiti_core.driver.neo4j_driver import Neo4jDriver
    from graphiti_core.llm_client import OpenAIClient
    from graphiti_core.embedder import OpenAIEmbedder
    from graphiti_core.cross_encoder import OpenAIRerankerClient

    # Initialize client
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

    # Use as retriever
    retriever = GraphitiRetriever(client=client)
    docs = await retriever.aget_relevant_documents("What is machine learning?")

    # Use as tools for agents
    tools = [
        AddEpisodeTool(client=client),
        SearchGraphTool(client=client),
    ]
    ```
"""

from ._client import GraphitiClient, create_graphiti_client
from .retrievers import GraphitiRetriever, GraphitiSemanticRetriever  
from .tools import (
    AddEpisodeTool,
    SearchGraphTool,
    BuildCommunitiesTool,
    RemoveEpisodeTool,
    create_agent_tools
)
from .vectorstores import GraphitiVectorStore

# Version information
__version__ = "0.1.0"
__author__ = "Mirza Bicer"

# Public API
__all__ = [
    # Core client
    "GraphitiClient",
    "create_graphiti_client",
    
    # Retrievers
    "GraphitiRetriever", 
    "GraphitiSemanticRetriever",
    
    # Tools for agents
    "AddEpisodeTool",
    "SearchGraphTool", 
    "BuildCommunitiesTool",
    "RemoveEpisodeTool",
    "create_agent_tools",
    # Vector store compatibility
    "GraphitiVectorStore",
    
    # Version info
    "__version__",
]


def get_version() -> str:
    """Get the current version of langchain-graphiti."""
    return __version__


