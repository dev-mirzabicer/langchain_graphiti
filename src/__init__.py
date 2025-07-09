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

from ._client import GraphitiClient
from .retrievers import GraphitiRetriever, GraphitiSemanticRetriever  
from .tools import (
    AddEpisodeTool,
    SearchGraphTool,
    BuildCommunitiesTool,
    RemoveEpisodeTool,
)
from .vectorstores import GraphitiVectorStore

# Version information
__version__ = "0.1.0"
__author__ = "Mirza Bicer"

# Public API
__all__ = [
    # Core client
    "GraphitiClient",
    
    # Retrievers
    "GraphitiRetriever", 
    "GraphitiSemanticRetriever",
    
    # Tools for agents
    "AddEpisodeTool",
    "SearchGraphTool", 
    "BuildCommunitiesTool",
    "RemoveEpisodeTool",
    
    # Vector store compatibility
    "GraphitiVectorStore",
    
    # Version info
    "__version__",
]


def get_version() -> str:
    """Get the current version of langchain-graphiti."""
    return __version__


# Convenience functions for common setup patterns
def create_graphiti_client(
    neo4j_uri: str,
    neo4j_user: str, 
    neo4j_password: str,
    openai_api_key: str,
    **kwargs
) -> GraphitiClient:
    """
    Convenience function to create a GraphitiClient with common OpenAI + Neo4j setup.
    
    Args:
        neo4j_uri: Neo4j database URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password  
        openai_api_key: OpenAI API key
        **kwargs: Additional arguments for GraphitiClient.from_connections()
        
    Returns:
        Configured GraphitiClient instance
        
    Example:
        ```python
        client = create_graphiti_client(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password", 
            openai_api_key="sk-...",
        )
        ```
    """
    from graphiti_core.driver.neo4j_driver import Neo4jDriver
    from graphiti_core.llm_client import OpenAIClient
    from graphiti_core.embedder import OpenAIEmbedder
    from graphiti_core.cross_encoder import OpenAIRerankerClient
    
    return GraphitiClient.from_connections(
        driver=Neo4jDriver(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password
        ),
        llm_client=OpenAIClient(api_key=openai_api_key),
        embedder=OpenAIEmbedder(api_key=openai_api_key),
        cross_encoder=OpenAIRerankerClient(api_key=openai_api_key),
        **kwargs
    )


def create_agent_tools(client: GraphitiClient) -> list:
    """
    Create a standard set of Graphiti tools for agent use.
    
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
    ]