"""
Internal client for managing the Graphiti core instance.

This module provides the GraphitiClient class, which serves as the primary
wrapper around the graphiti_core.Graphiti object. It is designed to be
instantiated once with all necessary configurations and then passed to
various LangChain components like retrievers, tools, and stores.

Design Rationale:
- Encapsulation: By wrapping the Graphiti instance, we create a clear
  separation between the core backend logic and the LangChain interface logic.
- Singleton Access: While not a strict singleton, this pattern encourages
  creating a single client instance that is shared, ensuring consistent
  access to the same graph database and client configurations.
- Ease of Use: The `from_connections` class method provides a user-friendly
  factory to simplify the setup process for users who haven't pre-configured
  a Graphiti instance.
- Modern Patterns: Uses modern Pydantic v2 and follows current LangChain
  best practices for configuration and validation.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, ConfigDict
from langsmith import traceable

# Import core Graphiti components for type hinting and instantiation
from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.embedder import EmbedderClient
from graphiti_core.llm_client import LLMClient


class GraphitiClient(BaseModel):
    """
    A client for interacting with the Graphiti knowledge graph system.

    This class serves as a wrapper around the core Graphiti instance,
    managing its configuration and providing a single point of access
    for other LangChain components.

    It is not intended to be used directly by end-users for graph operations,
    but rather to be instantiated and passed to other LangChain-Graphiti
    components like `GraphitiRetriever` and `AddEpisodeTool`.

    Example:
        ```python
        from langchain_graphiti import GraphitiClient
        from graphiti_core.driver.neo4j_driver import Neo4jDriver
        from graphiti_core.llm_client import OpenAIClient
        
        # Option 1: From pre-configured Graphiti instance
        graphiti = Graphiti(...)
        client = GraphitiClient(graphiti_instance=graphiti)
        
        # Option 2: From individual components
        client = GraphitiClient.from_connections(
            driver=Neo4jDriver(...),
            llm_client=OpenAIClient(),
            embedder=OpenAIEmbedder(),
            cross_encoder=OpenAIRerankerClient(),
        )
        ```
    """

    graphiti_instance: Graphiti = Field(
        ...,
        description="The core, fully-configured Graphiti instance.",
    )
    
    # Modern Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, graphiti_instance: Graphiti, **kwargs: Any):
        """
        Initializes the GraphitiClient with a pre-configured Graphiti instance.

        This is the primary constructor for advanced users who have already
        instantiated and configured their `graphiti_core.Graphiti` object.

        Args:
            graphiti_instance: A fully configured instance of the Graphiti class.
            **kwargs: Additional Pydantic model arguments.
        """
        super().__init__(graphiti_instance=graphiti_instance, **kwargs)

    @classmethod
    def from_connections(
        cls,
        *,
        driver: GraphDriver,
        llm_client: LLMClient,
        embedder: EmbedderClient,
        cross_encoder: CrossEncoderClient,
        store_raw_episode_content: bool = True,
        max_coroutines: Optional[int] = None,
        **kwargs: Any,
    ) -> GraphitiClient:
        """
        Creates a GraphitiClient from individual connection components.

        This is a convenience factory method for creating a Graphiti instance
        and wrapping it in a GraphitiClient, simplifying setup for users.

        Args:
            driver: A configured graph database driver (e.g., Neo4jDriver).
            llm_client: A configured LLM client (e.g., OpenAIClient).
            embedder: A configured embedder client (e.g., OpenAIEmbedder).
            cross_encoder: A configured cross-encoder client (e.g., OpenAIRerankerClient).
            store_raw_episode_content: Whether to store raw episode content.
                Defaults to True.
            max_coroutines: Maximum number of concurrent operations.
                If None, uses Graphiti's default.
            **kwargs: Additional arguments to pass to the Graphiti constructor.

        Returns:
            A new instance of GraphitiClient.

        Example:
            ```python
            import os
            from langchain_graphiti import GraphitiClient
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
                store_raw_episode_content=True,
            )
            ```
        """
        graphiti_kwargs = {
            "store_raw_episode_content": store_raw_episode_content,
            **kwargs
        }
        
        if max_coroutines is not None:
            graphiti_kwargs["max_coroutines"] = max_coroutines
            
        graphiti_instance = Graphiti(
            graph_driver=driver,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
            **graphiti_kwargs,
        )
        return cls(graphiti_instance=graphiti_instance)

    @traceable
    async def health_check(self) -> dict[str, Any]:
        """
        Perform a basic health check on the Graphiti instance.
        
        Returns:
            A dictionary with health status information.
        """
        try:
            # Test basic connectivity
            # This is a simple check - in a real implementation you might
            # want to test database connectivity, LLM availability, etc.
            health_info = {
                "status": "healthy",
                "driver_type": type(self.graphiti_instance.driver).__name__,
                "llm_client_type": type(self.graphiti_instance.llm_client).__name__,
                "embedder_type": type(self.graphiti_instance.embedder).__name__,
                "cross_encoder_type": type(self.graphiti_instance.cross_encoder).__name__,
            }
            
            # You could add more sophisticated health checks here
            # like testing database connectivity, checking LLM API availability, etc.
            
            return health_info
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "error_type": type(e).__name__,
            }

    def get_driver(self) -> GraphDriver:
        """Get the underlying graph driver."""
        return self.graphiti_instance.driver

    def get_llm_client(self) -> LLMClient:
        """Get the underlying LLM client."""
        return self.graphiti_instance.llm_client

    def get_embedder(self) -> EmbedderClient:
        """Get the underlying embedder client."""
        return self.graphiti_instance.embedder

    def get_cross_encoder(self) -> CrossEncoderClient:
        """Get the underlying cross-encoder client."""
        return self.graphiti_instance.cross_encoder

    def __repr__(self) -> str:
        """String representation of the client."""
        return (
            f"GraphitiClient("
            f"driver={type(self.graphiti_instance.driver).__name__}, "
            f"llm={type(self.graphiti_instance.llm_client).__name__}, "
            f"embedder={type(self.graphiti_instance.embedder).__name__}, "
            f"cross_encoder={type(self.graphiti_instance.cross_encoder).__name__}"
            f")"
        )
    

def create_graphiti_client(
    driver: Optional["GraphDriver"] = None,
    llm_client: Optional["LLMClient"] = None,
    embedder: Optional["EmbedderClient"] = None,
    cross_encoder: Optional["CrossEncoderClient"] = None,
    **kwargs,
) -> "GraphitiClient":
    """
    Convenience function to create a GraphitiClient.

    This function simplifies the setup process by allowing you to either pass
    pre-configured clients or letting it create default clients (OpenAI and Neo4j)
    based on environment variables.

    Environment Variables for default clients:
    - NEO4J_URI
    - NEO4J_USER
    - NEO4J_PASSWORD
    - OPENAI_API_KEY

    Args:
        driver: A pre-configured graph database driver. If None, a Neo4jDriver
                is created using environment variables.
        llm_client: A pre-configured LLM client. If None, an OpenAIClient is
                    created using environment variables.
        embedder: A pre-configured embedder client. If None, an OpenAIEmbedder
                  is created using environment variables.
        cross_encoder: A pre-configured cross-encoder. If None, an
                       OpenAIRerankerClient is created.
        **kwargs: Additional arguments for GraphitiClient.from_connections().

    Returns:
        A configured GraphitiClient instance.
    """
    import os
    from graphiti_core.driver.driver import GraphDriver
    from graphiti_core.llm_client.client import LLMClient
    from graphiti_core.embedder.client import EmbedderClient
    from graphiti_core.cross_encoder.client import CrossEncoderClient

    if driver is None:
        from graphiti_core.driver.neo4j_driver import Neo4jDriver
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        if not all([uri, user, password]):
            raise ValueError(
                "NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set in environment "
                "variables if a driver is not provided."
            )
        driver = Neo4jDriver(uri=uri, user=user, password=password)

    if llm_client is None:
        from graphiti_core.llm_client import OpenAIClient
        llm_client = OpenAIClient() # Uses OPENAI_API_KEY from env by default

    if embedder is None:
        from graphiti_core.embedder import OpenAIEmbedder
        embedder = OpenAIEmbedder() # Uses OPENAI_API_KEY from env by default

    if cross_encoder is None:
        from graphiti_core.cross_encoder import OpenAIRerankerClient
        cross_encoder = OpenAIRerankerClient() # Uses OPENAI_API_KEY from env by default

    return GraphitiClient.from_connections(
        driver=driver,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
        **kwargs,
    )