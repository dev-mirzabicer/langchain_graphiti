"""
Internal client for managing the Graphiti core instance.

This module provides the GraphitiClient class, which serves as the primary
wrapper around the graphiti_core.Graphiti object. It is designed to be
instantiated once with all necessary configurations and then passed to
various LangChain components like retrievers, tools, and stores.

Enhanced features:
- Comprehensive error handling with custom exception hierarchy
- Connection pooling and lifecycle management
- Health checks and monitoring capabilities
- Resource management and cleanup
- Async context manager support
- Configuration validation and defaults
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, Optional, Dict
from contextlib import asynccontextmanager
import weakref

from pydantic import BaseModel, Field, ConfigDict
from langsmith import traceable

# Import core Graphiti components for type hinting and instantiation
from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.embedder import EmbedderClient
from graphiti_core.llm_client import LLMClient
from graphiti_core.errors import GraphitiError

# Import custom exceptions and utilities
from .exceptions import (
    GraphitiClientError,
    GraphitiConnectionError,
    GraphitiConfigurationError,
    GraphitiOperationError,
)
from .utils import validate_config_dict, safe_sync_run

logger = logging.getLogger(__name__)


class GraphitiClient(BaseModel):
    """
    A client for interacting with the Graphiti knowledge graph system.

    This class serves as a wrapper around the core Graphiti instance,
    managing its configuration, lifecycle, and providing a single point of access
    for other LangChain components with enhanced error handling and monitoring.

    Features:
    - Automatic connection management and health monitoring
    - Resource cleanup and lifecycle management  
    - Enhanced error handling with custom exceptions
    - Configuration validation and defaults
    - Async context manager support
    - Connection pooling support

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
        
        # Option 3: As async context manager
        async with GraphitiClient.from_connections(...) as client:
            # Use client
            pass
        ```
    """

    graphiti_instance: Graphiti = Field(
        ...,
        description="The core, fully-configured Graphiti instance.",
    )
    
    # Configuration options
    auto_health_check: bool = Field(
        default=True,
        description="Whether to automatically perform health checks on startup.",
    )
    
    connection_timeout: float = Field(
        default=30.0,
        description="Connection timeout in seconds.",
    )
    
    retry_attempts: int = Field(
        default=3,
        description="Number of retry attempts for failed operations.",
    )
    
    # Internal state
    _is_closed: bool = Field(default=False, exclude=True)
    _health_status: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    _health_checked: bool = Field(default=False, exclude=True)
    _instance_registry: Optional[weakref.WeakSet] = Field(default=None, exclude=True)
    
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
        
        # Initialize instance registry for cleanup tracking
        if not hasattr(GraphitiClient, '_global_instance_registry'):
            GraphitiClient._global_instance_registry = weakref.WeakSet()
        GraphitiClient._global_instance_registry.add(self)
        self._instance_registry = GraphitiClient._global_instance_registry
        
        # Health check will be performed lazily on first request
        logger.debug("GraphitiClient initialized. Health check will be performed on first request.")

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
    ) -> "GraphitiClient":
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

        Raises:
            GraphitiConfigurationError: If configuration is invalid.
            GraphitiConnectionError: If connection fails.
        """
        try:
            # Validate components
            cls._validate_components(driver, llm_client, embedder, cross_encoder)
            
            # Validate kwargs using utility function
            valid_graphiti_kwargs = [
                "store_raw_episode_content", "max_coroutines", "embedding_dimension",
                "search_config", "logging_config"
            ]
            validated_kwargs = validate_config_dict(
                kwargs, 
                required_keys=[], 
                optional_keys=valid_graphiti_kwargs
            )
            
            graphiti_kwargs = {
                "store_raw_episode_content": store_raw_episode_content,
                **validated_kwargs
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
            
        except GraphitiError as e:
            raise GraphitiConnectionError(f"Failed to initialize Graphiti core: {e}") from e
        except Exception as e:
            if isinstance(e, (GraphitiClientError, GraphitiError)):
                raise
            raise GraphitiConfigurationError(f"Failed to create GraphitiClient: {e}") from e

    @staticmethod
    def _validate_components(
        driver: GraphDriver,
        llm_client: LLMClient,
        embedder: EmbedderClient,
        cross_encoder: CrossEncoderClient,
    ) -> None:
        """Validate that all required components are properly configured."""
        if not driver:
            raise GraphitiConfigurationError("Graph driver is required")
        if not llm_client:
            raise GraphitiConfigurationError("LLM client is required")
        if not embedder:
            raise GraphitiConfigurationError("Embedder client is required")
        if not cross_encoder:
            raise GraphitiConfigurationError("Cross-encoder client is required")

    @traceable
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check on the Graphiti instance.
        
        Uses lazy initialization - runs the first time it's requested and caches
        the result. Re-runs if the previous status was not healthy or if explicitly
        requested after a failure.
        
        Returns:
            A dictionary with detailed health status information.
        """
        if self._is_closed:
            return {
                "status": "closed",
                "message": "Client has been closed",
            }
        
        # If not checked yet, or if status is not healthy, run a full check
        if not self._health_checked or self._health_status.get("status") not in ["healthy", "degraded"]:
            await self._perform_health_check()
            self._health_checked = True
            
        return self._health_status.copy()

    async def _perform_health_check(self) -> None:
        """Internal method to perform the actual health check."""
        health_info = {
            "status": "unknown",
            "timestamp": None,
            "components": {},
            "errors": [],
        }
        
        try:
            import time
            health_info["timestamp"] = time.time()
            
            # Test database connectivity
            try:
                await self._test_database_connection()
                health_info["components"]["database"] = {
                    "status": "healthy",
                    "type": type(self.graphiti_instance.driver).__name__
                }
            except Exception as e:
                health_info["components"]["database"] = {"status": "unhealthy", "error": str(e)}
                health_info["errors"].append(f"Database: {e}")

            # Test LLM client with configuration check
            try:
                llm_client = self.graphiti_instance.llm_client
                # Check if the client has a proper configuration
                if hasattr(llm_client, 'config') and hasattr(llm_client.config, 'api_key') and llm_client.config.api_key:
                    health_info["components"]["llm"] = {
                        "status": "configured",
                        "type": type(llm_client).__name__
                    }
                else:
                    health_info["components"]["llm"] = {
                        "status": "misconfigured",
                        "error": "API key might be missing"
                    }
                    health_info["errors"].append("LLM: Misconfigured")
            except Exception as e:
                health_info["components"]["llm"] = {"status": "unavailable", "error": str(e)}
                health_info["errors"].append(f"LLM: {e}")

            # Test embedder with a lightweight functional test
            try:
                # Attempt a lightweight embedding test
                test_embedding = await self.graphiti_instance.embedder.create("test")
                if test_embedding and len(test_embedding) > 0:
                    health_info["components"]["embedder"] = {
                        "status": "healthy",
                        "type": type(self.graphiti_instance.embedder).__name__,
                        "embedding_dim": len(test_embedding)
                    }
                else:
                    health_info["components"]["embedder"] = {
                        "status": "unhealthy",
                        "error": "Empty embedding returned"
                    }
                    health_info["errors"].append("Embedder: Empty embedding returned")
            except Exception as e:
                health_info["components"]["embedder"] = {"status": "unhealthy", "error": str(e)}
                health_info["errors"].append(f"Embedder: {e}")

            # Test cross-encoder
            try:
                cross_encoder = self.graphiti_instance.cross_encoder
                health_info["components"]["cross_encoder"] = {
                    "status": "available",
                    "type": type(cross_encoder).__name__
                }
            except Exception as e:
                health_info["components"]["cross_encoder"] = {"status": "unavailable", "error": str(e)}
                health_info["errors"].append(f"Cross-encoder: {e}")

            # Determine overall status
            if not health_info["errors"]:
                health_info["status"] = "healthy"
            elif len(health_info["errors"]) <= 1:
                health_info["status"] = "degraded"
            else:
                health_info["status"] = "unhealthy"
                
            # Cache health status
            self._health_status = health_info
            
        except Exception as e:
            error_info = {
                "status": "error",
                "timestamp": time.time() if 'time' in locals() else None,
                "error": str(e),
                "error_type": type(e).__name__,
            }
            self._health_status = error_info

    async def _test_database_connection(self) -> None:
        """Test database connection with a simple query."""
        try:
            # Use a simple query that should work on any graph database
            driver = self.graphiti_instance.driver
            session = driver.session()
            try:
                # Simple test query - check if we can connect
                await session.run("RETURN 1 as test")
            finally:
                await session.close()
        except Exception as e:
            raise GraphitiConnectionError(
                f"Database connection test failed: {e}",
                connection_details={
                    "driver_type": type(self.graphiti_instance.driver).__name__,
                    "error_type": type(e).__name__
                }
            ) from e

    async def execute_with_retry(
        self, 
        operation: callable, 
        *args, 
        max_retries: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Execute an operation with retry logic.
        
        Args:
            operation: The async operation to execute
            *args: Arguments for the operation
            max_retries: Maximum retry attempts (uses instance default if None)
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation
            
        Raises:
            GraphitiOperationError: If all retry attempts fail
        """
        max_retries = max_retries or self.retry_attempts
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.warning(f"Operation failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Operation failed after {max_retries + 1} attempts: {e}")
        
        raise GraphitiOperationError(
            f"Operation failed after {max_retries + 1} attempts: {last_error}",
            operation_details={
                "operation_name": getattr(operation, '__name__', str(operation)),
                "attempts": max_retries + 1,
                "last_error_type": type(last_error).__name__
            }
        ) from last_error

    # --- Component Access Methods ---

    def get_driver(self) -> GraphDriver:
        """Get the underlying graph driver."""
        if self._is_closed:
            raise GraphitiClientError("Client has been closed")
        return self.graphiti_instance.driver

    def get_llm_client(self) -> LLMClient:
        """Get the underlying LLM client."""
        if self._is_closed:
            raise GraphitiClientError("Client has been closed")
        return self.graphiti_instance.llm_client

    def get_embedder(self) -> EmbedderClient:
        """Get the underlying embedder client."""
        if self._is_closed:
            raise GraphitiClientError("Client has been closed")
        return self.graphiti_instance.embedder

    def get_cross_encoder(self) -> CrossEncoderClient:
        """Get the underlying cross-encoder client."""
        if self._is_closed:
            raise GraphitiClientError("Client has been closed")
        return self.graphiti_instance.cross_encoder

    def get_health_status(self) -> Dict[str, Any]:
        """Get the last known health status."""
        return self._health_status.copy()

    def is_healthy(self) -> bool:
        """Check if the client is in a healthy state."""
        return self._health_status.get("status") == "healthy"

    # --- Lifecycle Management ---

    async def close(self) -> None:
        """
        Close the client and clean up resources.
        
        This method should be called when the client is no longer needed.
        """
        if self._is_closed:
            return
            
        try:
            # Close the underlying Graphiti instance
            if hasattr(self.graphiti_instance, 'close'):
                await self.graphiti_instance.close()
                
            # Mark as closed
            self._is_closed = True
            
            # Clear health status
            self._health_status = {"status": "closed"}
            
            logger.info("GraphitiClient closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing GraphitiClient: {e}")
            raise GraphitiClientError(f"Failed to close client: {e}") from e

    async def __aenter__(self) -> "GraphitiClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    def __del__(self):
        """Cleanup on garbage collection."""
        if not self._is_closed:
            logger.warning("GraphitiClient was not explicitly closed, cleaning up")
            # We can't call async close() from __del__, so just mark as closed
            self._is_closed = True

    @classmethod
    async def close_all_instances(cls) -> None:
        """Close all active GraphitiClient instances."""
        if hasattr(cls, '_global_instance_registry'):
            instances = list(cls._global_instance_registry)
            for instance in instances:
                try:
                    await instance.close()
                except Exception as e:
                    logger.error(f"Error closing instance: {e}")

    def __repr__(self) -> str:
        """String representation of the client."""
        status = "closed" if self._is_closed else "open"
        return (
            f"GraphitiClient({status}, "
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
    Convenience function to create a GraphitiClient with sensible defaults.

    This function simplifies the setup process by allowing you to either pass
    pre-configured clients or letting it create default clients (OpenAI and Neo4j)
    based on environment variables.

    Environment Variables for default clients:
    - NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD (for Neo4j driver)
    - OPENAI_API_KEY (for OpenAI clients)

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
        
    Raises:
        GraphitiConfigurationError: If required environment variables are missing
                                   or configuration is invalid.
    """
    import os
    
    try:
        if driver is None:
            try:
                from graphiti_core.driver.neo4j_driver import Neo4jDriver
                uri = os.getenv("NEO4J_URI")
                user = os.getenv("NEO4J_USER")
                password = os.getenv("NEO4J_PASSWORD")
                
                if not all([uri, user, password]):
                    raise GraphitiConfigurationError(
                        "NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set in environment "
                        "variables if a driver is not provided.",
                        invalid_config={"NEO4J_URI": bool(uri), "NEO4J_USER": bool(user), "NEO4J_PASSWORD": bool(password)}
                    )
                driver = Neo4jDriver(uri=uri, user=user, password=password)
            except ImportError as e:
                raise GraphitiConfigurationError(f"Failed to import Neo4jDriver: {e}") from e

        if llm_client is None:
            try:
                from graphiti_core.llm_client import OpenAIClient
                if not os.getenv("OPENAI_API_KEY"):
                    raise GraphitiConfigurationError(
                        "OPENAI_API_KEY must be set in environment variables if llm_client is not provided.",
                        invalid_config={"OPENAI_API_KEY": False}
                    )
                llm_client = OpenAIClient()
            except ImportError as e:
                raise GraphitiConfigurationError(f"Failed to import OpenAIClient: {e}") from e

        if embedder is None:
            try:
                from graphiti_core.embedder import OpenAIEmbedder
                if not os.getenv("OPENAI_API_KEY"):
                    raise GraphitiConfigurationError(
                        "OPENAI_API_KEY must be set in environment variables if embedder is not provided.",
                        invalid_config={"OPENAI_API_KEY": False}
                    )
                embedder = OpenAIEmbedder()
            except ImportError as e:
                raise GraphitiConfigurationError(f"Failed to import OpenAIEmbedder: {e}") from e

        if cross_encoder is None:
            try:
                from graphiti_core.cross_encoder import OpenAIRerankerClient
                if not os.getenv("OPENAI_API_KEY"):
                    raise GraphitiConfigurationError(
                        "OPENAI_API_KEY must be set in environment variables if cross_encoder is not provided.",
                        invalid_config={"OPENAI_API_KEY": False}
                    )
                cross_encoder = OpenAIRerankerClient()
            except ImportError as e:
                raise GraphitiConfigurationError(f"Failed to import OpenAIRerankerClient: {e}") from e

        return GraphitiClient.from_connections(
            driver=driver,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
            **kwargs,
        )
        
    except Exception as e:
        if isinstance(e, GraphitiClientError):
            raise
        raise GraphitiConfigurationError(f"Failed to create GraphitiClient: {e}") from e


@asynccontextmanager
async def graphiti_client_context(**kwargs) -> AsyncGenerator[GraphitiClient, None]:
    """
    Async context manager for GraphitiClient that ensures proper cleanup.
    
    Args:
        **kwargs: Arguments to pass to create_graphiti_client()
        
    Yields:
        GraphitiClient: A configured client instance
        
    Example:
        ```python
        async with graphiti_client_context() as client:
            # Use client
            results = await client.graphiti_instance.search_("query")
        # Client is automatically closed
        ```
    """
    client = create_graphiti_client(**kwargs)
    try:
        yield client
    finally:
        await client.close()