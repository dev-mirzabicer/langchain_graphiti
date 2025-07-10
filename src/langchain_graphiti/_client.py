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
from typing import Any, AsyncGenerator, Optional, Dict, Tuple, Type, Union, ClassVar, Callable
from contextlib import asynccontextmanager
import weakref
import importlib

from pydantic import BaseModel, Field, ConfigDict, PrivateAttr
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
from .utils import validate_config_dict
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
    
    database: Optional[str] = Field(
        default=None, description="The name of the database to connect to."
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
    _is_closed: bool = PrivateAttr(default=False)
    _health_status: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _health_checked: bool = PrivateAttr(default=False)
    
    # Class-level registry for all instances
    _global_instance_registry: ClassVar[weakref.WeakSet] = weakref.WeakSet()
    
    # Modern Pydantic v2 configuration
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

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
        GraphitiClient._global_instance_registry.add(self)
        
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
        database: Optional[str] = None,
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
            database: The name of the database to connect to.
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
            
            return cls(graphiti_instance=graphiti_instance, database=database)
            
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
        health_info: Dict[str, Any] = {
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
        if not self.database:
            raise GraphitiConfigurationError(
                "No database specified for GraphitiClient. "
                "Unable to perform database connection health check."
            )
        try:
            # Use a simple query that should work on any graph database
            driver = self.graphiti_instance.driver
            async with driver.session(database=self.database) as session:
                # Simple test query - check if we can connect
                await session.run("RETURN 1 as test")
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
        operation: Callable[..., Any],
        *args: Any,
        max_retries: Optional[int] = None,
        **kwargs: Any
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
                result = await operation(*args, **kwargs)
                return result
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
    

class GraphitiClientFactory:
    """
    Factory for creating GraphitiClient instances with various providers.
    
    This factory simplifies the process of configuring and instantiating
    Graphiti by dynamically loading the required clients based on user
    configuration. It supports conditional imports to avoid installing
    unnecessary dependencies.
    """

    @staticmethod
    def _import_class(module_path: str, class_name: str, extra: str) -> Type:
        """Dynamically import a class and handle ImportErrors."""
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except ImportError:
            raise ImportError(
                f"The '{class_name}' client requires the '{extra}' extra. "
                f"Please install it with: pip install langchain-graphiti[{extra}]"
            ) from None
        except AttributeError:
            raise ImportError(f"Class '{class_name}' not found in module '{module_path}'.")

    @classmethod
    def create(
        cls,
        llm_provider: LLMProvider,
        driver_provider: DriverProvider,
        llm_config: Union[OpenAIConfig, AzureOpenAIConfig, GeminiConfig, AnthropicConfig, GroqConfig],
        driver_config: Union[Neo4jConfig, FalkorDBConfig],
        **kwargs: Any,
    ) -> "GraphitiClient":
        """
        Create a GraphitiClient with the specified providers and configurations.
        
        Args:
            llm_provider: The LLM provider to use.
            driver_provider: The graph database driver to use.
            llm_config: The configuration for the LLM provider.
            driver_config: The configuration for the graph driver.
            **kwargs: Additional arguments for GraphitiClient.from_connections().
            
        Returns:
            A configured GraphitiClient instance.
        """
        driver = cls._create_driver(driver_provider, driver_config)
        llm_client, embedder, cross_encoder = cls._create_llm_services(llm_provider, llm_config)

        return GraphitiClient.from_connections(
            driver=driver,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
            **kwargs,
        )

    @classmethod
    def _create_driver(cls, provider: DriverProvider, config: Union[Neo4jConfig, FalkorDBConfig]) -> GraphDriver:
        """Create a graph driver instance."""
        if provider == DriverProvider.NEO4J:
            Neo4jDriver = cls._import_class("graphiti_core.driver.neo4j_driver", "Neo4jDriver", "neo4j")
            return Neo4jDriver(**config.model_dump())
        elif provider == DriverProvider.FALKORDB:
            FalkorDBDriver = cls._import_class("graphiti_core.driver.falkordb_driver", "FalkorDBDriver", "falkordb")
            return FalkorDBDriver(**config.model_dump())
        raise GraphitiConfigurationError(f"Unsupported driver provider: {provider}")

    @classmethod
    def _create_llm_services(
        cls, provider: LLMProvider, config: Any
    ) -> Tuple[LLMClient, EmbedderClient, CrossEncoderClient]:
        """Create instances of LLM, embedder, and cross-encoder clients."""
        if provider in [LLMProvider.OPENAI, LLMProvider.OLLAMA]:
            return cls._create_openai_compatible_clients(config)
        elif provider == LLMProvider.AZURE_OPENAI:
            return cls._create_azure_openai_clients(config)
        elif provider == LLMProvider.GEMINI:
            return cls._create_gemini_clients(config)
        elif provider == LLMProvider.ANTHROPIC:
            return cls._create_anthropic_clients(config)
        elif provider == LLMProvider.GROQ:
            return cls._create_groq_clients(config)
        raise GraphitiConfigurationError(f"Unsupported LLM provider: {provider}")

    @classmethod
    def _create_openai_compatible_clients(cls, config: OpenAIConfig):
        """Create clients for OpenAI and compatible services like Ollama."""
        OpenAIClient = cls._import_class("graphiti_core.llm_client.openai_client", "OpenAIClient", "openai")
        OpenAIEmbedder = cls._import_class("graphiti_core.embedder.openai", "OpenAIEmbedder", "openai")
        OpenAIRerankerClient = cls._import_class("graphiti_core.cross_encoder.openai_reranker_client", "OpenAIRerankerClient", "openai")
        LLMConfig = cls._import_class("graphiti_core.llm_client.config", "LLMConfig", "openai")
        OpenAIEmbedderConfig = cls._import_class("graphiti_core.embedder.openai", "OpenAIEmbedderConfig", "openai")

        llm_config = LLMConfig(api_key=config.api_key, model=config.model, small_model=config.small_model, base_url=config.base_url)
        llm_client = OpenAIClient(config=llm_config)
        embedder_config = OpenAIEmbedderConfig(
            api_key=config.api_key,
            embedding_model=config.embedding_model,
            embedding_dim=config.embedding_dim,
            base_url=config.base_url,
        )
        embedder = OpenAIEmbedder(config=embedder_config)
        cross_encoder = OpenAIRerankerClient(client=llm_client, config=llm_config)
        return llm_client, embedder, cross_encoder

    @classmethod
    def _create_azure_openai_clients(cls, config: AzureOpenAIConfig):
        """Create clients for Azure OpenAI."""
        # This requires more complex setup with multiple client instances
        raise NotImplementedError("Azure OpenAI client creation is not yet implemented in this factory.")

    @classmethod
    def _create_gemini_clients(cls, config: GeminiConfig):
        """Create clients for Google Gemini."""
        GeminiClient = cls._import_class("graphiti_core.llm_client.gemini_client", "GeminiClient", "gemini")
        GeminiEmbedder = cls._import_class("graphiti_core.embedder.gemini", "GeminiEmbedder", "gemini")
        GeminiRerankerClient = cls._import_class("graphiti_core.cross_encoder.gemini_reranker_client", "GeminiRerankerClient", "gemini")
        LLMConfig = cls._import_class("graphiti_core.llm_client.gemini_client", "LLMConfig", "gemini")
        GeminiEmbedderConfig = cls._import_class("graphiti_core.embedder.gemini", "GeminiEmbedderConfig", "gemini")

        llm_config = LLMConfig(api_key=config.api_key, model=config.model)
        llm_client = GeminiClient(config=llm_config)
        embedder_config = GeminiEmbedderConfig(api_key=config.api_key, embedding_model=config.embedding_model)
        embedder = GeminiEmbedder(config=embedder_config)
        reranker_config = LLMConfig(api_key=config.api_key, model=config.reranker_model)
        cross_encoder = GeminiRerankerClient(config=reranker_config)
        return llm_client, embedder, cross_encoder

    @classmethod
    def _create_anthropic_clients(cls, config: AnthropicConfig):
        """Create clients for Anthropic."""
        # Anthropic often doesn't have a dedicated embedder or reranker in graphiti-core
        # We might need to pair it with another provider's embedder
        raise NotImplementedError("Anthropic client creation is not yet fully implemented in this factory.")

    @classmethod
    def _create_groq_clients(cls, config: GroqConfig):
        """Create clients for Groq."""
        raise NotImplementedError("Groq client creation is not yet implemented in this factory.")


@asynccontextmanager
async def graphiti_client_context(
    llm_provider: LLMProvider,
    driver_provider: DriverProvider,
    llm_config: Any,
    driver_config: Any,
    **kwargs
) -> AsyncGenerator[GraphitiClient, None]:
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
    client = GraphitiClientFactory.create(
        llm_provider=llm_provider,
        driver_provider=driver_provider,
        llm_config=llm_config,
        driver_config=driver_config,
        **kwargs,
    )
    try:
        yield client
    finally:
        await client.close()