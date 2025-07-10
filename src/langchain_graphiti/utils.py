"""Utility functions for langchain-graphiti."""

import asyncio
import logging
from typing import Any, Callable, Optional, TypeVar, Union
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


def require_client(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to ensure GraphitiClient is available and healthy."""
    @wraps(func)
    async def async_wrapper(self, *args, **kwargs):
        if not hasattr(self, 'client') or self.client is None:
            raise ValueError(f"{self.__class__.__name__} requires a GraphitiClient")
        if hasattr(self.client, '_is_closed') and self.client._is_closed:
            raise ValueError("GraphitiClient has been closed")
        return await func(self, *args, **kwargs)
    
    @wraps(func)
    def sync_wrapper(self, *args, **kwargs):
        if not hasattr(self, 'client') or self.client is None:
            raise ValueError(f"{self.__class__.__name__} requires a GraphitiClient")
        if hasattr(self.client, '_is_closed') and self.client._is_closed:
            raise ValueError("GraphitiClient has been closed")
        return func(self, *args, **kwargs)
    
    # Return appropriate wrapper based on whether func is async
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def safe_sync_run(coro: Any, timeout: Optional[float] = None) -> Any:
    """
    Safely run an async coroutine in a sync context.
    
    Handles cases where there's already an event loop running.
    """
    try:
        loop = asyncio.get_running_loop()
        # If we're in an async context, we need to use a different approach
        if loop.is_running():
            import concurrent.futures
            import threading
            
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    if timeout:
                        return new_loop.run_until_complete(asyncio.wait_for(coro, timeout))
                    else:
                        return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=timeout)
        else:
            # Use the existing event loop
            if timeout:
                return loop.run_until_complete(asyncio.wait_for(coro, timeout))
            else:
                return loop.run_until_complete(coro)
                
    except RuntimeError:
        # No event loop running, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            if timeout:
                return loop.run_until_complete(asyncio.wait_for(coro, timeout))
            else:
                return loop.run_until_complete(coro)
        finally:
            loop.close()


def validate_config_dict(config: dict, required_keys: list[str], optional_keys: list[str] = None) -> dict:
    """Validate configuration dictionary has required keys."""
    optional_keys = optional_keys or []
    
    # Check required keys
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    # Check for unknown keys
    all_valid_keys = set(required_keys + optional_keys)
    unknown_keys = [key for key in config.keys() if key not in all_valid_keys]
    if unknown_keys:
        logger.warning(f"Unknown configuration keys (will be ignored): {unknown_keys}")
    
    # Return only valid keys
    return {key: value for key, value in config.items() if key in all_valid_keys}


def format_graph_results(results: Any, max_items: int = 10) -> str:
    """Format Graphiti search results for display."""
    if not results:
        return "No results found."
    
    if hasattr(results, 'edges') and hasattr(results, 'nodes'):
        # SearchResults object
        parts = []
        
        if results.edges:
            edge_count = len(results.edges)
            parts.append(f"📊 Found {edge_count} relationships")
            if edge_count <= max_items:
                for edge in results.edges:
                    parts.append(f"  • {edge.name}: {edge.fact[:100]}...")
            else:
                for edge in results.edges[:max_items]:
                    parts.append(f"  • {edge.name}: {edge.fact[:100]}...")
                parts.append(f"  ... and {edge_count - max_items} more")
        
        if results.nodes:
            node_count = len(results.nodes)
            parts.append(f"🔵 Found {node_count} entities")
            if node_count <= max_items:
                for node in results.nodes:
                    parts.append(f"  • {node.name}: {(node.summary or 'No summary')[:100]}...")
            else:
                for node in results.nodes[:max_items]:
                    parts.append(f"  • {node.name}: {(node.summary or 'No summary')[:100]}...")
                parts.append(f"  ... and {node_count - max_items} more")
        
        return "\n".join(parts)
    
    elif isinstance(results, list):
        # List of documents or similar
        count = len(results)
        parts = [f"📋 Found {count} results"]
        
        display_count = min(count, max_items)
        for i, item in enumerate(results[:display_count]):
            if hasattr(item, 'page_content'):
                content = item.page_content[:100] + "..." if len(item.page_content) > 100 else item.page_content
                parts.append(f"  {i+1}. {content}")
            else:
                parts.append(f"  {i+1}. {str(item)[:100]}...")
        
        if count > max_items:
            parts.append(f"  ... and {count - max_items} more")
        
        return "\n".join(parts)
    
    else:
        return str(results)[:500] + "..." if len(str(results)) > 500 else str(results)


__all__ = [
    "require_client",
    "safe_sync_run", 
    "validate_config_dict",
    "format_graph_results",
]