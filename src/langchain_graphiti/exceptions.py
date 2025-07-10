"""Custom exception hierarchy for langchain-graphiti."""

from typing import Optional
from langchain_core.exceptions import LangChainException


class LangChainGraphitiError(LangChainException):
    """Base exception for all langchain-graphiti errors."""
    pass


class GraphitiClientError(LangChainGraphitiError):
    """Base exception for GraphitiClient errors."""
    pass


class GraphitiConnectionError(GraphitiClientError):
    """Raised when connection to Graphiti fails."""
    
    def __init__(self, message: str, connection_details: Optional[dict] = None):
        super().__init__(message)
        self.connection_details = connection_details or {}


class GraphitiConfigurationError(GraphitiClientError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, invalid_config: Optional[dict] = None):
        super().__init__(message)
        self.invalid_config = invalid_config or {}


class GraphitiOperationError(GraphitiClientError):
    """Raised when a Graphiti operation fails."""
    
    def __init__(self, message: str, operation_details: Optional[dict] = None):
        super().__init__(message)
        self.operation_details = operation_details or {}


class GraphitiRetrieverError(LangChainGraphitiError):
    """Raised when retrieval operations fail."""
    pass


class GraphitiToolError(LangChainGraphitiError):
    """Raised when tool operations fail."""
    
    def __init__(self, message: str, tool_name: Optional[str] = None):
        super().__init__(message)
        self.tool_name = tool_name


__all__ = [
    "LangChainGraphitiError",
    "GraphitiClientError", 
    "GraphitiConnectionError",
    "GraphitiConfigurationError",
    "GraphitiOperationError",
    "GraphitiRetrieverError",
    "GraphitiToolError",
]