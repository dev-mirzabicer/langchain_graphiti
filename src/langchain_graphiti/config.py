"""
Configuration objects for Graphiti providers and drivers.

This module defines the data structures for configuring the various
backend services that Graphiti can use. It includes:
- Enums for provider and driver types for type-safe selection
- Pydantic models for provider-specific configurations
"""

from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional

# --- Provider Enums ---

class LLMProvider(str, Enum):
    """Enum for supported LLM providers."""
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    OLLAMA = "ollama"

class DriverProvider(str, Enum):
    """Enum for supported graph database drivers."""
    NEO4J = "neo4j"
    FALKORDB = "falkordb"

# --- Configuration Models ---

class OpenAIConfig(BaseModel):
    """Configuration for OpenAI and compatible providers (Ollama)."""
    api_key: str = Field(..., description="API key for the provider.")
    model: str = Field("gpt-4o", description="The primary LLM model to use.")
    small_model: Optional[str] = Field(None, description="A smaller, faster model for tasks like reranking.")
    embedding_model: str = Field("text-embedding-3-small", description="The embedding model to use.")
    embedding_dim: int = Field(1536, description="The dimension of the embeddings.")
    base_url: Optional[str] = Field(None, description="The base URL for the API, for proxies or local models like Ollama.")

class AzureOpenAIConfig(BaseModel):
    """Configuration for Azure OpenAI."""
    api_key: str = Field(..., description="Azure OpenAI API key.")
    api_version: str = Field(..., description="Azure OpenAI API version.")
    llm_endpoint: str = Field(..., description="The endpoint for the LLM deployment.")
    embedding_endpoint: str = Field(..., description="The endpoint for the embedding deployment.")
    llm_deployment: str = Field(..., description="The name of the LLM deployment.")
    embedding_deployment: str = Field(..., description="The name of the embedding deployment.")
    small_llm_deployment: Optional[str] = Field(None, description="The name of the smaller LLM deployment.")
    embedding_dim: int = Field(1536, description="The dimension of the embeddings.")

class GeminiConfig(BaseModel):
    """Configuration for Google Gemini."""
    api_key: str = Field(..., description="Google API key for Gemini.")
    model: str = Field("gemini-1.5-flash", description="The primary LLM model to use.")
    embedding_model: str = Field("text-embedding-004", description="The embedding model to use.")
    reranker_model: Optional[str] = Field("gemini-1.5-flash", description="The model to use for reranking.")

class AnthropicConfig(BaseModel):
    """Configuration for Anthropic."""
    api_key: str = Field(..., description="Anthropic API key.")
    model: str = Field("claude-3-sonnet-20240229", description="The primary LLM model to use.")

class GroqConfig(BaseModel):
    """Configuration for Groq."""
    api_key: str = Field(..., description="Groq API key.")
    model: str = Field("llama3-8b-8192", description="The primary LLM model to use.")

class Neo4jConfig(BaseModel):
    """Configuration for the Neo4j driver."""
    uri: str = Field(..., description="The URI for the Neo4j database.")
    user: str = Field("neo4j", description="The username for the Neo4j database.")
    password: str = Field(..., description="The password for the Neo4j database.")

class FalkorDBConfig(BaseModel):
    """Configuration for the FalkorDB driver."""
    uri: str = Field(..., description="The URI for the FalkorDB database (e.g., falkor://localhost:6379).")