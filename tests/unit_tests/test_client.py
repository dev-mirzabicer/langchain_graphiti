"""Unit tests for the GraphitiClient."""

import pytest
from unittest.mock import MagicMock, patch

from langchain_graphiti._client import GraphitiClient, create_graphiti_client
from langchain_graphiti.exceptions import (
    GraphitiConfigurationError,
)
from graphiti_core import Graphiti
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.llm_client import LLMClient
from graphiti_core.embedder import EmbedderClient
from graphiti_core.cross_encoder.client import CrossEncoderClient

# --- Mock Classes that pass isinstance checks ---

class MockGraphDriver(GraphDriver):
    def __init__(self, *args, **kwargs): pass
    async def close(self): pass
    def session(self): return MagicMock()
    async def delete_all_indexes(self): pass
    async def execute_query(self, query, **kwargs): return []

class MockLLMClient(LLMClient):
    def __init__(self, *args, **kwargs): pass
    async def _generate_response(self, messages, **kwargs): return "mock response"

class MockEmbedderClient(EmbedderClient):
    def __init__(self, *args, **kwargs): pass
    async def create(self, text, **kwargs): return [0.1, 0.2, 0.3]

class MockCrossEncoderClient(CrossEncoderClient):
    def __init__(self, *args, **kwargs): pass
    async def rank(self, query, documents, **kwargs): return documents

class MockGraphiti(Graphiti):
    def __init__(self, *args, **kwargs):
        self.driver = MockGraphDriver()
        self.llm_client = MockLLMClient()
        self.embedder = MockEmbedderClient()
        self.cross_encoder = MockCrossEncoderClient()
        self.clients = MagicMock()

# --- Fixtures ---

@pytest.fixture
def mock_driver():
    """Fixture for a mocked GraphDriver."""
    return MockGraphDriver()

@pytest.fixture
def mock_llm_client():
    """Fixture for a mocked LLMClient."""
    return MockLLMClient()

@pytest.fixture
def mock_embedder():
    """Fixture for a mocked EmbedderClient."""
    return MockEmbedderClient()

@pytest.fixture
def mock_cross_encoder():
    """Fixture for a mocked CrossEncoderClient."""
    return MockCrossEncoderClient()

@pytest.fixture
def mock_graphiti_instance():
    """Fixture for a mocked Graphiti instance."""
    return MockGraphiti()

# --- Tests ---

def test_from_connections_success(
    mock_driver, mock_llm_client, mock_embedder, mock_cross_encoder
):
    """Test successful creation of GraphitiClient from connections."""
    with patch("langchain_graphiti._client.Graphiti", return_value=MockGraphiti()) as MockGraphitiClass:
        client = GraphitiClient.from_connections(
            driver=mock_driver,
            llm_client=mock_llm_client,
            embedder=mock_embedder,
            cross_encoder=mock_cross_encoder,
        )
        assert isinstance(client, GraphitiClient)
        MockGraphitiClass.assert_called_once()

def test_from_connections_missing_driver(
    mock_llm_client, mock_embedder, mock_cross_encoder
):
    """Test that from_connections raises an error if the driver is missing."""
    with pytest.raises(GraphitiConfigurationError, match="Graph driver is required"):
        GraphitiClient.from_connections(
            driver=None,
            llm_client=mock_llm_client,
            embedder=mock_embedder,
            cross_encoder=mock_cross_encoder,
        )

def test_initialization_with_instance(mock_graphiti_instance):
    """Test initializing GraphitiClient with a pre-configured instance."""
    client = GraphitiClient(graphiti_instance=mock_graphiti_instance)
    assert client.graphiti_instance == mock_graphiti_instance

@patch.dict(
    "os.environ",
    {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "password",
        "OPENAI_API_KEY": "test-key",
    },
)
@patch("langchain_graphiti._client.Graphiti")
@patch("langchain_graphiti._client.GeminiRerankerClient")
@patch("langchain_graphiti._client.GeminiEmbedder")
@patch("langchain_graphiti._client.GeminiClient")
@patch("langchain_graphiti._client.Neo4jDriver")
def test_create_graphiti_client_with_defaults(
    MockNeo4jDriver, MockGeminiClient, MockGeminiEmbedder, MockGeminiRerankerClient, MockGraphitiClass
):
    """Test the create_graphiti_client factory with default clients."""
    with patch.dict(
        "os.environ",
        {
            "NEO4J_URI": "bolt://localhost:7687",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "GEMINI_API_KEY": "test-key",
        },
    ):
        MockNeo4jDriver.return_value = MockGraphDriver()
        MockGeminiClient.return_value = MockLLMClient()
        MockGeminiEmbedder.return_value = MockEmbedderClient()
        MockGeminiRerankerClient.return_value = MockCrossEncoderClient()
        MockGraphitiClass.return_value = MockGraphiti()

        client = create_graphiti_client()
        assert isinstance(client, GraphitiClient)
        MockNeo4jDriver.assert_called_once_with(
            uri="bolt://localhost:7687", user="neo4j", password="password"
        )
        MockGeminiClient.assert_called_once()
        MockGeminiEmbedder.assert_called_once()
        MockGeminiRerankerClient.assert_called_once()
        MockGraphitiClass.assert_called_once()

@patch.dict("os.environ", {}, clear=True)
def test_create_graphiti_client_missing_env_vars():
    """Test that create_graphiti_client raises an error if env vars are missing."""
    with pytest.raises(
        GraphitiConfigurationError, match="NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD"
    ):
        create_graphiti_client()