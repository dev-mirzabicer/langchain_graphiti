"""Unit tests for the GraphitiClient."""

import pytest
from unittest.mock import MagicMock, patch, ANY

from langchain_graphiti._client import GraphitiClient, GraphitiClientFactory
from langchain_graphiti.config import (
    LLMProvider,
    DriverProvider,
    OpenAIConfig,
    Neo4jConfig,
    GeminiConfig,
)
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

@patch("langchain_graphiti._client.GraphitiClientFactory._import_class")
def test_factory_create_openai_neo4j(mock_import):
    """Test the factory with OpenAI and Neo4j."""
    # Mock the imported classes
    MockNeo4jDriver = MagicMock()
    MockNeo4jDriver = MagicMock()
    MockOpenAIClient = MagicMock()
    MockOpenAIEmbedder = MagicMock()
    MockOpenAIRerankerClient = MagicMock()

    def import_side_effect(module, class_name, extra):
        if class_name == "Neo4jDriver":
            return MockNeo4jDriver
        if class_name == "OpenAIClient":
            return MockOpenAIClient
        if class_name == "OpenAIEmbedder":
            return MockOpenAIEmbedder
        if class_name == "OpenAIRerankerClient":
            return MockOpenAIRerankerClient
        if class_name in ["LLMConfig", "OpenAIEmbedderConfig"]:
            return MagicMock()
        return MagicMock()

    mock_import.side_effect = import_side_effect

    llm_config = OpenAIConfig(api_key="test-key")
    driver_config = Neo4jConfig(uri="bolt://localhost", user="neo4j", password="password")

    with patch("langchain_graphiti._client.GraphitiClient.from_connections") as mock_from_connections:
        GraphitiClientFactory.create(
            llm_provider=LLMProvider.OPENAI,
            driver_provider=DriverProvider.NEO4J,
            llm_config=llm_config,
            driver_config=driver_config,
        )
        mock_from_connections.assert_called_once_with(
            driver=ANY, llm_client=ANY, embedder=ANY, cross_encoder=ANY
        )
        MockNeo4jDriver.assert_called_once_with(uri="bolt://localhost", user="neo4j", password="password")
        MockOpenAIClient.assert_called_once()
        MockOpenAIEmbedder.assert_called_once()
        MockOpenAIRerankerClient.assert_called_once()

@patch("langchain_graphiti._client.importlib.import_module")
def test_factory_import_error(mock_import_module):
    """Test that the factory raises an ImportError if a package is missing."""
    
    def import_side_effect(module_path):
        if "gemini" in module_path:
            raise ImportError(f"No module named '{module_path}'")
        return MagicMock()

    mock_import_module.side_effect = import_side_effect
    
    llm_config = GeminiConfig(api_key="test-key")
    driver_config = Neo4jConfig(uri="bolt://localhost", user="neo4j", password="password")
    
    with pytest.raises(ImportError, match="The 'GeminiClient' client requires the 'gemini' extra."):
        GraphitiClientFactory.create(
            llm_provider=LLMProvider.GEMINI,
            driver_provider=DriverProvider.NEO4J,
            llm_config=llm_config,
            driver_config=driver_config,
        )