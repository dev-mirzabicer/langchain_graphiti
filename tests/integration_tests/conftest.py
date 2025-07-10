"""
Shared fixtures for Graphiti integration tests.
"""
import pytest
import os
from langchain_graphiti import (
    GraphitiClientFactory,
    LLMProvider,
    DriverProvider,
    GeminiConfig,
    Neo4jConfig,
    GraphitiClient,
)
from typing import Generator
from langchain_graphiti.utils import safe_sync_run


def create_test_client() -> GraphitiClient:
    """Helper to create a client for testing from environment variables."""
    gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("TEST_GEMINI_API_KEY")
    neo4j_uri = os.getenv("NEO4J_URI") or os.getenv("TEST_URI")
    neo4j_user = os.getenv("NEO4J_USER") or os.getenv("TEST_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD") or os.getenv("TEST_PASSWORD")

    if not all([gemini_api_key, neo4j_uri, neo4j_user, neo4j_password]):
        raise ConnectionError(
            "Missing env vars for integration tests: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, GEMINI_API_KEY"
        )

    llm_config = GeminiConfig(api_key=gemini_api_key)
    driver_config = Neo4jConfig(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)

    return GraphitiClientFactory.create(
        llm_provider=LLMProvider.GEMINI,
        driver_provider=DriverProvider.NEO4J,
        llm_config=llm_config,
        driver_config=driver_config,
    )


@pytest.fixture(scope="module")
def client_for_integration() -> Generator[GraphitiClient, None, None]:
    """Module-level fixture for integration tests."""
    try:
        client = create_test_client()
    except Exception as e:
        pytest.skip(f"Could not create Graphiti client, skipping integration test: {e}")

    yield client

    # Teardown
    safe_sync_run(client.close())