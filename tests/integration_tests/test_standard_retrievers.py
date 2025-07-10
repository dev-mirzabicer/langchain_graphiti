"""Standard LangChain integration tests for Graphiti retrievers."""

import pytest
import asyncio
from typing import Type, Dict, Any, Generator, ClassVar, Optional
from datetime import datetime
from langchain_core.retrievers import BaseRetriever
from langchain_tests.integration_tests.retrievers import RetrieversIntegrationTests

from langchain_graphiti import create_graphiti_client
from langchain_graphiti.retrievers import GraphitiRetriever
from langchain_graphiti._client import GraphitiClient

# --- Module-level client for all retriever tests ---

_client: Optional[GraphitiClient] = None
_TEST_GROUP_ID: ClassVar[str] = "standard-retriever-test"


@pytest.fixture(scope="module", autouse=True)
async def setup_and_teardown_client() -> Generator[None, None, None]:
    """Module-level fixture to manage the client for all retriever tests."""
    global _client
    try:
        _client = create_graphiti_client()
    except Exception as e:
        pytest.skip(f"Could not create Graphiti client: {e}")

    docs_to_add = [
        (
            "Standard Test Document 1",
            "This document is for the standard LangChain retriever tests.",
        ),
        ("Standard Test Document 2", "This is another document for testing."),
        ("Standard Test Document 3", "A third document to ensure k=3 works."),
    ]

    # Setup
    await _client.graphiti_instance.build_indices_and_constraints(delete_existing=True)
    for name, content in docs_to_add:
        await _client.graphiti_instance.add_episode(
            name=name,
            episode_body=content,
            source_description="Standard test suite",
            group_id=_TEST_GROUP_ID,
            reference_time=datetime.now(),
        )
    
    yield

    # Teardown
    await _client.graphiti_instance.driver.execute_query(
        "MATCH (n {group_id: $group_id}) DETACH DELETE n",
        group_id=_TEST_GROUP_ID,
    )
    await _client.close()


class TestGraphitiStandardRetriever(RetrieversIntegrationTests):
    """
    Standard LangChain integration test suite for the GraphitiRetriever.
    """

    @property
    def retriever_constructor(self) -> Type[BaseRetriever]:
        """The retriever class to test."""
        return GraphitiRetriever

    @property
    def retriever_constructor_params(self) -> Dict[str, Any]:
        """Parameters for constructing the retriever."""
        return {
            "client": _client,
            "group_ids": [_TEST_GROUP_ID],
        }

    @property
    def retriever_query_example(self) -> str:
        """An example query that should return results."""
        return "standard LangChain retriever tests"