"""Standard LangChain integration tests for Graphiti retrievers."""

import pytest
import asyncio
from typing import Type, Dict, Any, Generator, ClassVar, Optional
from datetime import datetime
from langchain_core.retrievers import BaseRetriever
from langchain_tests.integration_tests.retrievers import RetrieversIntegrationTests
from langchain_graphiti.retrievers import GraphitiRetriever
from langchain_graphiti._client import GraphitiClient

_TEST_GROUP_ID: ClassVar[str] = "standard-retriever-test"
_client_for_standard_retrievers: Optional[GraphitiClient] = None


@pytest.fixture(scope="module", autouse=True)
def client_for_standard_retrievers(client_for_integration: GraphitiClient) -> Generator[GraphitiClient, None, None]:
    """Setup and teardown for standard retriever tests."""
    global _client_for_standard_retrievers
    _client_for_standard_retrievers = client_for_integration
    
    docs_to_add = [
        (
            "Standard Test Document 1",
            "This document is for the standard LangChain retriever tests.",
        ),
        ("Standard Test Document 2", "This is another document for testing."),
        ("Standard Test Document 3", "A third document to ensure k=3 works."),
    ]

    async def setup():
        await _client_for_standard_retrievers.graphiti_instance.build_indices_and_constraints(delete_existing=True)
        for name, content in docs_to_add:
            await _client_for_standard_retrievers.graphiti_instance.add_episode(
                name=name,
                episode_body=content,
                source_description="Standard test suite",
                group_id=_TEST_GROUP_ID,
                reference_time=datetime.now(),
            )

    asyncio.run(setup())
    yield _client_for_standard_retrievers

    async def teardown():
        await _client_for_standard_retrievers.graphiti_instance.driver.execute_query(
            "MATCH (n {group_id: $group_id}) DETACH DELETE n",
            group_id=_TEST_GROUP_ID,
        )

    asyncio.run(teardown())


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
        # This is a bit of a hack to work with the langchain_tests framework
        # which doesn't nicely support pytest fixtures in properties.
        global _client_for_standard_retrievers
        return {
            "client": _client_for_standard_retrievers,
            "group_ids": [_TEST_GROUP_ID],
        }

    @property
    def retriever_query_example(self) -> str:
        """An example query that should return results."""
        return "standard LangChain retriever tests"
