# 🦜🔗 Graphiti for LangChain

[![PyPI version](https://badge.fury.io/py/langchain-graphiti.svg)](https://badge.fury.io/py/langchain-graphiti)
[![CI/CD Status](https://github.com/dev-mirzabicer/langchain-graphiti/actions/workflows/test.yml/badge.svg)](https://github.com/dev-mirzabicer/langchain-graphiti/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**`langchain-graphiti`** is a production-grade, high-quality LangChain integration for the [Graphiti](https://github.com/getzep/graphiti) knowledge graph system. It provides a robust set of tools and retrievers that empower LangChain agents with long-term, evolving memory and sophisticated graph-based reasoning capabilities.

This library allows your agents to dynamically build and query a knowledge graph, transforming unstructured information into structured, queryable memory that persists across sessions.

---

## Core Concepts

At its core, this integration bridges the gap between LangChain's powerful agentic frameworks and Graphiti's ability to create real-time knowledge graphs.

-   **Graphiti:** An open-source framework that ingests unstructured data (like conversations or documents) and uses LLMs to automatically extract entities and relationships, building a rich knowledge graph.
-   **LangChain Integration:** This library exposes Graphiti's functionality through standard LangChain interfaces (`BaseTool`, `BaseRetriever`), allowing agents to seamlessly read from, write to, and manage the knowledge graph as part of their workflow.

The result is an AI agent that doesn't just operate on short-term context but learns, remembers, and reasons over a persistent, interconnected web of information.

## Features

-   **Comprehensive Toolset:** A full suite of tools for adding, searching, and managing information in the knowledge graph.
-   **Advanced Retrievers:** Powerful retrievers that leverage Graphiti's hybrid search (text, vector, and graph traversal) for highly relevant and context-aware information retrieval.
-   **Robust Client:** A thread-safe, async-first client with built-in health checks, connection management, and error handling.
-   **Production-Ready:** Designed with a focus on quality, robustness, and meticulous engineering for reliable deployment.
-   **Seamless Integration:** Built to feel like a native part of the LangChain ecosystem.

## Installation

```bash
pip install langchain-graphiti
```

You will also need to install extras depending on the LLM and database providers you intend to use:

```bash
# Example for OpenAI and Neo4j
pip install "langchain-graphiti[openai,neo4j]"
```

See `pyproject.toml` for a full list of available extras.

## Quick Start

Here's a simple example of how to get up and running with `langchain-graphiti`. This example uses `langchain-openai` and a local Neo4j database.

```python
import asyncio
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from langchain_graphiti import GraphitiClient
from langchain_graphiti.tools import create_basic_agent_tools
from langchain_graphiti.config import LLMProvider, DriverProvider, OpenAIConfig, Neo4jConfig

# 1. Configure your providers
llm_config = OpenAIConfig()  # Assumes OPENAI_API_KEY is in your environment
driver_config = Neo4jConfig() # Assumes NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD are set

# 2. Create the GraphitiClient
# This client manages the connection to your knowledge graph
client = GraphitiClient.from_factory(
    llm_provider=LLMProvider.OPENAI,
    driver_provider=DriverProvider.NEO4J,
    llm_config=llm_config,
    driver_config=driver_config,
)

# 3. Create the agent tools
# These are the functions the agent can call to interact with the graph
tools = create_basic_agent_tools(client)

# 4. Set up your LangChain agent
llm = ChatOpenAI(model="gpt-4o")
agent_executor = create_react_agent(llm, tools)

async def main():
    # Let's teach the agent something new
    await agent_executor.ainvoke({
        "messages": [("user", "Add this fact: 'Project Phoenix is managed by Alice.'")]
    })

    # Now, let's ask a question that requires recalling that fact
    response = await agent_executor.ainvoke({
        "messages": [("user", "Who manages Project Phoenix?")]
    })

    print(response["messages"][-1].content)
    # Expected output will mention that Alice manages Project Phoenix.

    # Clean up the client connection
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Advanced Usage

### Using the `GraphitiRetriever`

The `GraphitiRetriever` is a powerful tool for fetching documents from your knowledge graph. It can be used in any standard LangChain RAG (Retrieval-Augmented Generation) chain.

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_graphiti import GraphitiRetriever

# Assume 'client' is an initialized GraphitiClient
retriever = GraphitiRetriever(client=client, k=5)

template = """
Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# This will search the graph for context related to the question
result = rag_chain.invoke("What were the key outcomes of the Q3 budget meeting?")
print(result)
```

### Using the Full Toolset

For agents that require more advanced control over the knowledge graph, you can use the `create_agent_tools` factory to get the complete set of tools.

```python
from langchain_graphiti.tools import create_agent_tools

# Get all available tools
advanced_tools = create_agent_tools(client)

# Create an agent with full capabilities
advanced_agent = create_react_agent(llm, advanced_tools)

# This agent can now perform advanced operations like:
# - Building communities
# - Removing specific episodes
# - Adding structured triplets directly
# - Optimizing the database with indices
```

## Contributing

We welcome contributions to `langchain-graphiti`! Whether it's reporting a bug, suggesting a new feature, or submitting a pull request, your help is valued.

Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for detailed guidelines on how to contribute to the project.