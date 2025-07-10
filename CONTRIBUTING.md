# Contributing to Graphiti for LangChain

First off, thank you for considering contributing to `langchain-graphiti`. Your help is invaluable in making this library better for everyone.

This document provides guidelines for contributing to the project. Whether you're reporting a bug, suggesting a feature, or writing code, these guidelines are here to help.

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior.

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please ensure the bug was not already reported by searching on GitHub under [Issues](https://github.com/dev-mirzabicer/langchain-graphiti/issues).

If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/dev-mirzabicer/langchain-graphiti/issues/new). Be sure to include a **title and clear description**, as much relevant information as possible, and a **code sample** or an **executable test case** demonstrating the expected behavior that is not occurring.

### Suggesting Enhancements

If you have an idea for a new feature or an improvement to an existing one, please open an issue with the "enhancement" label. Provide a clear and detailed explanation of the feature, why it's needed, and how it would work.

### Pull Requests

We love pull requests! If you're ready to contribute code, here's how to set up your environment and submit your changes.

## Development Setup

1.  **Fork the repository** on GitHub.

2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/your-username/langchain-graphiti.git
    cd langchain-graphiti
    ```

3.  **Install Poetry**, our dependency manager:
    ```bash
    pip install poetry
    ```

4.  **Install project dependencies**, including all optional extras for testing:
    ```bash
    poetry install --all-extras
    ```

5.  **Run the test suite** to ensure everything is set up correctly:
    ```bash
    poetry run pytest
    ```

6.  **Create a new branch** for your changes:
    ```bash
    git checkout -b your-feature-branch-name
    ```

## Pull Request Process

1.  **Make your changes.** Make sure to add tests for any new functionality.

2.  **Ensure all tests pass** before submitting:
    ```bash
    poetry run pytest
    ```

3.  **Run the linter and formatter** to ensure your code follows our style guidelines:
    ```bash
    # To check for linting issues
    poetry run ruff check .
    poetry run mypy src

    # To automatically format your code
    poetry run ruff format .
    ```

4.  **Commit your changes** with a clear and descriptive commit message.

5.  **Push your branch** to your fork on GitHub:
    ```bash
    git push origin your-feature-branch-name
    ```

6.  **Open a pull request** to the `main` branch of the original repository. Provide a clear description of the changes and link to any relevant issues.

Once you've submitted your pull request, a project maintainer will review your changes. We may ask for modifications before merging.

Thank you again for your contribution!