"""
Global pytest configuration for the test suite.

This file defines a session-scoped asyncio event_loop fixture, which allows
async fixtures with a scope wider than 'function' (e.g., 'module' or 'session')
to run correctly. This resolves the ScopeMismatch errors encountered when
running the integration tests.
"""

import pytest
import asyncio
from typing import Generator

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """
    Create an instance of the default event loop for the entire test session.
    
    This allows module-scoped async fixtures to run without ScopeMismatch errors.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()