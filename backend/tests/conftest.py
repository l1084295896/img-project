import os
import pytest

@pytest.fixture
def templates_dir():
    return os.path.join(os.path.dirname(__file__), "..", "templates")
