from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture
def client():
    mock_agent = MagicMock()
    api.app.dependency_overrides[api.get_agent] = lambda: mock_agent
    yield TestClient(api.app), mock_agent
    api.app.dependency_overrides.clear()


@pytest.mark.unit
def test_ask_returns_agent_answer(client):
    test_client, mock_agent = client
    mock_agent.answer.return_value = "Charging at work."

    response = test_client.post("/ask", json={"question": "What reduces fleet costs?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "Charging at work."}
    mock_agent.answer.assert_called_once_with("What reduces fleet costs?")


@pytest.mark.unit
def test_ask_rejects_empty_question(client):
    test_client, mock_agent = client

    response = test_client.post("/ask", json={"question": "   "})

    assert response.status_code == 422
    mock_agent.answer.assert_not_called()


@pytest.mark.unit
def test_ask_rejects_missing_question_field(client):
    test_client, mock_agent = client

    response = test_client.post("/ask", json={})

    assert response.status_code == 422
    mock_agent.answer.assert_not_called()
