import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import app.service  # noqa: F401  # Ensure service package finishes initialization before importing Agent.
from agentlang.event.correlation_id_manager import CorrelationIdManager
from app.magic.agent import Agent


def _create_uninitialized_agent(context_id: str) -> Agent:
    agent = Agent.__new__(Agent)
    agent.agent_context = SimpleNamespace(
        context_id=context_id,
        dispatch_event=AsyncMock(),
    )
    agent.agent_name = "mock-agent"
    agent._closed = False
    agent._context_registered = False
    return agent


def test_agent_close_clears_only_its_stream_fallback_scope(monkeypatch):
    manager = CorrelationIdManager()
    manager.set_stream_fallback_cid("parent-request", "parent-context")
    manager.set_stream_fallback_cid("child-request", "child-context")
    monkeypatch.setattr("agentlang.event.get_correlation_manager", lambda: manager)

    agent = _create_uninitialized_agent("child-context")
    agent.close()
    agent.close()

    assert manager.pop_stream_fallback_cid("child-context") is None
    assert manager.pop_stream_fallback_cid("parent-context") == "parent-request"


@pytest.mark.asyncio
async def test_run_main_agent_cancellation_clears_only_its_stream_fallback_scope(monkeypatch):
    manager = CorrelationIdManager()
    manager.set_stream_fallback_cid("parent-request", "parent-context")
    manager.set_stream_fallback_cid("child-request", "child-context")
    monkeypatch.setattr("agentlang.event.get_correlation_manager", lambda: manager)
    monkeypatch.setattr(
        "app.magic.agent.BeforeMainAgentRunEventData",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    agent = _create_uninitialized_agent("child-context")
    agent.run = AsyncMock(side_effect=asyncio.CancelledError())
    agent._reclaim_memory = Mock()

    with pytest.raises(asyncio.CancelledError):
        await agent.run_main_agent("mock query")

    agent._reclaim_memory.assert_called_once_with()
    assert manager.pop_stream_fallback_cid("child-context") is None
    assert manager.pop_stream_fallback_cid("parent-context") == "parent-request"
