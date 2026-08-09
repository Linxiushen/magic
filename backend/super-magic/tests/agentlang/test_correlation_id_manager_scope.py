import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agentlang.event.correlation_id_manager import CorrelationIdManager, EventPairType
from agentlang.event.reply_event_manager import ReplyEventManager
from agentlang.event.think_event_manager import ThinkEventManager
from agentlang.llms.processors.processor_config import ProcessorConfig
from agentlang.llms.processors.processor_manager import ProcessorManager
from agentlang.llms.processors.streaming_call_processor import StreamingCallProcessor


class _AgentContext:
    def __init__(self, context_id: str):
        self.context_id = context_id
        self.metadata = {}

    def set_metadata(self, key, value):
        self.metadata[key] = value


async def _execute_stream(context_id, request_id):
    return await ProcessorManager.execute_llm_call(
        client=None,
        llm_config=None,
        request_params={},
        model_id="mock-model",
        processor_config=ProcessorConfig(use_stream_mode=True),
        agent_context=_AgentContext(context_id),
        request_id=request_id,
    )


async def _execute_regular(context_id, request_id, response):
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=response)),
        )
    )
    return await ProcessorManager.execute_llm_call(
        client=client,
        llm_config=SimpleNamespace(name="Mock Model"),
        request_params={},
        model_id="mock-model",
        processor_config=ProcessorConfig(use_stream_mode=False),
        agent_context=_AgentContext(context_id),
        request_id=request_id,
    )


def test_tool_call_correlation_is_isolated_by_scope():
    manager = CorrelationIdManager()

    parent_correlation_id = manager.generate_for_before_event(
        EventPairType.TOOL_CALL,
        "mock-parent-context",
    )
    child_correlation_id = manager.generate_for_before_event(
        EventPairType.TOOL_CALL,
        "mock-child-context",
    )

    assert child_correlation_id != parent_correlation_id
    assert manager.consume_for_after_event(EventPairType.TOOL_CALL, "mock-child-context") == child_correlation_id
    assert manager.consume_for_after_event(EventPairType.TOOL_CALL, "mock-parent-context") == parent_correlation_id


def test_correlation_manager_keeps_global_scope_compatibility():
    manager = CorrelationIdManager()

    first_correlation_id = manager.generate_for_before_event(EventPairType.TOOL_CALL)
    retried_correlation_id = manager.generate_for_before_event(EventPairType.TOOL_CALL)

    assert retried_correlation_id == first_correlation_id
    assert manager.consume_for_after_event(EventPairType.TOOL_CALL) == first_correlation_id


def test_stream_fallback_correlation_is_isolated_by_scope():
    manager = CorrelationIdManager()

    manager.set_stream_fallback_cid("parent-request", "mock-parent-context")
    manager.set_stream_fallback_cid("child-request", "mock-child-context")

    assert manager.pop_stream_fallback_cid("mock-parent-context") == "parent-request"
    assert manager.pop_stream_fallback_cid("mock-child-context") == "child-request"


def test_clearing_stream_fallback_only_affects_target_scope():
    manager = CorrelationIdManager()

    manager.set_stream_fallback_cid("parent-request", "mock-parent-context")
    manager.set_stream_fallback_cid("child-request", "mock-child-context")
    manager.clear_stream_fallback_cid("mock-child-context")

    assert manager.pop_stream_fallback_cid("mock-child-context") is None
    assert manager.pop_stream_fallback_cid("mock-parent-context") == "parent-request"


def test_stream_fallback_correlation_keeps_global_scope_compatibility():
    manager = CorrelationIdManager()

    manager.set_stream_fallback_cid("stale-request")
    manager.set_stream_fallback_cid(None)
    assert manager.pop_stream_fallback_cid() is None

    manager.set_stream_fallback_cid("global-request")

    assert manager.pop_stream_fallback_cid() == "global-request"
    assert manager.pop_stream_fallback_cid() is None


@pytest.mark.asyncio
async def test_concurrent_stream_failures_keep_each_context_fallback(monkeypatch):
    manager = CorrelationIdManager()
    entered_requests = set()
    both_streams_entered = asyncio.Event()

    async def fail_after_both_streams_enter(**kwargs):
        entered_requests.add(kwargs["request_id"])
        if len(entered_requests) == 2:
            both_streams_entered.set()
        await asyncio.wait_for(both_streams_entered.wait(), timeout=5)
        raise RuntimeError("mock stream failure")

    monkeypatch.setattr("agentlang.event.get_correlation_manager", lambda: manager)
    monkeypatch.setattr(
        StreamingCallProcessor,
        "call_with_stream",
        fail_after_both_streams_enter,
    )

    results = await asyncio.gather(
        _execute_stream("mock-parent-context", "parent-request"),
        _execute_stream("mock-child-context", "child-request"),
        return_exceptions=True,
    )

    assert all(isinstance(result, RuntimeError) for result in results)
    after_reply = AsyncMock()
    before_reply = AsyncMock()
    monkeypatch.setattr(ThinkEventManager, "trigger_after_think", AsyncMock(return_value=True))
    monkeypatch.setattr(ReplyEventManager, "trigger_before_reply", before_reply)
    monkeypatch.setattr(ReplyEventManager, "trigger_after_reply", after_reply)

    response = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                reasoning_content=None,
                content="mock final response",
                tool_calls=None,
            )
        )]
    )

    await _execute_regular("mock-parent-context", "parent-fallback-request", response)
    await _execute_regular("mock-child-context", "child-fallback-request", response)

    before_reply.assert_not_awaited()
    fallback_correlations = [
        call.kwargs["correlation_id"]
        for call in after_reply.await_args_list
    ]
    assert fallback_correlations == ["parent-request", "child-request"]
    assert manager.pop_stream_fallback_cid("mock-parent-context") is None
    assert manager.pop_stream_fallback_cid("mock-child-context") is None


@pytest.mark.asyncio
async def test_successful_stream_clears_only_its_own_context(monkeypatch):
    manager = CorrelationIdManager()

    async def fail_parent_and_complete_child(**kwargs):
        if kwargs["request_id"] == "parent-request":
            raise RuntimeError("mock parent stream failure")
        return "mock child response"

    monkeypatch.setattr("agentlang.event.get_correlation_manager", lambda: manager)
    monkeypatch.setattr(
        StreamingCallProcessor,
        "call_with_stream",
        fail_parent_and_complete_child,
    )

    with pytest.raises(RuntimeError, match="mock parent stream failure"):
        await _execute_stream("mock-parent-context", "parent-request")
    assert await _execute_stream("mock-child-context", "child-request") == "mock child response"

    assert manager.pop_stream_fallback_cid("mock-parent-context") == "parent-request"
    assert manager.pop_stream_fallback_cid("mock-child-context") is None
