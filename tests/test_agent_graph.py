import json
from typing import ClassVar
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from agents.config import LLMConfig
from agents.graph import _create_agent_node


class RepeatingToolModel(BaseChatModel):
    bound_tool_names: list[str] = []
    observations: ClassVar[list[str]] = []

    @property
    def _llm_type(self):
        return "repeating-tool-test"

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(update={"bound_tool_names": [t.name for t in tools]})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if not self.bound_tool_names:
            answer = "Final report based on the tool result."
            response = AIMessage(content=answer)
        else:
            for message in messages:
                if isinstance(message, ToolMessage) and "duplicate call, result unchanged" in message.content:
                    self.observations.append(message.content)
            response = AIMessage(content="", tool_calls=[{
                "name": "counted_tool", "args": {"value": 7}, "id": str(uuid4()), "type": "tool_call"
            }])
        return ChatResult(generations=[ChatGeneration(message=response)])


def _initial_state():
    return {
        "messages": [HumanMessage(content="test")], "current_agent": "test",
        "scout_output": None, "analyst_output": None, "planner_output": None,
        "safety_output": None, "ops_brief": None, "events": [], "pipeline_error": False,
    }


def test_iteration_cap_forces_text_report_and_emits_event():
    calls = []

    @tool
    def counted_tool(value: int) -> str:
        """Count a test tool invocation."""
        calls.append(value)
        return json.dumps({"value": value})

    model = RepeatingToolModel()
    model.observations.clear()
    node = _create_agent_node(model, LLMConfig(provider="scripted", max_iterations=2), "system", [counted_tool], "test_agent", "ops_brief")
    result = node(_initial_state())
    assert result["ops_brief"] == "Final report based on the tool result."
    assert any(event["type"] == "iteration_limit" and event["agent"] == "test_agent" for event in result["events"])
    assert calls == [7]


def test_duplicate_tool_call_reuses_previous_result():
    calls = []

    @tool
    def counted_tool(value: int) -> str:
        """Count a test tool invocation."""
        calls.append(value)
        return json.dumps({"value": value})

    model = RepeatingToolModel()
    model.observations.clear()
    node = _create_agent_node(model, LLMConfig(provider="scripted", max_iterations=2), "system", [counted_tool], "test_agent", "ops_brief")
    result = node(_initial_state())
    assert calls == [7]
    assert any("duplicate call, result unchanged" in event.get("summary", "") for event in result["events"])
