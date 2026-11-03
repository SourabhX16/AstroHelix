"""LLM construction for configured and deterministic providers."""
from langchain_openai import ChatOpenAI

from agents.config import LLMConfig
from agents.scripted_llm import ScriptedChatModel


def make_llm(config: LLMConfig):
    if config.provider == "scripted":
        return ScriptedChatModel()
    if config.provider == "openai_compat":
        return ChatOpenAI(**config.to_llm_kwargs())
    raise ValueError(f"Unsupported LLM provider: {config.provider}")
