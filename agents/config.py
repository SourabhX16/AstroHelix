"""
LLM configuration for the Astro Helix agent system.

Supports three modes:
  1. LOCAL  — OpenAI-compatible local endpoint (target deployment)
  2. NIM    — NVIDIA API Catalog / NIM endpoint (fallback)
  3. OPENAI — OpenAI-compatible endpoint (dev/testing)

The optional scripts/setup_gx10.sh helper describes one possible future-scope
on-board deployment on GX10 / DGX Spark-class hardware. It is not required
for the software simulation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


@dataclass
class LLMConfig:
    """Configuration for the LLM backend."""
    # Endpoint
    base_url: str = "http://localhost:8001/v1"
    api_key: str = "not-needed"
    model: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"

    # Generation parameters
    temperature: float = 0.3          # low for deterministic tool-calling
    max_tokens: int = 2048
    top_p: float = 0.95

    # Agent parameters
    max_iterations: int = 10          # max tool-call loops per agent
    recursion_limit: int = 50         # LangGraph recursion limit
    provider: Literal["openai_compat", "scripted"] = "openai_compat"
    disable_thinking: bool = True

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Build config from environment variables."""
        return cls(
            base_url=os.getenv("NEMOTRON_BASE_URL", "http://localhost:8001/v1"),
            api_key=os.getenv("NEMOTRON_API_KEY", "not-needed"),
            model=os.getenv("NEMOTRON_MODEL", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"),
            temperature=float(os.getenv("NEMOTRON_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("NEMOTRON_MAX_TOKENS", "2048")),
            provider=os.getenv("LLM_PROVIDER", "openai_compat"),
            disable_thinking=os.getenv("NEMOTRON_DISABLE_THINKING", "true").lower() in {"1", "true", "yes", "on"},
        )

    def to_llm_kwargs(self) -> dict:
        """Return kwargs for ChatOpenAI constructor.

        max_tokens is omitted so vLLM automatically uses whatever
        context-window space remains after the input tokens.  This
        avoids 400 errors when accumulated agent context is large.
        """
        kwargs = {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model": self.model,
            "temperature": self.temperature,
        }
        if self.disable_thinking:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        return kwargs


# Quick presets
LOCAL_GX10 = LLMConfig(
    base_url="http://localhost:8001/v1",
    model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
)

NVIDIA_NIM = LLMConfig(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY", ""),
    model="nvidia/nvidia-nemotron-3-nano-30b-a3b-bf16",
)

# For development/testing when no GPU is available
OPENAI_FALLBACK = LLMConfig(
    base_url="https://api.openai.com/v1",
    api_key=os.getenv("OPENAI_API_KEY", ""),
    model="gpt-4o-mini",
)
