import os

import pytest


for _key in tuple(os.environ):
    if _key.startswith("NEMOTRON_") or _key.startswith("LLM_"):
        os.environ.pop(_key, None)
os.environ["ASTRO_HELIX_SKIP_DOTENV"] = "1"


@pytest.fixture(autouse=True)
def isolate_llm_environment(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("NEMOTRON_") or key.startswith("LLM_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ASTRO_HELIX_SKIP_DOTENV", "1")
