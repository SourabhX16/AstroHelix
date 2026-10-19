"""
Astro Helix Agents — Multi-agent collision avoidance orchestrator.

Uses an OpenAI-compatible LLM endpoint with LangGraph for multi-agent
tool-calling workflows.

Architecture:
  ┌─────────┐     ┌──────────┐     ┌─────────┐     ┌────────┐     ┌───────┐
  │  Scout  │ ──▶ │ Analyst  │ ──▶ │ Planner │ ──▶ │ Safety │ ──▶ │  Ops  │
  └─────────┘     └──────────┘     └─────────┘     └────────┘     └───────┘
     scan &          assess           propose         check          produce
     triage          risk             maneuvers       constraints    brief
"""
