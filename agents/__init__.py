"""Agents package for the Multi-Agent AI Academic Assistant."""

from agents.tutor_agent import create_tutor_agent
from agents.analytics_agent import create_analytics_agent
from agents.support_agent import create_support_agent

__all__ = [
    "create_tutor_agent",
    "create_analytics_agent",
    "create_support_agent",
]
