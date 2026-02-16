"""Multi-agent architecture — divide and conquer.

Each agent owns one responsibility. The OrchestratorAgent coordinates them.
"""

from pop_validation.agents.base import AgentResult, BaseAgent
from pop_validation.agents.extraction_agent import ExtractionAgent
from pop_validation.agents.image_loader_agent import ImageLoaderAgent
from pop_validation.agents.orchestrator import OrchestratorAgent
from pop_validation.agents.quality_agent import QualityAgent
from pop_validation.agents.reporting_agent import ReportingAgent
from pop_validation.agents.validation_agent import ValidationAgent

__all__ = [
    "AgentResult",
    "BaseAgent",
    "ExtractionAgent",
    "ImageLoaderAgent",
    "OrchestratorAgent",
    "QualityAgent",
    "ReportingAgent",
    "ValidationAgent",
]
