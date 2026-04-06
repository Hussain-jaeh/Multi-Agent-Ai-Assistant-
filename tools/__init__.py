"""Tools package for the Multi-Agent AI Academic Assistant."""

from tools.document_tools import DocumentSearchTool, PolicySearchTool, index_documents
from tools.analytics_tools import PerformanceAnalysisTool, ProgressTrackerTool, update_student_progress

__all__ = [
    "DocumentSearchTool",
    "PolicySearchTool",
    "index_documents",
    "PerformanceAnalysisTool",
    "ProgressTrackerTool",
    "update_student_progress",
]
