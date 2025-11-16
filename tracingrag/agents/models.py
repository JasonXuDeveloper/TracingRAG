"""Data models for agent system

DEPRECATED: This module is kept for backwards compatibility.
All shared types have been moved to tracingrag.types
"""

# Re-export from centralized types module
from tracingrag.types import Citation, MemorySuggestion

__all__ = ["Citation", "MemorySuggestion"]
