"""Engine: strategy-agnostic runtime for baseline_source campaigns."""
from .analyzer import Analyzer
from .executor import Executor
from .runner import run_campaign
from .state_manager import StateManager

__all__ = ["Analyzer", "Executor", "StateManager", "run_campaign"]



