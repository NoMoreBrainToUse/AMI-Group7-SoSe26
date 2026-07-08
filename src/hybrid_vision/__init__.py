"""Hybrid RGB + event drone detection — AMI 2026 Group 7.

Public surface:
    PipelineConfig  — every path and threshold in one dataclass
    run_sequence    — full pipeline for one FRED sequence directory
"""

from .config import PipelineConfig
from .pipeline import run_sequence

__all__ = ["PipelineConfig", "run_sequence"]
