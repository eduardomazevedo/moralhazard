"""
Public exports for the Moral Hazard module (v0).

This package surfaces:
  - MoralHazardProblem: public class to configure the model and solve the dual
  - SolveResults: immutable dataclass for solver outputs

Spec: "Moral Hazard — Minimal Interface (v0)". See docs. 
"""

from .types import SolveResults
from .problem import MoralHazardProblem

__all__ = ["MoralHazardProblem", "SolveResults"]
