"""The screens of the workbench, in the order the sidebar lists them.

Screens are grouped into four files rather than one file each: the data screens
share loader/cleaner helpers, the modelling screens share the training jobs, and
the analysis screens share the "best combination" machinery.  Splitting them one
per screen would have produced thirteen files with almost no content each and a
lot of repeated imports.
"""
from __future__ import annotations

from .analysis_steps import AutoPage, ClusterPage, OptimizePage, PlotsPage
from .data_steps import CleanPage, EdaPage, FeaturesPage, LoadPage
from .engineering_steps import (
    CurveFitPage, DoePage, StatisticsPage, TimeSeriesPage, UnitsPage,
)
from .model_steps import EvaluatePage, ModelPage, PredictPage, SavePage, TunePage
from .reliability_steps import ReliabilityPage
from .start_steps import StartPage

#: (sidebar group, pages) — the order here is the order on screen.
STEP_GROUPS = [
    ("Start", [StartPage]),
    ("Data", [LoadPage, EdaPage, CleanPage, FeaturesPage]),
    ("Model", [ModelPage, TunePage, EvaluatePage, SavePage, PredictPage]),
    ("Analyse", [OptimizePage, ClusterPage, PlotsPage]),
    ("Engineering", [StatisticsPage, CurveFitPage, DoePage, TimeSeriesPage,
                     UnitsPage, ReliabilityPage]),
    ("Finish", [AutoPage]),
]


def build_steps(state, runner, window=None):
    """Instantiate every page.  Returns ``(all_pages, groups)``."""
    pages, groups = [], []
    for group, classes in STEP_GROUPS:
        instances = [cls(state, runner, window) for cls in classes]
        groups.append((group, instances))
        pages.extend(instances)
    return pages, groups


__all__ = [
    "STEP_GROUPS", "build_steps",
    "LoadPage", "EdaPage", "CleanPage", "FeaturesPage",
    "ModelPage", "TunePage", "EvaluatePage", "SavePage", "PredictPage",
    "OptimizePage", "ClusterPage", "PlotsPage", "AutoPage",
]
