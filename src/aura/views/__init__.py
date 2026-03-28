"""Agent-type environment views for AURA."""

from aura.views.base import EnvironmentView, ViewConfig, ViewRegistry
from aura.views.coder import CoderView
from aura.views.researcher import ResearcherView
from aura.views.sysadmin import SysadminView

__all__ = [
    "EnvironmentView",
    "ViewConfig",
    "ViewRegistry",
    "CoderView",
    "SysadminView",
    "ResearcherView",
]
