"""
Advanced-mode campaign setup form.

A thin wrapper around ui/setup_simple.py's shared _render_setup_form() --
all the actual field logic lives there so simple and advanced mode never
drift out of sync. Advanced mode differs only in: per-categorical-parameter
encoding choice (instead of auto-picked OHE), and instructions for running
the campaign locally in your own IDE.
"""

from ui.setup_simple import _render_setup_form


def setup_advanced(campaign: dict) -> None:
    _render_setup_form(campaign, advanced=True)