"""
Survey Bot Web Application.

Provides a Flask-based web UI for the Survey Bot.

Usage:
    python -m survey_bot.web.app
"""
from .app import app

__all__ = ["app"]
