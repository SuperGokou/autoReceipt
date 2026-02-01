"""
Survey Bot - Automated Survey Completion System.

A complete pipeline for automating customer satisfaction surveys:
1. Extract survey URLs from receipt images (Ingestion Agent)
2. Configure response personas based on mood (Supervisor Agent)
3. Navigate and complete surveys automatically (Navigation Graph)
4. Extract coupon codes and send via email (Fulfillment Agent)

Quick Start:
    >>> from survey_bot import SurveyBot
    >>> 
    >>> bot = SurveyBot(verbose=True)
    >>> result = await bot.run(
    ...     image_path="receipt.jpg",
    ...     mood="happy",
    ...     email="user@example.com"
    ... )

CLI Usage:
    $ python -m survey_bot run -i receipt.jpg -m happy -e user@email.com

Modules:
    - agents: Core agent classes (Ingestion, Supervisor, Fulfillment)
    - browser: Browser automation (Launcher, Observer, Interactor)
    - llm: LLM integration (Navigation Graph, Decision Chains)
    - models: Data models (PageState, Persona, SurveyResult)
"""
from .main import SurveyBot, run_cli

__version__ = "0.1.0"
__author__ = "Survey Bot Team"

__all__ = [
    "SurveyBot",
    "run_cli",
    "__version__",
]
