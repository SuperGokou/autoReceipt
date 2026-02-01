"""
Agent modules for the Survey Automation Bot.

This package contains the four core agents:
- IngestionAgent: Extracts survey URLs from receipt images (Module 1)
- SupervisorAgent: Manages personas and generates prompts (Module 2)
- NavigatorAgent: Navigates and fills out surveys (Module 3) - Coming Soon
- FulfillmentAgent: Extracts coupons and sends emails (Module 4)
"""
from .fulfillment import FulfillmentAgent
from .ingestion import IngestionAgent
from .supervisor import SupervisorAgent

__all__ = [
    "FulfillmentAgent",
    "IngestionAgent",
    "SupervisorAgent",
]