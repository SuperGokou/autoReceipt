"""
LLM integration for the Survey Automation Bot.

This package provides LangChain/LangGraph-based decision making:
- prompts: System and action prompts for the Navigator
- chains: LangChain chains for structured decision output
- graph: LangGraph state machine for navigation loop

The LLM components enable intelligent survey navigation by:
1. Understanding page context
2. Deciding appropriate actions based on persona
3. Managing the observe → decide → act loop

Example Usage:
    >>> from survey_bot.llm import (
    ...     NavigationGraph,
    ...     create_navigation_graph,
    ...     get_llm,
    ...     ActionType,
    ... )
    >>> 
    >>> # Create LLM (optional - works without it using rules)
    >>> llm = get_llm(provider="openai", model="gpt-4o-mini")
    >>> 
    >>> # Create navigation graph
    >>> graph = create_navigation_graph(persona, llm=llm)
    >>> 
    >>> # Run survey
    >>> result = await graph.run(page, survey_url)
    >>> print(f"Coupon: {result['coupon_code']}")
"""

# Prompts
from .prompts import (
    NAVIGATOR_SYSTEM_PROMPT,
    ACTION_DECISION_PROMPT,
    build_navigator_prompt,
    build_action_prompt,
    get_sample_text_response,
)

# Chains
from .chains import (
    ActionType,
    NavigationAction,
    NavigationDecision,
    create_decision_chain,
    get_llm,
    make_decision,
    SimpleLLM,
    LANGCHAIN_AVAILABLE,
)

# Graph
from .graph import (
    SurveyState,
    NavigationGraph,
    create_navigation_graph,
    run_survey_navigation,
    create_initial_state,
    LANGGRAPH_AVAILABLE,
)

__all__ = [
    # Prompts
    "NAVIGATOR_SYSTEM_PROMPT",
    "ACTION_DECISION_PROMPT",
    "build_navigator_prompt",
    "build_action_prompt",
    "get_sample_text_response",
    
    # Chains
    "ActionType",
    "NavigationAction",
    "NavigationDecision",
    "create_decision_chain",
    "get_llm",
    "make_decision",
    "SimpleLLM",
    "LANGCHAIN_AVAILABLE",
    
    # Graph
    "SurveyState",
    "NavigationGraph",
    "create_navigation_graph",
    "run_survey_navigation",
    "create_initial_state",
    "LANGGRAPH_AVAILABLE",
]
