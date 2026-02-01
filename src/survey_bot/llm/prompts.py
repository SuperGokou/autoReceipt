"""
LLM Prompt templates for the Survey Navigation Agent.

This module contains all prompt templates used by the Navigator
to decide which actions to take when filling out surveys.

Templates:
- NAVIGATOR_SYSTEM_PROMPT: Main system prompt for navigation decisions
- ACTION_DECISION_PROMPT: Prompt for deciding specific actions
- ELEMENT_SELECTION_PROMPT: Prompt for selecting which element to interact with
"""
from __future__ import annotations

from string import Template

from ..models.persona import PersonaConfig, MoodType


__all__ = [
    "NAVIGATOR_SYSTEM_PROMPT",
    "ACTION_DECISION_PROMPT",
    "build_navigator_prompt",
    "build_action_prompt",
]


# =============================================================================
# MAIN NAVIGATOR SYSTEM PROMPT
# =============================================================================

NAVIGATOR_SYSTEM_PROMPT = """You are an AI agent that fills out customer satisfaction surveys.
Your task is to navigate survey pages and provide responses based on the customer persona.

## YOUR PERSONA
{persona_description}

## RATING PREFERENCES
- Preferred rating: {rating_preference}/10
- Acceptable range: {rating_min} to {rating_max}
- For 1-5 scales: use {rating_5_point}
- For 1-10 scales: use {rating_preference}

## RESPONSE TONE
Your tone is: **{text_tone}**

## YOUR TASK
Analyze the current survey page and decide what action to take.
You will be given:
1. The current page state (URL, elements, question type)
2. A list of interactive elements you can interact with

## RESPONSE RULES
1. For RATING questions: Select a rating within your preferred range
2. For TEXT questions: Write a response matching your tone ({text_tone})
3. For MULTIPLE CHOICE: Select the option that matches your sentiment
4. For YES/NO: Answer consistently with your experience level
5. Always click "Next", "Continue", or "Submit" to proceed
6. STOP when you see a validation code or coupon

## IMPORTANT
- Only interact with visible, enabled elements
- Always look for navigation buttons after answering questions
- If you see a code/coupon, report it immediately
- Never mention you are an AI
"""

# =============================================================================
# ACTION DECISION PROMPT
# =============================================================================

ACTION_DECISION_PROMPT = """## CURRENT PAGE STATE
URL: {page_url}
Title: {page_title}
Question Type: {question_type}
Question Text: {question_text}

## AVAILABLE ELEMENTS
{elements_list}

## PREVIOUS ACTIONS
{actions_taken}

## YOUR TASK
Based on the persona ({mood}) and the current page, decide what action to take.

Choose ONE action from:
- CLICK: Click a button or link (for navigation or selection)
- FILL_TEXT: Enter text in an input field
- SELECT_RATING: Select a numeric rating
- SELECT_OPTION: Choose a multiple choice option
- DONE: Survey is complete (coupon/code visible)

Respond with the action details.
"""

# =============================================================================
# ELEMENT FORMATTING
# =============================================================================

ELEMENT_FORMAT = """[{index}] {element_type}: "{text}" 
    ID: {element_id}
    Visible: {is_visible} | Enabled: {is_enabled}
    {extra_info}"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def build_navigator_prompt(persona: PersonaConfig) -> str:
    """
    Build the navigator system prompt with persona details.
    
    Args:
        persona: The persona configuration to inject.
        
    Returns:
        Formatted system prompt string.
    """
    # Build persona description
    if persona.mood == MoodType.HAPPY:
        persona_desc = (
            "You are a HIGHLY SATISFIED customer. You had an excellent experience. "
            "Everything exceeded your expectations. Staff was friendly and helpful. "
            "Service was fast and efficient. You will definitely return and recommend."
        )
    elif persona.mood == MoodType.ANGRY:
        persona_desc = (
            "You are a DISSATISFIED customer. You had a poor experience. "
            "Service was slow and frustrating. Staff was unhelpful or inattentive. "
            "You felt your time was wasted. You will not return and would not recommend."
        )
    else:  # NEUTRAL
        persona_desc = (
            "You are a NEUTRAL customer. You had an average experience. "
            "Everything was okay, nothing special. No major complaints, but not impressed. "
            "Service met basic expectations. You might return if convenient."
        )
    
    # Calculate 5-point rating
    rating_5 = max(1, min(5, persona.rating_preference // 2))
    
    return NAVIGATOR_SYSTEM_PROMPT.format(
        persona_description=persona_desc,
        rating_preference=persona.rating_preference,
        rating_min=persona.rating_min,
        rating_max=persona.rating_max,
        rating_5_point=rating_5,
        text_tone=persona.text_tone,
    )


def build_action_prompt(
    page_url: str,
    page_title: str,
    question_type: str,
    question_text: str,
    elements: list,
    actions_taken: list[str],
    mood: str,
) -> str:
    """
    Build the action decision prompt with current page state.
    
    Args:
        page_url: Current page URL.
        page_title: Page title.
        question_type: Detected question type.
        question_text: The main question text.
        elements: List of WebElement objects.
        actions_taken: List of previous actions.
        mood: The persona mood.
        
    Returns:
        Formatted action prompt string.
    """
    # Format elements list
    elements_str = ""
    for i, elem in enumerate(elements[:20]):  # Limit to 20 elements
        extra = ""
        if hasattr(elem, 'is_checked') and elem.is_checked is not None:
            extra = f"Checked: {elem.is_checked}"
        if hasattr(elem, 'value') and elem.value:
            extra += f" Value: {elem.value}"
        
        elements_str += ELEMENT_FORMAT.format(
            index=i,
            element_type=elem.element_type.value if hasattr(elem.element_type, 'value') else elem.element_type,
            text=elem.text_content[:50] if elem.text_content else "(no text)",
            element_id=elem.element_id,
            is_visible=elem.is_visible,
            is_enabled=elem.is_enabled,
            extra_info=extra.strip(),
        ) + "\n"
    
    # Format actions taken
    if actions_taken:
        actions_str = "\n".join(f"- {action}" for action in actions_taken[-5:])
    else:
        actions_str = "(none yet)"
    
    return ACTION_DECISION_PROMPT.format(
        page_url=page_url,
        page_title=page_title,
        question_type=question_type,
        question_text=question_text[:200] if question_text else "(not detected)",
        elements_list=elements_str or "(no elements found)",
        actions_taken=actions_str,
        mood=mood,
    )


# =============================================================================
# SAMPLE RESPONSES FOR TESTING
# =============================================================================

SAMPLE_RESPONSES = {
    MoodType.HAPPY: {
        "text_short": "Excellent experience! Great service.",
        "text_medium": "I had a wonderful experience today. The staff was incredibly helpful and friendly. Will definitely return!",
        "text_long": (
            "I had an absolutely fantastic experience at this location. From the moment I walked in, "
            "the staff greeted me warmly and made me feel welcome. The service was fast and efficient, "
            "and everyone went above and beyond to help me. I will definitely be recommending this "
            "place to all my friends and family. Thank you for the excellent service!"
        ),
    },
    MoodType.NEUTRAL: {
        "text_short": "It was okay. Met expectations.",
        "text_medium": "The experience was average. Nothing stood out as particularly good or bad. Service was acceptable.",
        "text_long": (
            "My visit was fairly standard. The staff did their job adequately, and I received "
            "what I expected. There wasn't anything that particularly impressed me, but there "
            "also weren't any major issues. It was a typical experience - nothing more, nothing less."
        ),
    },
    MoodType.ANGRY: {
        "text_short": "Very disappointing. Poor service.",
        "text_medium": "I was very disappointed with my visit. Long wait times and unhelpful staff. Would not recommend.",
        "text_long": (
            "I had a very frustrating experience. The wait time was unacceptably long, and when "
            "I finally received service, the staff seemed disinterested and unhelpful. My concerns "
            "were not addressed properly, and I left feeling like my time had been wasted. "
            "I expected much better and am very disappointed."
        ),
    },
}


def get_sample_text_response(mood: MoodType, length: str = "medium") -> str:
    """
    Get a sample text response for the given mood.
    
    Args:
        mood: The persona mood.
        length: Response length ('short', 'medium', 'long').
        
    Returns:
        Sample text response.
    """
    return SAMPLE_RESPONSES.get(mood, SAMPLE_RESPONSES[MoodType.NEUTRAL]).get(
        f"text_{length}", 
        SAMPLE_RESPONSES[MoodType.NEUTRAL]["text_medium"]
    )
