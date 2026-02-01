"""
Supervisor Agent for managing survey response personas.

This agent is responsible for:
1. Parsing user mood input (emoji, text, star ratings)
2. Returning the appropriate persona configuration
3. Generating system prompts for the Navigation Agent
4. Providing response templates for different question types

The Supervisor acts as the "director" that tells the Navigation Agent
how to behave when answering survey questions.

Example Usage:
    >>> from survey_bot.agents.supervisor import SupervisorAgent
    >>>
    >>> supervisor = SupervisorAgent()
    >>>
    >>> # Get persona from emoji
    >>> persona = supervisor.get_persona("😊")
    >>> print(persona.mood)  # MoodType.HAPPY
    >>>
    >>> # Generate system prompt for Navigator
    >>> prompt = supervisor.generate_system_prompt(persona)
    >>>
    >>> # Get response for specific question type
    >>> response = supervisor.get_response_for_question("rating", persona)
    >>> print(response)  # "10"
"""
from __future__ import annotations

import logging
import random
from typing import Literal

from ..models.persona import (
    MoodType,
    PersonaConfig,
    PERSONA_PRESETS,
)

__all__ = ["SupervisorAgent"]

logger = logging.getLogger(__name__)

# Question type aliases for flexible matching
QUESTION_TYPE_ALIASES: dict[str, list[str]] = {
    "rating": [
        "rating", "rate", "scale", "score", "stars", "points",
        "1-10", "1-5", "satisfaction_rating", "nps",
    ],
    "text": [
        "text", "comment", "feedback", "suggestion", "describe",
        "explain", "tell_us", "write", "textarea", "open_ended",
        "improvement", "additional_comments",
    ],
    "multiple_choice": [
        "multiple_choice", "select", "choose", "option", "radio",
        "dropdown", "experience", "satisfaction", "likelihood",
    ],
    "yes_no": [
        "yes_no", "yes/no", "boolean", "true_false", "binary",
        "checkbox", "did_you", "was_the", "were_you",
    ],
}


class SupervisorAgent:
    """
    Agent responsible for persona management and prompt generation.

    The Supervisor is the "brain" that decides how the bot should
    behave when filling out surveys. It translates user mood input
    into specific behavioral guidelines for the Navigation Agent.

    Attributes:
        personas: Dictionary mapping MoodType to PersonaConfig.
        default_mood: Fallback mood when input cannot be parsed.
    """

    def __init__(self, default_mood: MoodType = MoodType.NEUTRAL) -> None:
        """
        Initialize the Supervisor Agent.

        Args:
            default_mood: Fallback mood for unrecognized inputs.
        """
        self.personas = PERSONA_PRESETS
        self.default_mood = default_mood
        logger.debug(f"SupervisorAgent initialized with default mood: {default_mood}")

    def get_persona(self, mood_input: str) -> PersonaConfig:
        """
        Get the persona configuration for a given mood input.

        Parses various input formats including:
        - Emoji: 😊, 😐, 😡
        - Star ratings: ⭐⭐⭐⭐⭐, ⭐⭐⭐, ⭐
        - Text: happy, satisfied, angry, neutral
        - Numeric: 10, 5, 1

        Args:
            mood_input: User's mood selection in any supported format.

        Returns:
            PersonaConfig matching the detected mood.

        Example:
            >>> supervisor = SupervisorAgent()
            >>> persona = supervisor.get_persona("😊")
            >>> print(persona.rating_preference)  # 10
        """
        try:
            mood = MoodType.from_input(mood_input)
            logger.info(f"Parsed mood input '{mood_input}' → {mood.value}")
        except Exception as e:
            logger.warning(f"Failed to parse mood '{mood_input}': {e}, using default")
            mood = self.default_mood

        return self.personas[mood]

    def generate_system_prompt(self, persona: PersonaConfig) -> str:
        """
        Generate a detailed system prompt for the Navigation Agent
        Creates a comprehensive prompt that instructs the LLM on how

        to fill out survey questions based on the persona's characteristics.

        Args:
            persona: The persona configuration to base the prompt on.

        Returns:
            Complete system prompt string for the Navigator LLM.

        Example:
            >>> supervisor = SupervisorAgent()
            >>> persona = supervisor.get_persona("😊")
            >>> prompt = supervisor.generate_system_prompt(persona)
            >>> # Returns detailed prompt with rating preferences, tone, etc.
        """
        # Get mood-specific descriptors
        mood_descriptors = {
            MoodType.HAPPY: {
                "experience": "EXCELLENT",
                "satisfaction": "highly satisfied",
                "adjectives": "fantastic, wonderful, exceptional",
                "will_return": "definitely will return and recommend",
            },
            MoodType.NEUTRAL: {
                "experience": "AVERAGE",
                "satisfaction": "moderately satisfied",
                "adjectives": "okay, acceptable, decent",
                "will_return": "might return if convenient",
            },
            MoodType.ANGRY: {
                "experience": "POOR",
                "satisfaction": "very dissatisfied",
                "adjectives": "disappointing, frustrating, unacceptable",
                "will_return": "unlikely to return",
            },
        }

        desc = mood_descriptors[persona.mood]

        # Build the prompt
        prompt = f"""You are an AI assistant helping to fill out a customer satisfaction survey.

## YOUR PERSONA
You are acting as a customer who had a **{desc['experience']}** experience.
You are **{desc['satisfaction']}** with your visit.
Overall sentiment: {persona.mood.value.upper()}

## RATING PREFERENCES
- For 1-10 scale questions: Select **{persona.rating_preference}** (range: {persona.rating_min}-{persona.rating_max})
- For 1-5 scale questions: Select **{persona.rating_preference // 2 or 1}**
- For 1-5 star ratings: Select **{min(5, max(1, persona.rating_preference // 2))}** stars

## RESPONSE TONE
Your tone should be: **{persona.text_tone}**
Use words like: {desc['adjectives']}
Future behavior: {desc['will_return']}

## RESPONSE GUIDELINES BY QUESTION TYPE

### Rating Questions (numeric scales)
- Preferred rating: {persona.rating_preference}/10
- Acceptable range: {persona.rating_min} to {persona.rating_max}
- Always pick a number within your range

### Text/Comment Questions
- Write {1 if persona.mood == MoodType.ANGRY else 2}-3 sentences
- Tone: {persona.text_tone}
- Sample response: "{persona.sample_responses.get('text', 'Thank you.')[:100]}..."

### Multiple Choice Questions
- Select the option that best matches your {persona.mood.value} experience
- Satisfaction: Look for "{persona.sample_responses.get('satisfaction', 'Satisfied')}"
- Likelihood: Look for "{persona.sample_responses.get('likelihood', 'Likely')}"

### Yes/No Questions
- For positive experience questions (Was staff helpful?): {"Yes" if persona.mood == MoodType.HAPPY else "No" if persona.mood == MoodType.ANGRY else "Yes"}
- For negative experience questions (Were there problems?): {"No" if persona.mood == MoodType.HAPPY else "Yes" if persona.mood == MoodType.ANGRY else "No"}

## IMPORTANT RULES
1. Stay in character as a {persona.mood.value} customer throughout
2. Be consistent with your ratings across all questions
3. Keep text responses concise but meaningful
4. For "Next" or "Continue" buttons, always click them to proceed
5. Look for the final coupon/validation code at the end
6. Never mention that you are an AI or bot

## NAVIGATION INSTRUCTIONS
- Identify clickable elements (buttons, radio buttons, checkboxes)
- For each question, determine the type and select the appropriate response
- Click "Next", "Continue", or "Submit" buttons to proceed
- Stop when you see a validation code or coupon code
"""

        logger.debug(f"Generated system prompt for {persona.mood.value} persona")
        return prompt

    def get_response_for_question(
            self,
            question_type: str,
            persona: PersonaConfig,
            *,
            randomize: bool = False,
    ) -> str:
        """
        Get an appropriate response template for a question type.

        Maps question types to pre-defined response templates from
        the persona configuration.

        Args:
            question_type: Type of survey question.
            persona: The active persona configuration.
            randomize: If True, add slight variation to numeric responses.

        Returns:
            Response template string appropriate for the question type.

        Example:
            >>> supervisor = SupervisorAgent()
            >>> persona = supervisor.get_persona("😊")
            >>> response = supervisor.get_response_for_question("rating", persona)
            >>> print(response)  # "10"
        """
        # Normalize question type
        normalized_type = self._normalize_question_type(question_type)
        logger.debug(f"Normalized question type '{question_type}' → '{normalized_type}'")

        # Handle rating with optional randomization
        if normalized_type == "rating":
            if randomize:
                rating = random.randint(persona.rating_min, persona.rating_max)
                return str(rating)
            return str(persona.rating_preference)

        # Look up in sample responses
        response = persona.sample_responses.get(normalized_type)

        # Try original question_type if normalized not found
        if response is None:
            response = persona.sample_responses.get(question_type)

        # Fall back to generic
        if response is None:
            response = persona.sample_responses.get("generic", "Thank you.")
            logger.debug(f"No template for '{question_type}', using generic")

        return response

    def _normalize_question_type(self, question_type: str) -> str:
        """
        Normalize a question type to a standard category.

        Args:
            question_type: Raw question type string.

        Returns:
            Normalized question type category.
        """
        normalized = question_type.lower().strip().replace(" ", "_")

        # Check against aliases
        for category, aliases in QUESTION_TYPE_ALIASES.items():
            if normalized in aliases or any(alias in normalized for alias in aliases):
                return category

        # Return as-is if no match
        return normalized

    def get_rating_for_scale(
            self,
            persona: PersonaConfig,
            max_value: int = 10,
            *,
            randomize: bool = False,
    ) -> int:
        """
        Get an appropriate rating for a given scale.

        Scales the persona's preference to match different rating scales
        (e.g., 1-5, 1-10, 1-7).

        Args:
            persona: The active persona configuration.
            max_value: Maximum value of the rating scale.
            randomize: If True, add slight variation.

        Returns:
            Integer rating appropriate for the scale.

        Example:
            >>> supervisor = SupervisorAgent()
            >>> persona = supervisor.get_persona("😊")  # rating_preference = 10
            >>> supervisor.get_rating_for_scale(persona, max_value=5)  # Returns 5
            >>> supervisor.get_rating_for_scale(persona, max_value=7)  # Returns 7
        """
        # Scale the preference to the target range
        base_rating = round(persona.rating_preference * max_value / 10)

        if randomize:
            # Calculate scaled min/max
            scaled_min = round(persona.rating_min * max_value / 10)
            scaled_max = round(persona.rating_max * max_value / 10)
            return random.randint(max(1, scaled_min), min(max_value, scaled_max))

        return max(1, min(max_value, base_rating))

    def get_text_response(
            self,
            persona: PersonaConfig,
            question_context: str = "",
            max_length: int = 200,
    ) -> str:
        """
        Get a text response tailored to the persona and context.

        Args:
            persona: The active persona configuration.
            question_context: Optional context about the question.
            max_length: Maximum response length in characters.

        Returns:
            Text response string.
        """
        # Get base response
        base_response = persona.sample_responses.get("text", "")

        # Truncate if needed
        if len(base_response) > max_length:
            base_response = base_response[:max_length - 3] + "..."

        return base_response

    def summarize_persona(self, persona: PersonaConfig) -> dict:
        """
        Get a summary of the persona for logging/debugging.

        Args:
            persona: The persona to summarize.

        Returns:
            Dictionary with key persona attributes.
        """
        return {
            "mood": persona.mood.value,
            "rating_preference": persona.rating_preference,
            "rating_range": f"{persona.rating_min}-{persona.rating_max}",
            "text_tone": persona.text_tone,
            "response_types_available": list(persona.sample_responses.keys()),
        }