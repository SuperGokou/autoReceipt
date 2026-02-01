"""
Test suite for the Supervisor Agent (Module 2).

This module tests:
- Mood parsing from various input formats (emoji, text, stars)
- Persona configuration retrieval
- System prompt generation
- Response template selection
- Rating scale conversion

Run with: pytest tests/test_supervisor.py -v
"""
from __future__ import annotations

import pytest

from src.survey_bot.agents.supervisor import SupervisorAgent
from src.survey_bot.models.persona import (
    MoodType,
    PersonaConfig,
    PERSONA_PRESETS,
    get_persona_from_input,
)


class TestMoodParsing:
    """Test suite for mood input parsing."""

    def test_happy_emoji_maps_to_happy_persona(self):
        """Test that 😊 emoji maps to HAPPY mood."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("😊")

        assert persona.mood == MoodType.HAPPY
        assert persona.rating_preference >= 9
        assert persona.text_tone == "enthusiastic"

    def test_angry_emoji_maps_to_angry_persona(self):
        """Test that 😡 emoji maps to ANGRY mood."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("😡")

        assert persona.mood == MoodType.ANGRY
        assert persona.rating_preference <= 3
        assert persona.text_tone == "frustrated"

    def test_neutral_emoji_maps_to_neutral_persona(self):
        """Test that 😐 emoji maps to NEUTRAL mood."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("😐")

        assert persona.mood == MoodType.NEUTRAL
        assert 5 <= persona.rating_preference <= 7
        assert persona.text_tone == "neutral"

    def test_star_rating_five_stars_maps_to_happy(self):
        """Test that ⭐⭐⭐⭐⭐ maps to HAPPY mood."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("⭐⭐⭐⭐⭐")

        assert persona.mood == MoodType.HAPPY

    def test_star_rating_three_stars_maps_to_neutral(self):
        """Test that ⭐⭐⭐ maps to NEUTRAL mood."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("⭐⭐⭐")

        assert persona.mood == MoodType.NEUTRAL

    def test_star_rating_one_star_maps_to_angry(self):
        """Test that ⭐ maps to ANGRY mood."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("⭐")

        assert persona.mood == MoodType.ANGRY

    def test_text_happy_maps_correctly(self):
        """Test that text 'happy' maps to HAPPY mood."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("happy")

        assert persona.mood == MoodType.HAPPY

    def test_text_satisfied_maps_to_happy(self):
        """Test that 'satisfied' maps to HAPPY mood."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("satisfied")

        assert persona.mood == MoodType.HAPPY

    def test_text_angry_maps_correctly(self):
        """Test that text 'angry' maps to ANGRY mood."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("angry")

        assert persona.mood == MoodType.ANGRY

    def test_text_dissatisfied_maps_to_angry(self):
        """Test that 'dissatisfied' maps to ANGRY mood."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("dissatisfied")

        assert persona.mood == MoodType.ANGRY

    def test_text_neutral_maps_correctly(self):
        """Test that text 'neutral' maps to NEUTRAL mood."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("neutral")

        assert persona.mood == MoodType.NEUTRAL

    def test_text_okay_maps_to_neutral(self):
        """Test that 'okay' maps to NEUTRAL mood."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("okay")

        assert persona.mood == MoodType.NEUTRAL

    def test_unknown_input_defaults_to_neutral(self):
        """Test that unrecognized input defaults to NEUTRAL mood."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("asdfgh")

        assert persona.mood == MoodType.NEUTRAL

    def test_empty_string_defaults_to_neutral(self):
        """Test that empty string defaults to NEUTRAL mood."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("")

        assert persona.mood == MoodType.NEUTRAL

    def test_numeric_10_maps_to_happy(self):
        """Test that '10' or '10/10' maps to HAPPY mood."""
        supervisor = SupervisorAgent()

        persona1 = supervisor.get_persona("10")
        persona2 = supervisor.get_persona("10/10")

        assert persona1.mood == MoodType.HAPPY
        assert persona2.mood == MoodType.HAPPY

    def test_numeric_1_maps_to_angry(self):
        """Test that '1' or '1/10' maps to ANGRY mood."""
        supervisor = SupervisorAgent()

        persona1 = supervisor.get_persona("1")
        persona2 = supervisor.get_persona("1/10")

        assert persona1.mood == MoodType.ANGRY
        assert persona2.mood == MoodType.ANGRY

    def test_case_insensitive_parsing(self):
        """Test that mood parsing is case-insensitive."""
        supervisor = SupervisorAgent()

        assert supervisor.get_persona("HAPPY").mood == MoodType.HAPPY
        assert supervisor.get_persona("Happy").mood == MoodType.HAPPY
        assert supervisor.get_persona("hApPy").mood == MoodType.HAPPY

    def test_whitespace_handling(self):
        """Test that whitespace is handled correctly."""
        supervisor = SupervisorAgent()

        persona = supervisor.get_persona("  happy  ")
        assert persona.mood == MoodType.HAPPY


class TestSystemPromptGeneration:
    """Test suite for system prompt generation."""

    def test_system_prompt_contains_mood_keywords_happy(self):
        """Test that HAPPY persona prompt contains positive keywords."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("😊")
        prompt = supervisor.generate_system_prompt(persona)

        prompt_lower = prompt.lower()

        # Should contain positive indicators
        assert any(word in prompt_lower for word in [
            "excellent", "satisfied", "positive", "happy"
        ])

        # Should mention high ratings
        assert "9" in prompt or "10" in prompt

    def test_system_prompt_contains_mood_keywords_angry(self):
        """Test that ANGRY persona prompt contains negative keywords."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("😡")
        prompt = supervisor.generate_system_prompt(persona)

        prompt_lower = prompt.lower()

        # Should contain negative indicators
        assert any(word in prompt_lower for word in [
            "poor", "dissatisfied", "negative", "angry", "frustrated"
        ])

        # Should mention low ratings
        assert "1" in prompt or "2" in prompt or "3" in prompt

    def test_system_prompt_contains_mood_keywords_neutral(self):
        """Test that NEUTRAL persona prompt contains balanced keywords."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("😐")
        prompt = supervisor.generate_system_prompt(persona)

        prompt_lower = prompt.lower()

        # Should contain neutral indicators
        assert any(word in prompt_lower for word in [
            "average", "neutral", "okay", "moderate"
        ])

    def test_system_prompt_includes_rating_preference(self):
        """Test that prompt includes the persona's rating preference."""
        supervisor = SupervisorAgent()

        for mood in MoodType:
            persona = PERSONA_PRESETS[mood]
            prompt = supervisor.generate_system_prompt(persona)

            assert str(persona.rating_preference) in prompt

    def test_system_prompt_includes_tone(self):
        """Test that prompt includes the text tone."""
        supervisor = SupervisorAgent()

        for mood in MoodType:
            persona = PERSONA_PRESETS[mood]
            prompt = supervisor.generate_system_prompt(persona)

            assert persona.text_tone in prompt.lower()

    def test_system_prompt_includes_navigation_instructions(self):
        """Test that prompt includes navigation guidance."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("😊")
        prompt = supervisor.generate_system_prompt(persona)

        prompt_lower = prompt.lower()

        assert any(word in prompt_lower for word in [
            "next", "continue", "submit", "click"
        ])

    def test_system_prompt_mentions_coupon(self):
        """Test that prompt mentions looking for coupon/validation code."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("😊")
        prompt = supervisor.generate_system_prompt(persona)

        prompt_lower = prompt.lower()

        assert any(word in prompt_lower for word in [
            "coupon", "validation", "code"
        ])

    def test_system_prompt_is_string(self):
        """Test that generated prompt is a non-empty string."""
        supervisor = SupervisorAgent()

        for mood in MoodType:
            persona = PERSONA_PRESETS[mood]
            prompt = supervisor.generate_system_prompt(persona)

            assert isinstance(prompt, str)
            assert len(prompt) > 100  # Should be substantial


class TestResponseTemplates:
    """Test suite for response template retrieval."""

    def test_rating_response_happy(self):
        """Test rating response for HAPPY persona."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("😊")

        response = supervisor.get_response_for_question("rating", persona)

        assert response.isdigit()
        assert int(response) >= 9

    def test_rating_response_angry(self):
        """Test rating response for ANGRY persona."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("😡")

        response = supervisor.get_response_for_question("rating", persona)

        assert response.isdigit()
        assert int(response) <= 3

    def test_rating_response_neutral(self):
        """Test rating response for NEUTRAL persona."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("😐")

        response = supervisor.get_response_for_question("rating", persona)

        assert response.isdigit()
        assert 5 <= int(response) <= 7

    def test_text_response_happy(self):
        """Test text response for HAPPY persona is positive."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("😊")

        response = supervisor.get_response_for_question("text", persona)

        response_lower = response.lower()
        assert any(word in response_lower for word in [
            "excellent", "great", "helpful", "friendly", "definitely"
        ])

    def test_text_response_angry(self):
        """Test text response for ANGRY persona is negative."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("😡")

        response = supervisor.get_response_for_question("text", persona)

        response_lower = response.lower()
        assert any(word in response_lower for word in [
            "disappointing", "slow", "unhelpful", "poor", "frustrating", "wait"
        ])

    def test_yes_no_response_exists(self):
        """Test that yes/no responses are available."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("😊")

        response = supervisor.get_response_for_question("yes_no_positive", persona)

        assert response.lower() in ["yes", "no", "yes, definitely!", "probably"]

    def test_unknown_question_type_returns_generic(self):
        """Test that unknown question types return generic response."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("😊")

        response = supervisor.get_response_for_question("unknown_type_xyz", persona)

        assert response is not None
        assert len(response) > 0

    def test_response_randomization(self):
        """Test that randomized responses stay within bounds."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("😊")

        # Get multiple randomized ratings
        ratings = []
        for _ in range(10):
            response = supervisor.get_response_for_question(
                "rating", persona, randomize=True
            )
            ratings.append(int(response))

        # All should be within persona's range
        for rating in ratings:
            assert persona.rating_min <= rating <= persona.rating_max


class TestRatingScaleConversion:
    """Test suite for rating scale conversion."""

    def test_happy_persona_5_point_scale(self):
        """Test HAPPY persona on 1-5 scale returns 5."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("😊")

        rating = supervisor.get_rating_for_scale(persona, max_value=5)

        assert rating == 5

    def test_angry_persona_5_point_scale(self):
        """Test ANGRY persona on 1-5 scale returns 1."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("😡")

        rating = supervisor.get_rating_for_scale(persona, max_value=5)

        assert rating == 1

    def test_neutral_persona_5_point_scale(self):
        """Test NEUTRAL persona on 1-5 scale returns 3-4."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("😐")

        rating = supervisor.get_rating_for_scale(persona, max_value=5)

        assert 3 <= rating <= 4

    def test_happy_persona_7_point_scale(self):
        """Test HAPPY persona on 1-7 scale returns 7."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("😊")

        rating = supervisor.get_rating_for_scale(persona, max_value=7)

        assert rating == 7

    def test_happy_persona_10_point_scale(self):
        """Test HAPPY persona on 1-10 scale returns 10."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("😊")

        rating = supervisor.get_rating_for_scale(persona, max_value=10)

        assert rating == 10

    def test_rating_never_exceeds_max(self):
        """Test that rating never exceeds the scale maximum."""
        supervisor = SupervisorAgent()

        for mood in MoodType:
            persona = PERSONA_PRESETS[mood]
            for max_val in [3, 5, 7, 10]:
                rating = supervisor.get_rating_for_scale(persona, max_value=max_val)
                assert 1 <= rating <= max_val

    def test_rating_randomization_stays_in_bounds(self):
        """Test that randomized ratings stay within bounds."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("😊")

        for _ in range(20):
            rating = supervisor.get_rating_for_scale(
                persona, max_value=5, randomize=True
            )
            assert 1 <= rating <= 5


class TestPersonaSummary:
    """Test suite for persona summary functionality."""

    def test_summary_contains_required_fields(self):
        """Test that summary contains all required fields."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("😊")

        summary = supervisor.summarize_persona(persona)

        assert "mood" in summary
        assert "rating_preference" in summary
        assert "rating_range" in summary
        assert "text_tone" in summary
        assert "response_types_available" in summary

    def test_summary_values_are_correct(self):
        """Test that summary values match persona config."""
        supervisor = SupervisorAgent()
        persona = supervisor.get_persona("😊")

        summary = supervisor.summarize_persona(persona)

        assert summary["mood"] == "happy"
        assert summary["rating_preference"] == 10
        assert summary["text_tone"] == "enthusiastic"


class TestMoodTypeEnum:
    """Test suite for MoodType enum directly."""

    def test_from_input_all_happy_aliases(self):
        """Test all HAPPY aliases are recognized."""
        happy_inputs = [
            "😊", "😃", "😄", "happy", "positive", "satisfied",
            "excellent", "great", "⭐⭐⭐⭐⭐", "5", "10",
        ]

        for inp in happy_inputs:
            mood = MoodType.from_input(inp)
            assert mood == MoodType.HAPPY, f"'{inp}' should map to HAPPY"

    def test_from_input_all_angry_aliases(self):
        """Test all ANGRY aliases are recognized."""
        angry_inputs = [
            "😡", "😠", "angry", "negative", "dissatisfied",
            "terrible", "awful", "⭐", "1",
        ]

        for inp in angry_inputs:
            mood = MoodType.from_input(inp)
            assert mood == MoodType.ANGRY, f"'{inp}' should map to ANGRY"

    def test_from_input_all_neutral_aliases(self):
        """Test all NEUTRAL aliases are recognized."""
        neutral_inputs = [
            "😐", "neutral", "okay", "ok", "average",
            "fine", "⭐⭐⭐", "3",
        ]

        for inp in neutral_inputs:
            mood = MoodType.from_input(inp)
            assert mood == MoodType.NEUTRAL, f"'{inp}' should map to NEUTRAL"


class TestGetPersonaFromInput:
    """Test suite for the convenience function."""

    def test_convenience_function_works(self):
        """Test get_persona_from_input returns correct persona."""
        persona = get_persona_from_input("😊")

        assert persona.mood == MoodType.HAPPY
        assert persona.rating_preference == 10

    def test_convenience_function_matches_supervisor(self):
        """Test convenience function matches SupervisorAgent output."""
        supervisor = SupervisorAgent()

        for inp in ["😊", "😡", "😐", "happy", "angry"]:
            func_persona = get_persona_from_input(inp)
            agent_persona = supervisor.get_persona(inp)

            assert func_persona.mood == agent_persona.mood
            assert func_persona.rating_preference == agent_persona.rating_preference