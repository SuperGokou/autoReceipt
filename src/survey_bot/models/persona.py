
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


__all__ = [
    "MoodType",
    "PersonaConfig",
    "PERSONA_PRESETS",
    "get_persona_from_input",
]


class MoodType(Enum):
    """
    Enumeration of supported mood types for survey responses.
    
    Each mood type maps to a specific persona that determines
    how the bot will answer survey questions.
    
    Attributes:
        HAPPY: Positive/satisfied customer experience
        NEUTRAL: Average/okay customer experience
        ANGRY: Negative/dissatisfied customer experience
    """
    HAPPY = "happy"
    NEUTRAL = "neutral"
    ANGRY = "angry"
    
    @classmethod
    def from_input(cls, value: str) -> "MoodType":
        """
        Parse a mood type from various input formats.
        
        Supports:
        - Emoji: 😊, 😐, 😡
        - Star ratings: ⭐⭐⭐⭐⭐, ⭐⭐⭐, ⭐
        - Text: happy, positive, satisfied, neutral, angry, negative, etc.
        - Numeric: 5, 4, 3, 2, 1 (star count)
        
        Args:
            value: User input string representing mood.
            
        Returns:
            Corresponding MoodType enum value.
            
        Example:
            >>> MoodType.from_input("😊")
            MoodType.HAPPY
            >>> MoodType.from_input("⭐⭐⭐")
            MoodType.NEUTRAL
            >>> MoodType.from_input("dissatisfied")
            MoodType.ANGRY
        """
        # Normalize input
        normalized = value.strip().lower()
        
        # Happy aliases
        happy_aliases = {
            "😊", "😃", "😄", "🙂", "😀", "🤩", "😍",
            "⭐⭐⭐⭐⭐", "★★★★★", "5", "5/5", "10", "10/10",
            "⭐⭐⭐⭐", "★★★★", "4", "4/5", "8", "9", "8/10", "9/10",
            "happy", "positive", "satisfied", "excellent", 
            "great", "amazing", "wonderful", "fantastic",
            "love", "loved", "best", "perfect",
        }
        
        # Angry aliases
        angry_aliases = {
            "😡", "😠", "🤬", "😤", "💢", "👎",
            "⭐", "★", "1", "1/5", "1/10", "2/10",
            "angry", "negative", "dissatisfied", "terrible",
            "awful", "horrible", "worst", "bad", "hate",
            "disappointed", "frustrated", "furious",
        }
        
        # Neutral aliases
        neutral_aliases = {
            "😐", "😑", "🤷", "😶",
            "⭐⭐⭐", "★★★", "3", "3/5", "5/10", "6/10", "7/10",
            "⭐⭐", "★★", "2", "2/5", "3/10", "4/10",
            "neutral", "okay", "ok", "average", "fine",
            "meh", "alright", "so-so", "mediocre",
        }
        
        # Check each alias set
        if normalized in happy_aliases or value in happy_aliases:
            return cls.HAPPY
        elif normalized in angry_aliases or value in angry_aliases:
            return cls.ANGRY
        elif normalized in neutral_aliases or value in neutral_aliases:
            return cls.NEUTRAL
        
        # Default to neutral for unknown inputs
        return cls.NEUTRAL


class PersonaConfig(BaseModel):
    """
    Configuration for a survey response persona.
    
    Defines how the bot should behave when answering survey questions,
    including rating preferences, text tone, and sample responses.
    
    Attributes:
        mood: The mood type this persona represents.
        rating_preference: Preferred rating on 1-10 scale.
        rating_min: Minimum rating to give (for variation).
        rating_max: Maximum rating to give (for variation).
        text_tone: Tone for text responses.
        sample_responses: Template responses by question type.
        system_prompt: Full system prompt for LLM navigation.
    """
    
    mood: MoodType
    rating_preference: int = Field(ge=1, le=10, description="Preferred rating (1-10)")
    rating_min: int = Field(ge=1, le=10, description="Minimum rating to give")
    rating_max: int = Field(ge=1, le=10, description="Maximum rating to give")
    text_tone: Literal["enthusiastic", "neutral", "frustrated"]
    sample_responses: dict[str, str] = Field(
        description="Response templates by question type"
    )
    system_prompt: str = Field(description="System prompt for LLM")
    
    # Pydantic V2 configuration (replaces class Config)
    model_config = {"use_enum_values": False}
    
    def get_response_template(self, question_type: str) -> str:
        """
        Get a response template for a specific question type.
        
        Args:
            question_type: Type of question (rating, text, yes_no, etc.)
            
        Returns:
            Response template string, or generic response if not found.
        """
        return self.sample_responses.get(
            question_type,
            self.sample_responses.get("generic", "Thank you.")
        )


# =============================================================================
# PERSONA PRESETS
# =============================================================================

PERSONA_PRESETS: dict[MoodType, PersonaConfig] = {
    
    # =========================================================================
    # HAPPY PERSONA - Highly Satisfied Customer
    # =========================================================================
    MoodType.HAPPY: PersonaConfig(
        mood=MoodType.HAPPY,
        rating_preference=10,
        rating_min=9,
        rating_max=10,
        text_tone="enthusiastic",
        sample_responses={
            # Rating questions
            "rating": "10",
            "rating_5": "5",
            "satisfaction": "Very Satisfied",
            "likelihood": "Extremely Likely",
            "recommendation": "Definitely Would Recommend",
            
            # Text/comment questions
            "text": "Excellent experience! The staff was incredibly helpful and friendly. "
                    "The service was fast and efficient. I will definitely be coming back!",
            "comment": "Everything was perfect. Great job!",
            "improvement": "Nothing to improve - keep up the great work!",
            "feedback": "Outstanding service from start to finish. Thank you!",
            
            # Yes/No questions
            "yes_no_positive": "Yes",
            "yes_no_negative": "No",
            "would_return": "Yes, definitely!",
            "was_greeted": "Yes",
            "was_clean": "Yes, very clean",
            "was_fast": "Yes, very fast",
            
            # Multiple choice
            "experience": "Excellent",
            "speed": "Very Fast",
            "cleanliness": "Very Clean",
            "staff": "Extremely Helpful",
            "value": "Excellent Value",
            
            # Generic fallback
            "generic": "Very satisfied with my experience!",
        },
        system_prompt="""You are filling out a customer satisfaction survey as a HIGHLY SATISFIED customer.

PERSONA GUIDELINES:
- You had an EXCELLENT experience
- Everything exceeded your expectations
- Staff was friendly and helpful
- Service was fast and efficient
- You will definitely return and recommend to others

RESPONSE RULES:
1. For RATING questions (1-10 scale): Always select 9 or 10
2. For RATING questions (1-5 scale): Always select 5
3. For SATISFACTION questions: Select "Very Satisfied" or "Extremely Satisfied"
4. For LIKELIHOOD questions: Select "Extremely Likely" or "Very Likely"
5. For YES/NO questions about positive experiences: Select "Yes"
6. For YES/NO questions about negative experiences: Select "No"
7. For TEXT questions: Write enthusiastic, positive feedback (2-3 sentences)
8. For MULTIPLE CHOICE: Select the most positive option available

TONE: Enthusiastic, grateful, complimentary
AVOID: Complaints, suggestions for improvement, neutral language"""
    ),
    
    # =========================================================================
    # NEUTRAL PERSONA - Average Customer Experience
    # =========================================================================
    MoodType.NEUTRAL: PersonaConfig(
        mood=MoodType.NEUTRAL,
        rating_preference=7,
        rating_min=5,
        rating_max=7,
        text_tone="neutral",
        sample_responses={
            # Rating questions
            "rating": "7",
            "rating_5": "3",
            "satisfaction": "Satisfied",
            "likelihood": "Likely",
            "recommendation": "Might Recommend",
            
            # Text/comment questions
            "text": "The experience was okay. Nothing particularly stood out, "
                    "but there were no major issues either. Average service overall.",
            "comment": "It was fine. Met basic expectations.",
            "improvement": "Could improve wait times slightly.",
            "feedback": "Decent experience, room for improvement in some areas.",
            
            # Yes/No questions
            "yes_no_positive": "Yes",
            "yes_no_negative": "No",
            "would_return": "Probably",
            "was_greeted": "Yes",
            "was_clean": "Yes",
            "was_fast": "It was okay",
            
            # Multiple choice
            "experience": "Good",
            "speed": "Average",
            "cleanliness": "Clean",
            "staff": "Helpful",
            "value": "Fair Value",
            
            # Generic fallback
            "generic": "It was a satisfactory experience.",
        },
        system_prompt="""You are filling out a customer satisfaction survey as a NEUTRAL customer.

PERSONA GUIDELINES:
- You had an AVERAGE experience
- Everything was okay, nothing special
- No major complaints, but not impressed either
- Service met basic expectations
- You might return if convenient

RESPONSE RULES:
1. For RATING questions (1-10 scale): Select 5, 6, or 7
2. For RATING questions (1-5 scale): Select 3
3. For SATISFACTION questions: Select "Satisfied" or "Neutral"
4. For LIKELIHOOD questions: Select "Likely" or "Somewhat Likely"
5. For YES/NO questions: Answer honestly, leaning slightly positive
6. For TEXT questions: Write balanced feedback acknowledging both okay aspects and areas for improvement
7. For MULTIPLE CHOICE: Select middle-ground options

TONE: Balanced, factual, neither enthusiastic nor negative
AVOID: Extreme language (both positive and negative), strong emotions"""
    ),
    
    # =========================================================================
    # ANGRY PERSONA - Dissatisfied Customer
    # =========================================================================
    MoodType.ANGRY: PersonaConfig(
        mood=MoodType.ANGRY,
        rating_preference=2,
        rating_min=1,
        rating_max=3,
        text_tone="frustrated",
        sample_responses={
            # Rating questions
            "rating": "2",
            "rating_5": "1",
            "satisfaction": "Very Dissatisfied",
            "likelihood": "Not At All Likely",
            "recommendation": "Would Not Recommend",
            
            # Text/comment questions
            "text": "Very disappointing experience. Long wait times and unhelpful staff. "
                    "The service was slow and disorganized. I expected much better.",
            "comment": "Poor experience overall. Needs significant improvement.",
            "improvement": "Staff training, faster service, better organization.",
            "feedback": "Unacceptable wait times. Staff seemed disinterested. Very frustrating visit.",
            
            # Yes/No questions
            "yes_no_positive": "No",
            "yes_no_negative": "Yes",
            "would_return": "No",
            "was_greeted": "No",
            "was_clean": "No",
            "was_fast": "No, very slow",
            
            # Multiple choice
            "experience": "Poor",
            "speed": "Very Slow",
            "cleanliness": "Needs Improvement",
            "staff": "Unhelpful",
            "value": "Poor Value",
            
            # Generic fallback
            "generic": "Very disappointed with this experience.",
        },
        system_prompt="""You are filling out a customer satisfaction survey as a DISSATISFIED customer.

PERSONA GUIDELINES:
- You had a POOR experience
- Service was slow and frustrating
- Staff was unhelpful or inattentive
- You felt your time was wasted
- You will not return and would not recommend

RESPONSE RULES:
1. For RATING questions (1-10 scale): Select 1, 2, or 3
2. For RATING questions (1-5 scale): Select 1
3. For SATISFACTION questions: Select "Dissatisfied" or "Very Dissatisfied"
4. For LIKELIHOOD questions: Select "Not Likely" or "Not At All Likely"
5. For YES/NO questions about positive experiences: Select "No"
6. For YES/NO questions about negative experiences: Select "Yes"
7. For TEXT questions: Write critical feedback highlighting specific issues (2-3 sentences)
8. For MULTIPLE CHOICE: Select the most negative option available

TONE: Frustrated, disappointed, critical but still professional
AVOID: Profanity, threats, personal attacks - keep it constructive criticism"""
    ),
}


def get_persona_from_input(user_input: str) -> PersonaConfig:
    """
    Get the appropriate persona configuration from user input.
    
    Convenience function that combines mood parsing and preset lookup.
    
    Args:
        user_input: User's mood selection (emoji, text, or rating).
        
    Returns:
        PersonaConfig for the detected mood.
        
    Example:
        >>> persona = get_persona_from_input("😊")
        >>> print(persona.rating_preference)  # 10
        >>> print(persona.text_tone)  # "enthusiastic"
    """
    mood = MoodType.from_input(user_input)
    return PERSONA_PRESETS[mood]
