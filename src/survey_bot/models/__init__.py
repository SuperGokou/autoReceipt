

# =============================================================================
# Receipt Models (Module 1 - Ingestion Agent)
# Used for: Extracting survey URLs from receipt images
# =============================================================================
from .receipt import (
    ExtractionResult,  # Wrapper for extraction success/failure
    ReceiptData,  # Extracted receipt information
)

# =============================================================================
# Persona Models (Module 2 - Supervisor)
# Used for: Mapping user mood to survey response behavior
# =============================================================================
from .persona import (
    MoodType,  # Enum: HAPPY, NEUTRAL, ANGRY
    PersonaConfig,  # Full persona configuration
    PERSONA_PRESETS,  # Pre-built persona configs
    get_persona_from_input,  # Convenience function
)

# =============================================================================
# Page State Models (Module 3 - Navigator)
# Used for: Observing and interacting with survey pages
# =============================================================================
from .page_state import (
    BoundingBox,  # Element position (x, y, width, height)
    ElementType,  # Enum: BUTTON, RADIO, CHECKBOX, etc.
    PageState,  # Complete page snapshot
    QuestionType,  # Enum: RATING_SCALE, TEXT_INPUT, etc.
    WebElement,  # Single interactive element
)

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Module 1: Receipt/Ingestion
    "ExtractionResult",
    "ReceiptData",

    # Module 2: Persona/Supervisor
    "MoodType",
    "PersonaConfig",
    "PERSONA_PRESETS",
    "get_persona_from_input",

    # Module 3: Page State/Navigator
    "BoundingBox",
    "ElementType",
    "PageState",
    "QuestionType",
    "WebElement",
]