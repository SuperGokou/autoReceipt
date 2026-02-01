from __future__ import annotations

from enum import Enum
from typing import Any, Optional, List

from pydantic import BaseModel, Field


__all__ = [
    "QuestionType",
    "ElementType",
    "WebElement",
    "BoundingBox",
    "PageState",
]


class QuestionType(Enum):
    """
    Types of survey questions that can be detected on a page.
    
    Used by the PageObserver to classify what kind of response
    is expected from the current survey page.
    """
    RATING_SCALE = "rating_scale"       # 1-10, 1-5, stars
    TEXT_INPUT = "text_input"           # Comment box, feedback
    MULTIPLE_CHOICE = "multiple_choice" # Radio buttons, single select
    MULTI_SELECT = "multi_select"       # Checkboxes, multiple answers
    YES_NO = "yes_no"                   # Binary choice
    DROPDOWN = "dropdown"               # Select dropdown
    NAVIGATION = "navigation"           # Just a next/continue button
    COMPLETION = "completion"           # Final page with coupon/code
    UNKNOWN = "unknown"                 # Cannot determine type


class ElementType(Enum):
    """
    Types of interactive HTML elements.
    """
    BUTTON = "button"
    LINK = "link"
    RADIO = "radio"
    CHECKBOX = "checkbox"
    TEXT_INPUT = "text_input"
    TEXTAREA = "textarea"
    SELECT = "select"
    OPTION = "option"
    STAR_RATING = "star_rating"
    SLIDER = "slider"
    IMAGE = "image"
    OTHER = "other"


class BoundingBox(BaseModel):
    """
    Bounding box coordinates for an element.
    
    Used for visual positioning and click targeting.
    """
    x: float = Field(description="X coordinate (left)")
    y: float = Field(description="Y coordinate (top)")
    width: float = Field(description="Element width")
    height: float = Field(description="Element height")
    
    @property
    def center_x(self) -> float:
        """Get center X coordinate."""
        return self.x + self.width / 2
    
    @property
    def center_y(self) -> float:
        """Get center Y coordinate."""
        return self.y + self.height / 2
    
    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


class WebElement(BaseModel):
    """
    Represents an interactive element on a web page.
    
    Contains all information needed for the Navigator to
    identify and interact with the element.
    
    Attributes:
        element_id: Unique identifier for this element.
        element_type: Type of element (button, input, etc.).
        text_content: Visible text content.
        aria_label: ARIA label for accessibility.
        is_visible: Whether element is currently visible.
        is_enabled: Whether element is enabled for interaction.
        bounding_box: Element position and size.
        attributes: Additional HTML attributes.
        selector: CSS selector to locate this element.
        value: Current value (for inputs).
    """
    element_id: str = Field(description="Unique element identifier")
    element_type: ElementType = Field(description="Type of element")
    text_content: str = Field(default="", description="Visible text")
    aria_label: Optional[str] = Field(default=None, description="ARIA label")
    placeholder: Optional[str] = Field(default=None, description="Input placeholder")
    is_visible: bool = Field(default=True, description="Is element visible")
    is_enabled: bool = Field(default=True, description="Is element enabled")
    is_checked: Optional[bool] = Field(default=None, description="Checked state for checkboxes/radios")
    bounding_box: Optional[BoundingBox] = Field(default=None, description="Position and size")
    attributes: dict[str, str] = Field(default_factory=dict, description="HTML attributes")
    selector: str = Field(default="", description="CSS selector")
    value: Optional[str] = Field(default=None, description="Input value")
    
    model_config = {"use_enum_values": False}
    
    def matches_text(self, *keywords: str, case_sensitive: bool = False) -> bool:
        """
        Check if element text matches any of the keywords.
        
        Args:
            keywords: Words to search for.
            case_sensitive: Whether to match case.
            
        Returns:
            True if any keyword found in text or label.
        """
        searchable = f"{self.text_content} {self.aria_label or ''}"
        if not case_sensitive:
            searchable = searchable.lower()
            keywords = tuple(k.lower() for k in keywords)
        
        return any(kw in searchable for kw in keywords)
    
    def get_display_text(self) -> str:
        """Get the best available display text."""
        return self.text_content or self.aria_label or self.placeholder or self.element_id


class PageState(BaseModel):
    """
    Complete snapshot of a survey page's state.
    
    Contains all information the Navigator needs to decide
    what action to take on the current page.
    
    Attributes:
        url: Current page URL.
        title: Page title.
        elements: List of interactive elements.
        question_type: Detected question type.
        question_text: The main question being asked.
        has_coupon: Whether a coupon/code is visible.
        coupon_code: Extracted coupon code if found.
        page_text: Simplified text content of page.
        error_message: Any error message displayed.
    """
    url: str = Field(description="Current page URL")
    title: str = Field(default="", description="Page title")
    elements: list[WebElement] = Field(default_factory=list, description="Interactive elements")
    question_type: QuestionType = Field(default=QuestionType.UNKNOWN, description="Type of question")
    question_text: str = Field(default="", description="Main question text")
    has_coupon: bool = Field(default=False, description="Coupon/code detected")
    coupon_code: Optional[str] = Field(default=None, description="Extracted coupon code")
    page_text: str = Field(default="", description="Page text content (truncated)")
    error_message: Optional[str] = Field(default=None, description="Error message if any")
    
    model_config = {"use_enum_values": False}
    
    def get_elements_by_type(self, element_type: ElementType) -> list[WebElement]:
        """Get all elements of a specific type."""
        return [e for e in self.elements if e.element_type == element_type]
    
    def get_visible_elements(self) -> list[WebElement]:
        """Get only visible elements."""
        return [e for e in self.elements if e.is_visible]
    
    def get_buttons(self) -> list[WebElement]:
        """Get all button elements."""
        return self.get_elements_by_type(ElementType.BUTTON)
    
    def get_inputs(self) -> list[WebElement]:
        """Get all input elements (text, textarea)."""
        return [
            e for e in self.elements 
            if e.element_type in (ElementType.TEXT_INPUT, ElementType.TEXTAREA)
        ]
    
    def get_radio_buttons(self) -> list[WebElement]:
        """Get all radio button elements."""
        return self.get_elements_by_type(ElementType.RADIO)
    
    def get_checkboxes(self) -> list[WebElement]:
        """Get all checkbox elements."""
        return self.get_elements_by_type(ElementType.CHECKBOX)
    
    def find_element_by_text(self, *keywords: str) -> Optional[WebElement]:
        """Find first element matching any keyword."""
        for element in self.elements:
            if element.matches_text(*keywords):
                return element
        return None
    
    def find_submit_button(self) -> Optional[WebElement]:
        """Find the submit/next/continue button."""
        submit_keywords = [
            "next", "continue", "submit", "proceed", "finish",
            "done", "complete", "send", "enviar", "siguiente"
        ]
        
        for element in self.get_buttons():
            if element.matches_text(*submit_keywords):
                return element
        
        # Also check links that might act as buttons
        for element in self.get_elements_by_type(ElementType.LINK):
            if element.matches_text(*submit_keywords):
                return element
        
        return None
    
    def to_summary(self) -> dict[str, Any]:
        """Get a summarized version for logging."""
        return {
            "url": self.url,
            "title": self.title,
            "question_type": self.question_type.value,
            "element_count": len(self.elements),
            "visible_elements": len(self.get_visible_elements()),
            "has_coupon": self.has_coupon,
            "coupon_code": self.coupon_code,
        }
