"""
Optimized Vision-based Survey Navigator using Qwen3-VL.

OPTIMIZATIONS:
1. Reduce vision model calls - use heuristics for obvious actions
2. Batch multiple questions per page in single model call
3. Compress screenshots before sending
4. Auto-click Next without model when appropriate
5. Cache page analysis patterns
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal, List, Any, Dict

import httpx

logger = logging.getLogger(__name__)


@dataclass
class VisionAction:
    """Action decided by the vision model."""
    action_type: Literal["click", "fill", "select", "done", "wait", "error"]
    target: Optional[str] = None
    value: Optional[str] = None
    reasoning: str = ""


@dataclass 
class PageAnalysis:
    """Complete page analysis with multiple actions."""
    page_type: str  # "question", "rating", "text", "email", "complete", "error"
    actions: List[VisionAction] = field(default_factory=list)
    has_next_button: bool = False
    is_complete: bool = False
    error_message: Optional[str] = None


class OptimizedVisionNavigator:
    """
    Optimized navigator that minimizes vision model calls.
    
    Key optimizations:
    - Detects page type with DOM first, only uses vision when needed
    - Batches all actions for a page in single model call
    - Auto-clicks Next without model
    - Compresses screenshots for faster transfer
    """
    
    def __init__(
        self,
        ollama_model: Optional[str] = None,
        ollama_host: Optional[str] = None,
        mood: str = "happy",
        email: Optional[str] = None,
    ):
        self.model = ollama_model or os.environ.get("OLLAMA_VISION_MODEL", "qwen3-vl")
        from .ollama_host import detect_ollama_host
        self.host = ollama_host or detect_ollama_host()
        self.mood = mood
        self.email = email
        
        self.rating_map = {
            "happy": {"scale_10": "10", "scale_5": "5", "satisfaction": "Highly Satisfied", "likelihood": "Definitely"},
            "neutral": {"scale_10": "5", "scale_5": "3", "satisfaction": "Neither", "likelihood": "Maybe"},
            "angry": {"scale_10": "1", "scale_5": "1", "satisfaction": "Highly Dissatisfied", "likelihood": "Never"},
        }.get(mood, {})
        
        logger.info(f"OptimizedVisionNavigator: model={self.model}, mood={mood}")
    
    async def analyze_and_act(self, page, step: int = 0, retry_count: int = 0) -> PageAnalysis:
        """
        Analyze page and determine ALL actions needed before clicking Next.

        This is the main optimization - instead of one action per model call,
        we get all actions for the page at once.
        """
        # OPTIMIZATION 1: Try DOM-based detection first (no model call needed)
        dom_analysis = await self._quick_dom_analysis(page)

        if dom_analysis:
            logger.info(f"Step {step}: Quick DOM analysis - {dom_analysis.page_type}")
            return dom_analysis

        # OPTIMIZATION 2: Use vision model only when DOM analysis isn't enough
        try:
            # Take compressed screenshot
            screenshot_b64 = await self._take_compressed_screenshot(page)

            # Get batch analysis from model
            url = page.url
            title = await page.title()
            prompt = self._build_batch_prompt(url, title, step)

            response = await self._call_vision_llm(prompt, screenshot_b64)

            # Log the raw vision model response for debugging
            logger.debug(f"[VISION MODEL] Raw response:\n{response[:500]}...")

            analysis = self._parse_batch_response(response)

            logger.info(f"Step {step}: Vision analysis - {analysis.page_type}, {len(analysis.actions)} actions")

            # CRITICAL FIX: If vision model returns 0 actions, try fallback heuristic
            if len(analysis.actions) == 0:
                logger.warning(f"[VISION MODEL] Returned 0 actions. Response preview: {response[:200]}")

                # Try fallback: click any visible option buttons on the page
                fallback = await self._fallback_statement_click(page)
                if fallback:
                    logger.info("[FALLBACK] Using heuristic click for statement comparison")
                    return fallback

                # If still no actions, just try to click Next to move forward
                logger.warning("[FALLBACK] No actions found, will try clicking Next to proceed")
                return PageAnalysis(
                    page_type="fallback",
                    actions=[],
                    has_next_button=True,
                )

            return analysis

        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return PageAnalysis(page_type="error", error_message=str(e))
    
    async def _fallback_statement_click(self, page) -> Optional[PageAnalysis]:
        """
        Fallback heuristic for statement comparison questions when vision model fails.
        Tries to click ALL unanswered options based on mood without reading the statements.
        """
        try:
            body_text = await page.inner_text("body")
            body_lower = body_text.lower()

            # Check if this is a statement comparison page
            if "statement a" not in body_lower or "statement b" not in body_lower:
                return None

            logger.info("[FALLBACK] Detected statement comparison, using mood-based heuristic")

            # Determine preferred option text based on mood
            if self.mood == "happy":
                # For happy mood, prefer "much more" (strong preference)
                # We'll try to click multiple instances
                primary_option = "much more"
            elif self.mood == "angry":
                primary_option = "much more"
            else:  # neutral
                primary_option = "somewhat more"

            # Find ALL clickable options that match our preference
            # This handles multiple statement pairs on the same page
            actions = []

            # Try to find all divs/buttons with the target text
            selectors_to_try = [
                f"div:has-text('{primary_option}')",
                f"button:has-text('{primary_option}')",
                f"text=/{primary_option}/i",
            ]

            for selector in selectors_to_try:
                try:
                    elements = page.locator(selector)
                    count = await elements.count()
                    logger.info(f"[FALLBACK] Found {count} elements matching '{selector}'")

                    # Click each visible element
                    for i in range(count):
                        try:
                            element = elements.nth(i)
                            is_visible = await element.is_visible(timeout=500)
                            if is_visible:
                                # Get the full text to use as target
                                text = await element.inner_text()
                                text = text.strip()

                                # Only add if it's one of the standard statement comparison options
                                if "statement" in text.lower() and ("much more" in text.lower() or "somewhat more" in text.lower()):
                                    # Check if we already have this action
                                    if not any(action.target == text for action in actions):
                                        actions.append(VisionAction(action_type="click", target=text))
                                        logger.info(f"[FALLBACK] Will click option {i+1}: {text[:60]}...")
                        except Exception as e:
                            logger.debug(f"[FALLBACK] Element {i} not clickable: {e}")
                            continue
                except Exception as e:
                    logger.debug(f"[FALLBACK] Selector '{selector}' failed: {e}")
                    continue

            if actions:
                logger.info(f"[FALLBACK] Found {len(actions)} statement comparison options to click")
                return PageAnalysis(
                    page_type="question",
                    actions=actions,
                    has_next_button=True,
                )

            # If we couldn't find by primary option, try the full text patterns
            full_text_options = []
            if self.mood == "happy":
                full_text_options = [
                    "Statement B describes me much more than Statement A",
                    "Statement A describes me much more than Statement B",
                ]
            elif self.mood == "angry":
                full_text_options = [
                    "Statement A describes me much more than Statement B",
                    "Statement B describes me much more than Statement A",
                ]
            else:  # neutral
                full_text_options = [
                    "Statement A describes me somewhat more than Statement B",
                    "Statement B describes me somewhat more than Statement A",
                ]

            # Try to find at least one option to click
            for option_text in full_text_options:
                try:
                    element = page.locator(f"text=/{option_text}/i").first
                    if await element.is_visible(timeout=1000):
                        logger.info(f"[FALLBACK] Found single clickable option: {option_text}")
                        return PageAnalysis(
                            page_type="question",
                            actions=[VisionAction(action_type="click", target=option_text)],
                            has_next_button=True,
                        )
                except Exception:
                    continue

            return None

        except Exception as e:
            logger.debug(f"[FALLBACK] Failed: {e}")
            return None

    async def _quick_dom_analysis(self, page) -> Optional[PageAnalysis]:
        """
        Fast DOM-based page analysis without calling vision model.
        
        Handles common patterns:
        - Completion pages (thank you, validation code)
        - Error pages (already used, invalid)
        - Simple Next/Submit pages
        - Email entry pages
        """
        try:
            body_text = await page.inner_text("body")
            body_lower = body_text.lower()

            # Check progress indicator
            import re
            progress_match = re.search(r'(\d+)\s*%', body_text)
            progress_value = int(progress_match.group(1)) if progress_match else 0

            # If progress is >= 95% and page has "optional" text, skip by clicking Next
            # BUT only if there are no required inputs (radio buttons, dropdowns, etc.)
            if progress_value >= 95:
                if "optional" in body_lower or "demographic" in body_lower:
                    # Check if there are any input fields that might be required
                    radio_count = await page.locator("input[type='radio']").count()
                    select_count = await page.locator("select").count()

                    # If there are select dropdowns or radio buttons, don't auto-skip
                    # The survey might still require an answer even though it says "optional"
                    if radio_count == 0 and select_count == 0:
                        logger.info(f"[OPTIONAL SKIP] Detected optional page at {progress_value}% with no inputs, skipping")
                        return PageAnalysis(
                            page_type="optional_skip",
                            actions=[VisionAction(action_type="click", target="Next")],
                            has_next_button=True,
                        )
                    else:
                        logger.info(f"[OPTIONAL DETECTED] Found optional page at {progress_value}% but has {radio_count} radios and {select_count} dropdowns - will process normally")

            # FIRST: Check for Start button - this is a WELCOME page, not completion!
            start_btn = page.locator("button:has-text('Start'), input[value='Start' i], a:has-text('Start')")
            has_start_button = await start_btn.count() > 0
            
            if has_start_button:
                # This is the welcome/start page - click Start!
                logger.info("Detected welcome page with Start button")
                return PageAnalysis(
                    page_type="welcome",
                    actions=[VisionAction(action_type="click", target="Start")],
                    has_next_button=True,
                )
            
            # SECOND: Check for email fields BEFORE completion check!
            # This prevents "receive your coupon code" from triggering false completion
            # Look for email inputs by type, name, or id attributes
            email_inputs = await page.locator("input[type='email'], input[name*='email' i], input[id*='email' i]").count()

            # Also check if page text mentions email and has text inputs (for sites that use type='text' for email)
            if email_inputs == 0 and ("email address" in body_lower or "enter your email" in body_lower or "provide your email" in body_lower):
                # Count text inputs that might be email fields
                text_inputs = await page.locator("input[type='text']").count()
                if text_inputs > 0:
                    logger.info(f"[EMAIL DETECTION] Page mentions email with {text_inputs} text inputs - treating as email page")
                    email_inputs = text_inputs

            if email_inputs > 0:
                if self.email:
                    logger.info(f"[EMAIL DETECTED] Found email entry page with {email_inputs} field(s)")
                    actions = []
                    for i in range(email_inputs):
                        actions.append(VisionAction(
                            action_type="fill",
                            target="email",
                            value=self.email,
                        ))
                    return PageAnalysis(
                        page_type="email",
                        actions=actions,
                        has_next_button=True,
                    )
                else:
                    # No email provided - just click Next to skip
                    logger.info("[EMAIL SKIP] Email page detected but no email provided, skipping")
                    return PageAnalysis(
                        page_type="email_skip",
                        actions=[VisionAction(action_type="click", target="Next")],
                        has_next_button=True,
                    )
            
            # Check for completion - URL-based detection first
            url_lower = page.url.lower()
            if "/thanks" in url_lower or "/thankyou" in url_lower or "/complete" in url_lower:
                logger.info(f"Detected completion page from URL: {page.url}")
                return PageAnalysis(page_type="complete", is_complete=True)

            # Check for completion - must have specific phrases
            # Be careful: "receive your coupon code" is NOT completion
            # Only "your code is", "validation code: XXX" are completion
            completion_indicators = [
                "survey complete", "validation code:", "your code is",
                "your validation code", "your coupon code is",
                "thank you for completing", "thanks for completing",
                "thank you for taking the time to complete",
                "thank you for your feedback",  # Added for Zendesk surveys
                "you have completed", "has been completed",
                "your feedback has been submitted", "survey has been submitted",
            ]
            
            # Check if page has a visible code (6+ digits together)
            import re
            visible_code = re.search(r'\b(\d{6,12})\b', body_text)
            
            # Strong completion: specific phrase + visible code
            if visible_code and any(phrase in body_lower for phrase in completion_indicators):
                logger.info(f"Detected completion page (strong indicator + code: {visible_code.group(1)})")
                return PageAnalysis(page_type="complete", is_complete=True)
            
            # Very strong indicators that don't need a code
            definite_completion = [
                "your validation code", "validation code:", 
                "here is your code", "your code is",
            ]
            if any(phrase in body_lower for phrase in definite_completion):
                logger.info("Detected completion page (definite indicator)")
                return PageAnalysis(page_type="complete", is_complete=True)
            
            # Check for errors
            error_phrases = [
                "already been used", "already used", "invalid code", 
                "expired", "not valid", "code is not valid"
            ]
            for phrase in error_phrases:
                if phrase in body_lower:
                    return PageAnalysis(
                        page_type="error",
                        error_message=f"Survey error: {phrase}",
                    )
            
            # Check for statement comparison questions (Best Buy, tech surveys, etc.)
            # These use clickable divs/buttons instead of radio inputs
            # Let vision model read and decide based on mood
            if "statement a" in body_lower and "statement b" in body_lower:
                if "describes me" in body_lower or "prefer" in body_lower:
                    # Check if there are multiple pairs of statements on the same page
                    # Count how many "Statement A" mentions there are (indicates multiple questions)
                    import re
                    statement_a_count = len(re.findall(r'statement\s+a', body_lower))

                    # Store the expected count for later validation
                    if not hasattr(self, '_expected_statement_count'):
                        self._expected_statement_count = {}

                    # Use URL as key to track per-page
                    page_key = page.url.split('?')[0]  # Remove query params
                    self._expected_statement_count[page_key] = statement_a_count

                    # If multiple statement pairs, we need vision model to handle all of them
                    if statement_a_count > 1:
                        logger.info(f"[STATEMENT COMPARISON] Detected {statement_a_count} statement pairs - will keep answering until all are done")
                    else:
                        logger.info("[STATEMENT COMPARISON] Detected statement A/B comparison question - will use vision model to decide")

                    # Return None to trigger vision model analysis
                    # Vision model will read statements and choose based on mood
                    return None

            # Check for dropdowns (select elements) - especially for optional demographic questions
            select_elements = await page.locator("select").count()
            if select_elements > 0:
                logger.info(f"[DROPDOWN DETECTED] Found {select_elements} dropdown(s)")
                # Check if they're truly optional or have default "- Select One -" values
                # If at high progress and marked optional, we can try to just click Next
                if progress_value >= 95 and ("optional" in body_lower or "demographic" in body_lower):
                    logger.info(f"[DROPDOWN SKIP] Dropdowns are optional at {progress_value}%, attempting to skip")
                    return PageAnalysis(
                        page_type="optional_dropdown",
                        actions=[VisionAction(action_type="click", target="Next")],
                        has_next_button=True,
                    )

            # Check for simple rating questions - handle with heuristics
            radio_buttons = await page.locator("input[type='radio']").count()
            text_inputs = await page.locator("input[type='text']").count()
            textareas = await page.locator("textarea").count()

            if radio_buttons > 0:
                logger.info(f"[RADIO DETECTED] Found {radio_buttons} radio button(s)")
                # Get visible radio button options
                analysis = await self._analyze_radio_options(page)
                if analysis:
                    return analysis
                else:
                    logger.warning(f"[RADIO WARNING] Radio analysis returned None despite {radio_buttons} radios found")

            # Check for text input fields (brand awareness, etc.)
            if text_inputs > 0:
                logger.info(f"[TEXT INPUT DETECTED] Found {text_inputs} text input fields")
                # Check if this is asking for brand names or similar
                if any(word in body_lower for word in ["retailer", "brand", "store", "aware", "know", "comes to mind"]):
                    # Generate brand names for electronics retailers
                    brands = ["Best Buy", "Amazon", "Walmart", "Target", "Apple Store", "Microsoft Store", "Costco", "Sam's Club"]
                    actions = []
                    for i in range(min(text_inputs, len(brands))):
                        actions.append(VisionAction(
                            action_type="fill",
                            target="brand",
                            value=brands[i],
                        ))
                    logger.info(f"[BRAND FILL] Generated {len(actions)} brand filling actions")
                    return PageAnalysis(
                        page_type="text",
                        actions=actions,
                        has_next_button=True,
                    )

            # Check for textarea/comment fields
            if textareas > 0:
                logger.info(f"Detected {textareas} textarea fields")
                # Check if this is a comment/feedback page
                body_lower = (await page.inner_text("body")).lower()
                if any(word in body_lower for word in ["comment", "feedback", "tell us", "what you liked", "experience", "describe"]):
                    # Generate a mood-appropriate comment
                    comment = self._generate_comment()
                    actions = []
                    for i in range(textareas):
                        actions.append(VisionAction(
                            action_type="fill",
                            target="comment",
                            value=comment,
                        ))
                    logger.info(f"Generated comment filling action: '{comment[:50]}...'")
                    return PageAnalysis(
                        page_type="text",
                        actions=actions,
                        has_next_button=True,
                    )

            # Check for slider-based rating questions
            sliders = await page.locator("input[type='range']").count()
            if sliders > 0:
                logger.info(f"[SLIDER DETECTED] Found {sliders} slider element(s)")
                # Sliders are typically for rating retailers/brands on a scale
                # For happy mood, set sliders to high values (9-10 out of 10)
                # For sad mood, set to low values (1-3 out of 10)
                if self.user_mood == "happy":
                    slider_value = 9  # High rating for happy mood
                elif self.user_mood == "sad":
                    slider_value = 2  # Low rating for sad mood
                else:
                    slider_value = 7  # Neutral/default

                actions = []
                for i in range(sliders):
                    actions.append(VisionAction(
                        action_type="slider",
                        target=f"slider_{i}",
                        value=str(slider_value),
                    ))
                logger.info(f"[SLIDER ACTION] Generated {len(actions)} slider actions with value={slider_value}")
                return PageAnalysis(
                    page_type="slider",
                    actions=actions,
                    has_next_button=True,
                )

            # Check if just a Next button (no questions)
            next_btn = await self._find_next_button(page)
            if next_btn and radio_buttons == 0 and textareas == 0 and select_elements == 0 and text_inputs == 0 and sliders == 0:
                # Before declaring it a transition page, check if there might be statement comparisons we missed
                # (they use divs/buttons, not radio inputs)
                if "statement a" not in body_lower and "statement b" not in body_lower:
                    # Truly a transition page with just a Next button
                    logger.info("[TRANSITION] Detected transition page with just Next button")
                    return PageAnalysis(
                        page_type="transition",
                        actions=[VisionAction(action_type="click", target="Next")],
                        has_next_button=True,
                    )
                # Otherwise, let vision model check for hidden statement questions
                logger.info("[TRANSITION SKIP] Has Next button but also statement text - using vision model")

            return None  # Need vision model
            
        except Exception as e:
            logger.debug(f"Quick DOM analysis failed: {e}")
            return None
    
    async def _analyze_radio_options(self, page) -> Optional[PageAnalysis]:
        """Analyze radio button options and select based on mood."""
        try:
            # Find all radio groups (by name attribute)
            radios = page.locator("input[type='radio']")
            count = await radios.count()
            
            logger.debug(f"Found {count} radio buttons")
            
            if count == 0:
                return None
            
            # Group radios by name
            groups = {}
            for i in range(count):
                radio = radios.nth(i)
                name = await radio.get_attribute("name") or f"group_{i}"
                value = await radio.get_attribute("value") or ""
                
                # Get label text - try multiple methods
                label_text = ""
                
                # Method 1: Label with for attribute
                radio_id = await radio.get_attribute("id")
                if radio_id:
                    label = page.locator(f"label[for='{radio_id}']")
                    if await label.count() > 0:
                        label_text = await label.first.inner_text()
                
                # Method 2: Parent label element
                if not label_text:
                    try:
                        parent_label = radio.locator("xpath=ancestor::label")
                        if await parent_label.count() > 0:
                            label_text = await parent_label.first.inner_text()
                    except Exception:
                        pass
                
                # Method 3: Adjacent text (for matrix-style)
                if not label_text:
                    try:
                        # Get aria-label or title
                        label_text = await radio.get_attribute("aria-label") or ""
                        if not label_text:
                            label_text = await radio.get_attribute("title") or ""
                    except Exception:
                        pass
                
                if name not in groups:
                    groups[name] = []
                groups[name].append({
                    "value": value, 
                    "label": label_text.strip(), 
                    "index": i,
                })
            
            logger.debug(f"Found {len(groups)} radio groups")
            
            # Check if this looks like a satisfaction scale (matrix style)
            # Look for header text in the page
            page_text = await page.inner_text("body")
            page_lower = page_text.lower()
            
            is_satisfaction_scale = any(phrase in page_lower for phrase in [
                "highly satisfied", "very satisfied", "satisfied",
                "very likely", "likely", "unlikely",
                "strongly agree", "agree", "disagree",
            ])
            
            actions = []
            for group_name, options in groups.items():
                # Check if already selected
                try:
                    selected = await page.locator(f"input[name='{group_name}']:checked").count()
                    if selected > 0:
                        logger.debug(f"Group {group_name} already selected")
                        continue  # Already answered
                except Exception:
                    pass
                
                logger.debug(f"Processing group {group_name} with {len(options)} options")
                
                # Determine which option to select based on mood
                choice_index = None
                
                if is_satisfaction_scale:
                    # For satisfaction scales, position matters:
                    # Usually: Highly Satisfied (leftmost/first) ... Highly Dissatisfied (rightmost/last)
                    if self.mood == "happy":
                        choice_index = 0  # First = best rating
                    elif self.mood == "angry":
                        choice_index = len(options) - 1  # Last = worst rating
                    else:
                        choice_index = len(options) // 2  # Middle
                else:
                    # Try label-based selection
                    choice = self._choose_option(options)
                    if choice:
                        choice_index = options.index(choice)
                    else:
                        choice_index = 0  # Fallback to first
                
                if choice_index is not None and 0 <= choice_index < len(options):
                    option = options[choice_index]
                    logger.info(f"[RADIO SELECT] Will select option {choice_index}/{len(options)-1} in group '{group_name}': {option.get('label') or option.get('value')} (global index: {option['index']})")
                    actions.append(VisionAction(
                        action_type="select",
                        target=option["label"] or option["value"] or f"radio_{option['index']}",
                        value=str(option["index"]),  # Store the global index for direct clicking
                    ))
                else:
                    logger.warning(f"[RADIO ERROR] Invalid choice_index {choice_index} for group '{group_name}' with {len(options)} options")
            
            if actions:
                logger.info(f"Generated {len(actions)} select actions")
                return PageAnalysis(
                    page_type="question",
                    actions=actions,
                    has_next_button=True,
                )
            
            logger.debug("No radio actions needed")
            return None
            
        except Exception as e:
            logger.warning(f"Radio analysis failed: {e}")
            return None
    
    def _choose_option(self, options: list) -> Optional[dict]:
        """Choose the best option based on mood."""
        if not options:
            return None
        
        # Keywords for each mood
        happy_keywords = ["highly satisfied", "excellent", "definitely", "very likely", "10", "5", "yes", "always"]
        angry_keywords = ["highly dissatisfied", "poor", "never", "very unlikely", "1", "no", "rarely"]
        neutral_keywords = ["neither", "neutral", "maybe", "somewhat", "5", "3", "sometimes"]
        
        if self.mood == "happy":
            keywords = happy_keywords
            # For numeric scales, prefer highest
            fallback_index = -1  # Last option (usually highest)
        elif self.mood == "angry":
            keywords = angry_keywords
            fallback_index = 0  # First option (usually lowest)
        else:
            keywords = neutral_keywords
            fallback_index = len(options) // 2  # Middle option
        
        # Try to match keywords
        for option in options:
            label_lower = option["label"].lower()
            value_lower = option["value"].lower()
            for keyword in keywords:
                if keyword in label_lower or keyword == value_lower:
                    return option
        
        # Fallback to positional choice
        return options[fallback_index] if options else None

    def _generate_comment(self) -> str:
        """Generate a mood-appropriate comment for textarea fields."""
        import random

        if self.mood == "happy":
            comments = [
                "The food was excellent and the service was outstanding! I really enjoyed my visit.",
                "Great experience! Everything was fresh and the staff was very friendly.",
                "Wonderful service and delicious food. Will definitely come back again!",
                "Very satisfied with my visit. The restaurant was clean and the food was hot and fresh.",
                "Excellent customer service! The team was helpful and my order was perfect.",
            ]
        elif self.mood == "angry":
            comments = [
                "The service was slow and the food was not fresh.",
                "Very disappointed with my experience. The order was wrong.",
                "Poor quality food and inattentive staff.",
                "Not satisfied. The restaurant was not clean and service was poor.",
                "Unsatisfactory visit. Long wait time and cold food.",
            ]
        else:  # neutral
            comments = [
                "The experience was okay. Nothing special but nothing terrible either.",
                "Average visit. Service was acceptable and food was decent.",
                "Standard experience. Met basic expectations.",
                "The visit was fine. Food was okay and service was average.",
                "Reasonable experience overall.",
            ]

        return random.choice(comments)

    async def _find_next_button(self, page) -> Optional[bool]:
        """Check if there's a clickable Next button."""
        selectors = [
            "button:has-text('Next')",
            "input[value='Next' i]",
            "button:has-text('Continue')",
            "input[value='Continue' i]",
            "button:has-text('Submit')",
            "input[type='submit']",
            "button:has-text('>>')",
            "button:has-text('>')",
        ]
        
        for selector in selectors:
            try:
                btn = page.locator(selector).first
                if await btn.is_visible(timeout=500):
                    disabled = await btn.get_attribute("disabled")
                    if not disabled:
                        return True
            except Exception:
                continue
        return False
    
    async def _take_compressed_screenshot(self, page) -> str:
        """Take a compressed screenshot to reduce transfer time."""
        try:
            from PIL import Image
            
            # Take screenshot
            screenshot_bytes = await page.screenshot()
            
            # Compress with PIL
            img = Image.open(io.BytesIO(screenshot_bytes))
            
            # Resize if too large (max 1024px width)
            if img.width > 1024:
                ratio = 1024 / img.width
                new_size = (1024, int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            
            # Save as JPEG with compression
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=75)
            
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
            
        except ImportError:
            # PIL not available, use original
            screenshot_bytes = await page.screenshot()
            return base64.b64encode(screenshot_bytes).decode("utf-8")
    
    def _build_batch_prompt(self, url: str, title: str, step: int) -> str:
        """Build prompt that requests ALL actions for the page at once."""

        return f"""Survey page analysis. Mood: {self.mood}.

CRITICAL: If there are MULTIPLE statement comparison questions on the page, you MUST provide an action for EACH ONE.

RULES for {self.mood} mood:
- Ratings: {"Pick HIGHEST (5/5, 10/10, Highly Satisfied)" if self.mood == "happy" else "Pick LOWEST (1/5, 1/10, Dissatisfied)" if self.mood == "angry" else "Pick MIDDLE"}
- Statement A vs B: {"Choose positive/social/in-person/expert-help/service statements with 'much more'" if self.mood == "happy" else "Choose negative/independent/online/self-service statements with 'much more'" if self.mood == "angry" else "Choose balanced with 'somewhat more'"}

EXAMPLES for {self.mood} mood:
- "Statement A: prefer online" vs "Statement B: prefer in-store" → {"CLICK: Statement B describes me much more than Statement A" if self.mood == "happy" else "CLICK: Statement A describes me much more than Statement B"}
- "Statement A: expert help" vs "Statement B: self-service" → {"CLICK: Statement A describes me much more than Statement B" if self.mood == "happy" else "CLICK: Statement B describes me much more than Statement A"}

OUTPUT FORMAT - LIST ALL ACTIONS:
PAGE_TYPE: QUESTION
ACTIONS:
1. CLICK: [exact text of option to click]
2. CLICK: [exact text of second option if multiple questions]
3. CLICK: [exact text of third option if multiple questions]
HAS_NEXT: yes

NOW SCAN THE ENTIRE PAGE, FIND ALL UNANSWERED QUESTIONS, AND LIST ALL ACTIONS:"""
    
    async def _call_vision_llm(self, prompt: str, image_b64: str) -> str:
        """Call Ollama with optimized timeout."""

        timeout = httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": {
                        "num_predict": 800,  # Increased to allow for thinking + response
                        "temperature": 0.1,  # Low temperature for consistent responses
                    }
                }
            )

            if response.status_code != 200:
                logger.error(f"[OLLAMA ERROR] Status {response.status_code}: {response.text}")
                raise Exception(f"Ollama error: {response.status_code}")

            result = response.json()
            model_response = result.get("response", "")

            # CRITICAL FIX: Some models output to 'thinking' field instead of 'response'
            # Check if response is empty but thinking has content
            if not model_response or len(model_response.strip()) == 0:
                thinking = result.get("thinking", "")
                if thinking and len(thinking.strip()) > 0:
                    logger.warning(f"[OLLAMA FIX] Response was empty, using 'thinking' field ({len(thinking)} chars)")
                    model_response = thinking
                else:
                    logger.error(f"[OLLAMA ERROR] Both response and thinking are empty! Full result: {result}")

            logger.debug(f"[OLLAMA] Model returned {len(model_response)} chars")
            return model_response
    
    def _parse_batch_response(self, response: str) -> PageAnalysis:
        """Parse batch response into PageAnalysis."""

        logger.debug(f"[PARSER] Parsing response (length: {len(response)} chars)")

        response_upper = response.upper()
        
        # Detect page type
        page_type = "question"
        if "PAGE_TYPE: COMPLETE" in response_upper or "COMPLETE" in response_upper[:100]:
            return PageAnalysis(page_type="complete", is_complete=True)
        if "PAGE_TYPE: ERROR" in response_upper or "ERROR:" in response_upper[:100]:
            return PageAnalysis(page_type="error", error_message="Survey error detected")
        if "PAGE_TYPE: EMAIL" in response_upper:
            page_type = "email"
        if "PAGE_TYPE: TEXT" in response_upper:
            page_type = "text"
        
        # Parse actions
        actions = []
        for line in response.split("\n"):
            line = line.strip()
            
            # Match patterns like "1. SELECT: Highly Satisfied" or "SELECT: 10"
            if "SELECT:" in line.upper():
                target = line.split(":", 1)[-1].strip().strip("=").strip()
                if target:
                    actions.append(VisionAction(action_type="select", target=target))
            
            elif "FILL:" in line.upper():
                parts = line.split(":", 1)[-1]
                if "=" in parts:
                    target, value = parts.split("=", 1)
                    actions.append(VisionAction(
                        action_type="fill",
                        target=target.strip(),
                        value=value.strip(),
                    ))
            
            elif "CLICK:" in line.upper():
                target = line.split(":", 1)[-1].strip()
                if target:
                    logger.debug(f"[PARSER] Found CLICK action: {target}")
                    actions.append(VisionAction(action_type="click", target=target))

        logger.debug(f"[PARSER] Extracted {len(actions)} action(s): {[f'{a.action_type}:{a.target}' for a in actions]}")

        # Check for Next button
        has_next = "HAS_NEXT: YES" in response_upper or "HAS_NEXT:YES" in response_upper
        
        return PageAnalysis(
            page_type=page_type,
            actions=actions,
            has_next_button=has_next or len(actions) > 0,
        )
    
    async def execute_actions(self, page, analysis: PageAnalysis) -> bool:
        """Execute all actions from analysis."""

        if analysis.is_complete:
            return True

        if analysis.error_message:
            return False

        success_count = 0
        logger.info(f"[ACTION EXEC] Executing {len(analysis.actions)} actions for page type: {analysis.page_type}")

        for i, action in enumerate(analysis.actions):
            try:
                logger.info(f"[ACTION {i+1}/{len(analysis.actions)}] Executing: {action.action_type} -> target='{action.target}', value='{action.value}'")

                if action.action_type == "select":
                    # Pass both target and value (value may contain radio index)
                    if await self._do_select(page, action.target, action.value):
                        success_count += 1
                        logger.info(f"[ACTION SUCCESS {i+1}] Selected: {action.target}")
                        # Give time for the page to register the selection
                        await asyncio.sleep(0.2)
                    else:
                        logger.error(f"[ACTION FAILED {i+1}] Failed to select: {action.target}")

                elif action.action_type == "fill":
                    if await self._do_fill(page, action.target, action.value):
                        success_count += 1
                        logger.info(f"[ACTION SUCCESS {i+1}] Filled: {action.target}")
                        await asyncio.sleep(0.1)
                    else:
                        logger.error(f"[ACTION FAILED {i+1}] Failed to fill: {action.target}")

                elif action.action_type == "slider":
                    if await self._do_slider(page, action.target, action.value):
                        success_count += 1
                        logger.info(f"[ACTION SUCCESS {i+1}] Set slider: {action.target} to {action.value}")
                        await asyncio.sleep(0.2)
                    else:
                        logger.error(f"[ACTION FAILED {i+1}] Failed to set slider: {action.target}")

                elif action.action_type == "click":
                    if await self._do_click(page, action.target):
                        success_count += 1
                        logger.info(f"[ACTION SUCCESS {i+1}] Clicked: {action.target}")
                        # Track statement comparison clicks
                        if "statement" in action.target.lower():
                            if not hasattr(self, '_statement_clicks'):
                                self._statement_clicks = {}
                            page_key = page.url.split('?')[0]
                            self._statement_clicks[page_key] = self._statement_clicks.get(page_key, 0) + 1
                        await asyncio.sleep(0.2)
                    else:
                        logger.error(f"[ACTION FAILED {i+1}] Failed to click: {action.target}")

            except Exception as e:
                logger.warning(f"Action failed: {action} - {e}")

        # Check if we've answered enough statement comparison questions before clicking Next
        page_key = page.url.split('?')[0]
        expected_count = getattr(self, '_expected_statement_count', {}).get(page_key, 0)
        actual_count = getattr(self, '_statement_clicks', {}).get(page_key, 0)

        if expected_count > 0 and actual_count < expected_count:
            logger.warning(f"[STATEMENT CHECK] Answered {actual_count}/{expected_count} statement questions - need more answers before Next")
            # Don't click Next yet, return True to continue trying
            logger.info(f"[ACTION SUMMARY] Completed {success_count}/{len(analysis.actions)} actions successfully")
            return True  # Continue but don't click Next

        # Click Next if needed
        if analysis.has_next_button:
            # Wait longer to ensure all actions are processed (especially radio buttons with validation)
            logger.info(f"[NEXT BUTTON] Waiting 0.8s before clicking Next...")
            await asyncio.sleep(0.8)

            # Check if Next button is enabled
            is_next_enabled = await self._is_next_button_enabled(page)
            if not is_next_enabled:
                logger.warning("[NEXT BUTTON] Next button is disabled - may need to answer more questions")
                # Return False to indicate we couldn't proceed
                return False

            logger.info(f"[NEXT BUTTON] Next button is enabled, attempting to click...")
            next_clicked = await self._click_next(page)
            if next_clicked:
                logger.info("[NEXT SUCCESS] Clicked Next button")
                # Reset counters for this page when we successfully move forward
                if page_key in getattr(self, '_statement_clicks', {}):
                    del self._statement_clicks[page_key]
                if page_key in getattr(self, '_expected_statement_count', {}):
                    del self._expected_statement_count[page_key]
            else:
                logger.warning("[NEXT FAILED] No Next button found or click failed")
                return False

        logger.info(f"[ACTION SUMMARY] Completed {success_count}/{len(analysis.actions)} actions successfully")
        return success_count > 0 or analysis.has_next_button
    
    async def _do_select(self, page, target: str, value: str = None) -> bool:
        """Select a radio button or checkbox."""
        if not target:
            return False
        
        logger.debug(f"_do_select: target='{target}', value='{value}'")
        
        # If value contains a radio index, click it directly
        if value and value.isdigit():
            try:
                idx = int(value)
                radios = page.locator("input[type='radio']")
                count = await radios.count()
                logger.debug(f"[RADIO CLICK] Attempting to click radio at index {idx} of {count} total radios")
                if 0 <= idx < count:
                    radio = radios.nth(idx)

                    # Log radio attributes for debugging
                    try:
                        radio_name = await radio.get_attribute("name")
                        radio_value = await radio.get_attribute("value")
                        radio_id = await radio.get_attribute("id")
                        is_visible = await radio.is_visible()
                        logger.info(f"[RADIO DETAILS] Radio {idx}: name={radio_name}, value={radio_value}, id={radio_id}, visible={is_visible}")
                    except Exception as e:
                        logger.debug(f"Could not get radio attributes: {e}")

                    # Try multiple methods to click
                    clicked = False

                    # Method 1: Direct force click
                    try:
                        logger.debug(f"[RADIO CLICK] Method 1: Direct force click on radio {idx}")
                        await radio.click(force=True, timeout=2000)
                        clicked = True
                        logger.debug(f"[RADIO CLICK] Method 1 succeeded")
                    except Exception as e:
                        logger.debug(f"[RADIO CLICK] Method 1 failed: {e}")
                        # Method 2: Click via JavaScript
                        try:
                            logger.debug(f"[RADIO CLICK] Method 2: JavaScript click on radio {idx}")
                            await radio.evaluate("el => el.click()")
                            clicked = True
                            logger.debug(f"[RADIO CLICK] Method 2 succeeded")
                        except Exception as e:
                            logger.debug(f"[RADIO CLICK] Method 2 failed: {e}")
                            # Method 3: Click label if available
                            try:
                                logger.debug(f"[RADIO CLICK] Method 3: Label click for radio {idx}")
                                radio_id = await radio.get_attribute("id")
                                if radio_id:
                                    label = page.locator(f"label[for='{radio_id}']")
                                    if await label.count() > 0:
                                        await label.click()
                                        clicked = True
                                        logger.debug(f"[RADIO CLICK] Method 3 succeeded")
                                    else:
                                        logger.debug(f"[RADIO CLICK] Method 3: No label found for id={radio_id}")
                                else:
                                    logger.debug(f"[RADIO CLICK] Method 3: Radio has no id")
                            except Exception as e:
                                logger.debug(f"[RADIO CLICK] Method 3 failed: {e}")

                    if clicked:
                        # Verify it's actually selected
                        logger.debug(f"[RADIO VERIFY] Waiting 0.1s before verification...")
                        await asyncio.sleep(0.1)
                        is_checked = await radio.is_checked()
                        logger.info(f"[RADIO VERIFY] Radio {idx} is_checked = {is_checked}")
                        if is_checked:
                            logger.info(f"[RADIO SUCCESS] Verified radio button {idx} is selected")
                            return True
                        else:
                            logger.warning(f"[RADIO WARNING] Radio button {idx} clicked but not checked - attempting force-check")
                            # Try one more time with JavaScript
                            try:
                                logger.debug(f"[RADIO FORCE] Attempting to force-check radio {idx} with JavaScript")
                                await radio.evaluate("el => { el.checked = true; el.dispatchEvent(new Event('change', { bubbles: true })); }")
                                await asyncio.sleep(0.15)
                                is_checked = await radio.is_checked()
                                logger.info(f"[RADIO FORCE] After force-check, radio {idx} is_checked = {is_checked}")
                                if is_checked:
                                    logger.info(f"[RADIO SUCCESS] Force-checked radio button {idx}")
                                    return True
                                else:
                                    logger.error(f"[RADIO FAILURE] Force-check failed for radio {idx}")
                            except Exception as e:
                                logger.error(f"[RADIO FAILURE] Force-check exception for radio {idx}: {e}")
                    logger.error(f"[RADIO FAILURE] Failed to select radio button at index {idx}")
            except Exception as e:
                logger.debug(f"Index-based click failed: {e}")
        
        # Handle radio_N format (direct radio button selection by index)
        if target.startswith("radio_"):
            try:
                idx = int(target.split("_")[1])
                radios = page.locator("input[type='radio']")
                if idx < await radios.count():
                    radio = radios.nth(idx)
                    # Try clicking the radio or its parent
                    try:
                        await radio.click(force=True)
                        logger.info(f"Clicked radio button {idx} directly")
                        return True
                    except Exception:
                        # Try clicking parent element (sometimes radios are hidden)
                        parent = radio.locator("xpath=..")
                        await parent.click()
                        return True
            except Exception as e:
                logger.debug(f"Direct radio click failed: {e}")
        
        # Try label click first (most reliable)
        try:
            label = page.locator(f"label:has-text('{target}')").first
            if await label.is_visible(timeout=1000):
                await label.click()
                return True
        except Exception:
            pass
        
        # Try radio by value
        try:
            radio = page.locator(f"input[type='radio'][value='{target}' i]").first
            if await radio.is_visible(timeout=500):
                await radio.click()
                return True
        except Exception:
            pass
        
        # Try clicking radio buttons directly (for matrix-style questions)
        try:
            # Find all radio groups and select from unselected ones
            all_radios = page.locator("input[type='radio']")
            count = await all_radios.count()
            
            if count > 0:
                groups = {}
                for i in range(count):
                    radio = all_radios.nth(i)
                    name = await radio.get_attribute("name") or f"group_{i}"
                    if name not in groups:
                        groups[name] = []
                    groups[name].append(i)
                
                # Find first unselected group
                for group_name, indices in groups.items():
                    selected = await page.locator(f"input[name='{group_name}']:checked").count()
                    if selected == 0 and indices:
                        # Click based on mood: first for happy, last for angry
                        idx = indices[0] if self.mood == "happy" else indices[-1]
                        radio = all_radios.nth(idx)
                        await radio.click(force=True)
                        logger.info(f"Clicked radio {idx} in unselected group {group_name}")
                        return True
        except Exception as e:
            logger.debug(f"Group-based radio selection failed: {e}")
        
        # Try text match
        try:
            await page.click(f"text='{target}'", timeout=1000)
            return True
        except Exception:
            pass
        
        return False
    
    async def _do_fill(self, page, target: str, value: str) -> bool:
        """Fill a text input."""
        if not value:
            logger.warning("[FILL] No value provided to fill")
            return False

        target_lower = (target or "").lower()
        filled_count = 0

        selectors = []
        if "email" in target_lower:
            logger.debug(f"[FILL EMAIL] Looking for email fields to fill with: {value}")
            selectors = [
                "input[type='email']",
                "input[name*='email' i]",
                "input[id*='email' i]",
                "input[type='text']",  # Also try text inputs for email
            ]
        elif "brand" in target_lower:
            logger.debug(f"[FILL BRAND] Looking for text input fields to fill with: {value}")
            selectors = [
                "input[type='text']",
                "input[name*='brand' i]",
                "input[name*='retailer' i]",
                "input[placeholder*='brand' i]",
            ]
        elif "comment" in target_lower:
            logger.debug(f"[FILL COMMENT] Looking for comment/textarea fields")
            selectors = [
                "textarea",
                f"textarea[name*='{target}' i]",
                "input[type='text']",
            ]
        else:
            logger.debug(f"[FILL] Looking for '{target}' fields to fill")
            selectors = [
                f"input[name*='{target}' i]",
                f"textarea[name*='{target}' i]",
                "input[type='text']",
                "textarea",
            ]

        for selector in selectors:
            try:
                inputs = page.locator(selector)
                count = await inputs.count()
                logger.debug(f"[FILL] Selector '{selector}' found {count} field(s)")
                for i in range(count):
                    inp = inputs.nth(i)
                    is_visible = await inp.is_visible(timeout=500)
                    if is_visible:
                        current = await inp.input_value()
                        if not current:
                            await inp.fill(value)
                            filled_count += 1
                            logger.info(f"[FILL SUCCESS] Filled field {i+1} (selector: {selector}) with {value[:20]}...")
                            # For brand filling, only fill one field per action
                            if "brand" in target_lower:
                                return True
                        else:
                            logger.debug(f"[FILL SKIP] Field {i+1} already has value: {current[:20]}...")
                    else:
                        logger.debug(f"[FILL SKIP] Field {i+1} not visible")
            except Exception as e:
                logger.debug(f"[FILL ERROR] Selector '{selector}' failed: {e}")
                continue

        if filled_count > 0:
            logger.info(f"[FILL COMPLETE] Filled {filled_count} field(s) total")
        else:
            logger.warning(f"[FILL FAILED] No fields were filled for target '{target}'")
        return filled_count > 0

    async def _do_slider(self, page, target: str, value: str) -> bool:
        """Set a slider (input[type='range']) to a specific value."""
        if not value:
            logger.warning("[SLIDER] No value provided")
            return False

        try:
            # Get all sliders on the page
            sliders = page.locator("input[type='range']")
            count = await sliders.count()
            logger.debug(f"[SLIDER] Found {count} slider(s) on page")

            if count == 0:
                logger.warning("[SLIDER] No sliders found on page")
                return False

            # Extract index from target (e.g., "slider_0", "slider_1")
            slider_index = 0
            if "_" in target:
                try:
                    slider_index = int(target.split("_")[-1])
                except ValueError:
                    logger.warning(f"[SLIDER] Could not parse index from target '{target}', using 0")

            if slider_index >= count:
                logger.warning(f"[SLIDER] Index {slider_index} out of range (only {count} sliders)")
                return False

            slider = sliders.nth(slider_index)

            # Get slider attributes
            min_val = await slider.get_attribute("min") or "1"
            max_val = await slider.get_attribute("max") or "10"
            step_val = await slider.get_attribute("step") or "1"

            logger.info(f"[SLIDER] Slider {slider_index} attributes: min={min_val}, max={max_val}, step={step_val}")

            # Set the slider value using JavaScript
            # This is more reliable than clicking/dragging
            await slider.evaluate(f"el => el.value = '{value}'")

            # Trigger change events to ensure the page registers the change
            await slider.evaluate("el => el.dispatchEvent(new Event('input', { bubbles: true }))")
            await slider.evaluate("el => el.dispatchEvent(new Event('change', { bubbles: true }))")

            # Verify the value was set
            current_value = await slider.input_value()
            logger.info(f"[SLIDER SUCCESS] Set slider {slider_index} to {value} (verified: {current_value})")

            return True

        except Exception as e:
            logger.error(f"[SLIDER ERROR] Failed to set slider: {e}")
            return False

    async def _do_click(self, page, target: str) -> bool:
        """Click a button or clickable element with text."""
        if not target:
            logger.warning("[CLICK] No target provided")
            return False

        logger.debug(f"[CLICK] Attempting to click element with text: '{target}'")

        # Try multiple approaches to find and click the element
        selectors = [
            f"text='{target}'",  # Exact text match
            f"text=/{target}/i",  # Case-insensitive regex match
            f"button:has-text('{target}')",  # Button with text
            f"div:has-text('{target}')",  # Clickable div
            f"a:has-text('{target}')",  # Link with text
        ]

        # Also try partial text match for statement comparison questions
        if "statement" in target.lower():
            # Extract key phrases
            if "much more" in target.lower():
                selectors.append("div:has-text('much more')")
                selectors.append("text=/much more/i")
            elif "somewhat more" in target.lower():
                selectors.append("div:has-text('somewhat more')")
                selectors.append("text=/somewhat more/i")

        for selector in selectors:
            try:
                logger.debug(f"[CLICK] Trying selector: {selector}")
                # Find all matching elements
                elements = page.locator(selector)
                count = await elements.count()

                # For statement comparison, try to find an unclicked option
                if "statement" in target.lower() and count > 1:
                    logger.debug(f"[CLICK] Found {count} elements matching selector, looking for unclicked option")

                    # Try each element to find one that's not already selected
                    for i in range(count):
                        element = elements.nth(i)
                        try:
                            if await element.is_visible(timeout=1000):
                                # Check if this option is already selected (has a specific class or attribute)
                                classes = await element.get_attribute("class") or ""
                                aria_selected = await element.get_attribute("aria-selected")

                                # Skip if already selected
                                if "selected" in classes.lower() or aria_selected == "true":
                                    logger.debug(f"[CLICK] Element {i} already selected, trying next")
                                    continue

                                # Try to click this element
                                await element.click()
                                logger.info(f"[CLICK SUCCESS] Clicked element {i+1}/{count} using selector: {selector}")
                                try:
                                    await page.wait_for_load_state("networkidle", timeout=5000)
                                except Exception:
                                    await asyncio.sleep(0.5)  # Give time for page to update
                                return True
                        except Exception as e:
                            logger.debug(f"[CLICK] Element {i} failed: {e}")
                            continue
                else:
                    # Single element or non-statement question
                    element = elements.first
                    if await element.is_visible(timeout=1000):
                        await element.click()
                        logger.info(f"[CLICK SUCCESS] Clicked element using selector: {selector}")
                        try:
                            await page.wait_for_load_state("networkidle", timeout=5000)
                        except Exception:
                            await asyncio.sleep(0.5)  # Give time for page to update
                        return True
            except Exception as e:
                logger.debug(f"[CLICK] Selector '{selector}' failed: {e}")
                continue

        logger.error(f"[CLICK FAILED] Could not find or click element with text: '{target}'")
        return False
    
    async def _is_next_button_enabled(self, page) -> bool:
        """Check if Next button exists and is enabled."""
        selectors = [
            "button:has-text('Next')",
            "input[value='Next' i]",
            "button:has-text('Continue')",
            "input[value='Continue' i]",
            "button:has-text('Submit')",
            "input[type='submit']",
            "button:has-text('>>')",
            "button:has-text('>')",
        ]

        for selector in selectors:
            try:
                btn = page.locator(selector).first
                if await btn.is_visible(timeout=500):
                    # Check multiple disability indicators
                    disabled_attr = await btn.get_attribute("disabled")
                    aria_disabled = await btn.get_attribute("aria-disabled")
                    classes = await btn.get_attribute("class") or ""

                    # Check if button is disabled
                    if disabled_attr is not None:
                        logger.debug(f"[NEXT CHECK] Button has disabled attribute: {disabled_attr}")
                        return False
                    if aria_disabled == "true":
                        logger.debug(f"[NEXT CHECK] Button has aria-disabled=true")
                        return False
                    if "disabled" in classes.lower():
                        logger.debug(f"[NEXT CHECK] Button has 'disabled' class")
                        return False

                    logger.debug(f"[NEXT CHECK] Button is enabled")
                    return True
            except Exception as e:
                logger.debug(f"[NEXT CHECK] Selector '{selector}' check failed: {e}")
                continue

        logger.warning("[NEXT CHECK] No Next button found")
        return False

    async def _click_next(self, page) -> bool:
        """Click Next/Continue/Submit button."""
        selectors = [
            "button:has-text('Next')",
            "input[value='Next' i]",
            "button:has-text('Continue')",
            "input[value='Continue' i]",
            "button:has-text('Submit')",
            "input[type='submit']",
            "button:has-text('>>')",  # Some surveys use >> for next
            "button:has-text('>')",
        ]

        for selector in selectors:
            try:
                btn = page.locator(selector).first
                if await btn.is_visible(timeout=500):
                    disabled = await btn.get_attribute("disabled")
                    if not disabled:
                        await btn.click()
                        await page.wait_for_load_state("networkidle", timeout=10000)
                        return True
            except Exception:
                continue

        return False


# =============================================================================
# MAIN NAVIGATION FUNCTION
# =============================================================================

async def run_optimized_navigation(
    page,
    survey_url: str,
    mood: str = "happy",
    email: Optional[str] = None,
    survey_code: Optional[str] = None,
    max_steps: int = 200,
) -> dict:
    """
    Run optimized survey navigation.
    
    Much faster than standard navigation because:
    - Uses DOM heuristics to skip vision model calls
    - Batches all actions per page
    - Compresses screenshots
    - Limits unnecessary model calls
    """
    from ..browser.interactor import PageInteractor
    
    # Pre-flight Ollama check
    from .ollama_host import detect_ollama_host_async
    ollama_host = await detect_ollama_host_async()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.get(f"{ollama_host}/api/tags")
    except Exception:
        return {
            "is_complete": False,
            "error": "Ollama not running. Start with: ollama serve",
            "current_step": 0,
            "actions_taken": [],
        }
    
    navigator = OptimizedVisionNavigator(mood=mood, email=email)
    interactor = PageInteractor()

    result: Dict[str, Any] = {
        "is_complete": False,
        "coupon_code": None,
        "error": None,
        "current_step": 0,
        "actions_taken": [],
    }
    
    try:
        # Navigate to URL
        logger.info(f"Navigating to: {survey_url}")
        # Use domcontentloaded instead of networkidle for pages with continuous network activity
        # Some survey platforms (Groupon, etc.) have analytics/ads that prevent networkidle
        await page.goto(survey_url, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(1.0)  # Wait for dynamic content to load
        
        # Handle survey code entry
        if survey_code:
            body = await page.inner_text("body")
            if any(p in body.lower() for p in ["survey code", "enter the", "digit code"]):
                logger.info("Filling survey code...")
                if await interactor.fill_survey_code(page, survey_code):
                    # Click Start
                    try:
                        await page.click("text='Start'", timeout=3000)
                        await page.wait_for_load_state("networkidle")
                    except Exception:
                        pass
                    
                    # Check for error
                    body = await page.inner_text("body")
                    if any(e in body.lower() for e in ["already used", "invalid", "expired"]):
                        result["error"] = "Survey code already used or invalid"
                        return result
        
        # Main navigation loop
        vision_calls = 0
        max_vision_calls = 200  # Increased limit to handle surveys with 50+ slider questions
        last_url = None
        stuck_count = 0
        last_statement_count = 0
        last_page_hash = None
        max_stuck_attempts = 8  # Increased to allow answering multiple questions on same page

        for step in range(max_steps):
            result["current_step"] = step
            current_url = page.url

            # Get page content hash to detect changes even if URL stays same
            try:
                body_text = await page.inner_text("body")
                import hashlib
                current_page_hash = hashlib.md5(body_text.encode()).hexdigest()
            except Exception:
                current_page_hash = None

            # Check if we're making progress on statement questions
            page_key = current_url.split('?')[0]
            current_statement_count = getattr(navigator, '_statement_clicks', {}).get(page_key, 0)

            # Check if stuck on same page
            if current_url == last_url:
                # Check if page content changed (even if URL didn't)
                if current_page_hash and last_page_hash and current_page_hash != last_page_hash:
                    logger.info(f"[PROGRESS] Page content changed (URL same but content different)")
                    stuck_count = 0
                # If we're answering statement questions, don't count as stuck
                elif current_statement_count > last_statement_count:
                    logger.info(f"[PROGRESS] Answering statement questions: {current_statement_count} answered so far")
                    stuck_count = 0  # Reset since we're making progress
                else:
                    stuck_count += 1
                    logger.warning(f"[STUCK DETECTION] Same URL and content for {stuck_count} iterations: {current_url}")
                    if stuck_count >= max_stuck_attempts:
                        result["error"] = f"Stuck on same page after {stuck_count} attempts"
                        logger.error(f"[STUCK] Aborting - stuck on {current_url}")
                        break
            else:
                stuck_count = 0  # Reset counter when we move to a new page
                logger.info(f"[PROGRESS] URL changed from {last_url} to {current_url}")

            last_url = current_url
            last_statement_count = current_statement_count
            last_page_hash = current_page_hash

            # Analyze page
            # If we're stuck on a transition page for multiple iterations, force vision model check
            force_vision = (stuck_count >= 2 and last_url == current_url)

            if force_vision:
                logger.warning(f"[FORCE VISION] Stuck for {stuck_count} iterations, forcing vision model check")
                # Temporarily clear the DOM analysis to force vision model
                analysis = None
                try:
                    screenshot_b64 = await navigator._take_compressed_screenshot(page)
                    url = page.url
                    title = await page.title()
                    prompt = navigator._build_batch_prompt(url, title, step)
                    response = await navigator._call_vision_llm(prompt, screenshot_b64)
                    analysis = navigator._parse_batch_response(response)
                    vision_calls += 1
                    logger.info(f"[FORCE VISION] Step {step}: {analysis.page_type}, {len(analysis.actions)} actions")
                except Exception as e:
                    logger.error(f"[FORCE VISION] Failed: {e}")
                    # Fall back to normal analysis
                    analysis = await navigator.analyze_and_act(page, step)
            else:
                analysis = await navigator.analyze_and_act(page, step)

            # Track if we used vision model
            if analysis and analysis.page_type not in ("complete", "error", "transition", "email", "fallback"):
                vision_calls += 1

            result["actions_taken"].append(
                f"Step {step}: {analysis.page_type} - {len(analysis.actions)} actions"
            )

            # Check completion
            if analysis.is_complete:
                result["is_complete"] = True
                result["coupon_code"] = await _extract_coupon(page)
                logger.info(f"Survey complete! Coupon: {result['coupon_code']}")
                break

            # Check error
            if analysis.error_message:
                result["error"] = analysis.error_message
                logger.error(f"Survey error: {analysis.error_message}")
                break

            # Execute actions and check if Next button became enabled
            action_result = await navigator.execute_actions(page, analysis)

            # If execute_actions returned False (Next button was disabled), we might need more actions
            if action_result is False:
                logger.warning("[RETRY] Next button still disabled after actions, may need to answer more questions")
                # Don't increment stuck_count here, as we're actively trying different actions
                # The vision model or fallback should find remaining questions on next iteration

            # Small delay
            await asyncio.sleep(0.3)

            # Check if we've used too many vision calls
            if vision_calls >= max_vision_calls:
                logger.warning(f"Used {vision_calls} vision calls, reached limit")
                # Continue but stop using vision model - rely on fallbacks
        
        else:
            result["error"] = "Maximum steps reached"
        
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Navigation failed: {e}")
    
    return result


async def _extract_coupon(page) -> Optional[str]:
    """Extract coupon code from completion page."""
    
    # Blacklist of common words that aren't codes
    blacklist = {
        "COMPLETING", "COMPLETE", "COMPLETED", "SURVEY", "FEEDBACK", 
        "CUSTOMER", "SATISFACTION", "THANKYOU", "THANKS", "THANK",
        "RECEIPT", "RESTAURANT", "MCDONALDS", "VALIDATION", "VALIDATE",
        "REQUIRED", "OPTIONAL", "PARTICIPATING", "LOCATION", "LOCATIONS",
        "VISITING", "VISITED", "VISIT", "WELCOME", "VALUED",
        "SERVICE", "EXPERIENCE", "QUALITY", "MANAGER", "EMPLOYEE",
        "PRIVACY", "POLICY", "TERMS", "RESERVED", "RIGHTS",
        "PANDAEXPRESS", "TACOBELL", "CHICKFILA", "WENDYS", "SUBWAY",
        "START", "STARTED", "BEGIN", "CONTINUE", "SUBMIT",
        "ENGLISH", "SPANISH", "ESPANOL", "VERSION", "ACCESSIBILITY",
        "POWERED", "GROUP", "MANAGEMENT", "COPYRIGHT",
        "INDICATES", "INFORMATION", "ADDRESS", "CONFIRM", "PROVIDE",
        "RECEIVE", "PURPOSE", "CLICK", "COMPLETE", "FEEDBACK",
        # Thank you page words
        "CONNECTED", "REWARDS", "FORWARD", "SERVING", "APPRECIATE",
        "SOON", "AGAIN", "STAY", "LOOKING", "FORWARD",
    }
    
    try:
        text = await page.inner_text("body")
        text_upper = text.upper()
        
        # Priority 1: Look for explicit "Validation Code: XXXX" pattern
        validation_patterns = [
            r'VALIDATION\s*CODE[:\s]+(\d{6,12})',
            r'VALIDATION\s*CODE[:\s]+([A-Z0-9]{6,12})',
            r'COUPON\s*CODE[:\s]+([A-Z0-9]{6,12})',
            r'YOUR\s*CODE[:\s]+([A-Z0-9]{6,12})',
            r'REDEMPTION\s*CODE[:\s]+([A-Z0-9]{6,12})',
            r'CODE[:\s]+(\d{6,10})\b',
        ]
        
        for pattern in validation_patterns:
            match = re.search(pattern, text_upper)
            if match:
                code = match.group(1).strip()
                if code and code not in blacklist:
                    logger.info(f"Extracted validation code: {code}")
                    return code
        
        # Priority 2: General patterns (with blacklist check)
        general_patterns = [
            r'\b([A-Z]{2,4}[-]?\d{4,8})\b',
            r'\b(\d{4,8}[-]?[A-Z]{2,4})\b',
            r'\b(\d{6,8})\b',  # Pure numeric codes like McDonald's
        ]
        
        for pattern in general_patterns:
            matches = re.findall(pattern, text_upper)
            for code in matches:
                code_clean = code.replace("-", "").replace(" ", "")
                if code_clean not in blacklist and code not in blacklist:
                    # Must have at least one digit
                    if any(c.isdigit() for c in code):
                        logger.info(f"Extracted coupon code: {code}")
                        return code
        
        return None
    except Exception as e:
        logger.error(f"Coupon extraction error: {e}")
        return None
