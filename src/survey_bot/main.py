"""
Survey Bot Main Module - Complete Survey Automation Pipeline.

This module provides the SurveyBot class which orchestrates the complete
survey automation flow:
1. Ingestion: Extract survey URL from receipt image
2. Supervisor: Configure persona based on mood
3. Navigator: Navigate and complete the survey
4. Fulfillment: Extract coupon and send via email

CLI Usage:
    $ python -m survey_bot run --image receipt.jpg --mood happy --email user@email.com
    $ python -m survey_bot run -i receipt.png -m angry -e test@example.com --verbose

Example Python Usage:
    >>> from survey_bot.main import SurveyBot
    >>> 
    >>> bot = SurveyBot(verbose=True)
    >>> result = await bot.run(
    ...     image_path="receipt.jpg",
    ...     mood="happy",
    ...     email="user@example.com"
    ... )
    >>> 
    >>> if result.success:
    ...     print(f"Coupon: {result.coupon.code}")
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import click

from .agents.ingestion import IngestionAgent
from .agents.supervisor import SupervisorAgent
from .agents.fulfillment import FulfillmentAgent
from .agents.formspree_sender import FormspreeSender
from .browser.launcher import BrowserManager
from .llm.graph import run_survey_navigation
from .llm.optimized_navigator import run_optimized_navigation
from .models.survey_result import SurveyResult, CouponResult
from .models.receipt import ReceiptData


__all__ = ["SurveyBot", "run_cli"]

logger = logging.getLogger(__name__)


# =============================================================================
# PROGRESS INDICATORS
# =============================================================================

class ProgressIndicator:
    """Helper class for progress messages."""
    
    ICONS = {
        "image": "[Image]",
        "persona": "[Mood]",
        "navigate": "[Navigate]",
        "email": "[Email]",
        "success": "[OK]",
        "error": "[ERROR]",
        "warning": "[WARN]",
        "info": "[INFO]",
    }
    
    def __init__(self, verbose: bool = True, callback: Optional[Callable] = None):
        """
        Initialize progress indicator.
        
        Args:
            verbose: Whether to print progress messages.
            callback: Optional callback for custom progress handling.
        """
        self.verbose = verbose
        self.callback = callback
    
    def update(self, stage: str, message: str) -> None:
        """
        Update progress.
        
        Args:
            stage: Stage identifier (image, persona, navigate, email, etc.)
            message: Progress message.
        """
        icon = self.ICONS.get(stage, "•")
        full_message = f"{icon} {message}"
        
        if self.verbose:
            click.echo(full_message)
        
        if self.callback:
            self.callback(stage, message)
        
        logger.info(message)
    
    def success(self, message: str) -> None:
        """Show success message."""
        self.update("success", message)
    
    def error(self, message: str) -> None:
        """Show error message."""
        self.update("error", message)
    
    def warning(self, message: str) -> None:
        """Show warning message."""
        self.update("warning", message)


# =============================================================================
# SURVEY BOT CLASS
# =============================================================================

class SurveyBot:
    """
    Main orchestrator for the survey automation pipeline.
    
    Coordinates all four modules to:
    1. Extract survey URL from receipt image
    2. Configure persona based on user mood
    3. Navigate and complete the survey
    4. Extract coupon and send via email
    
    Attributes:
        verbose: Whether to show progress messages.
        headless: Whether to run browser in headless mode.
        screenshot_dir: Directory for saving screenshots.
    """
    
    def __init__(
        self,
        verbose: bool = True,
        headless: bool = True,
        screenshot_dir: Optional[Path] = None,
        progress_callback: Optional[Callable] = None,
        use_vision: bool = True,
    ) -> None:
        """
        Initialize SurveyBot.
        
        Args:
            verbose: Show progress messages in console.
            headless: Run browser without visible window.
            screenshot_dir: Directory for screenshots.
            progress_callback: Optional callback for progress updates.
            use_vision: Use Qwen3-VL vision model for navigation (recommended).
        """
        self.verbose = verbose
        self.headless = headless
        self.screenshot_dir = screenshot_dir or Path("screenshots")
        self.use_vision = use_vision
        
        # Create progress indicator
        self.progress = ProgressIndicator(verbose=verbose, callback=progress_callback)
        
        # Initialize agents
        self.ingestion_agent = IngestionAgent()
        self.supervisor_agent = SupervisorAgent()
        self.fulfillment_agent = FulfillmentAgent(screenshot_dir=self.screenshot_dir)
        
        # Ensure screenshot directory exists
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"SurveyBot initialized: headless={headless}")
    
    async def run(
        self,
        image_path: str,
        mood: str,
        email: str,
        max_steps: int = 200,
    ) -> SurveyResult:
        """
        Run the complete survey automation pipeline.
        
        Args:
            image_path: Path to receipt image.
            mood: User mood (happy, neutral, angry).
            email: Email address for coupon delivery.
            max_steps: Maximum navigation steps.
            
        Returns:
            SurveyResult with completion status and coupon.
        """
        start_time = datetime.now()
        actions_log = []
        survey_url = ""
        
        try:
            # =================================================================
            # STEP 1: INGESTION - Extract URL from receipt
            # =================================================================
            self.progress.update("image", "Extracting URL from receipt...")
            actions_log.append("Starting URL extraction from receipt")
            
            extraction_result = await self.ingestion_agent.extract_from_image(image_path)

            if not extraction_result.success or extraction_result.data is None:
                self.progress.error(f"Failed to extract URL: {extraction_result.error}")
                return SurveyResult(
                    success=False,
                    survey_url="",
                    receipt_image_path=str(image_path),
                    start_time=start_time,
                    end_time=datetime.now(),
                    actions_log=actions_log,
                    error_message=f"URL extraction failed: {extraction_result.error}",
                )

            survey_url = extraction_result.data.url or extraction_result.data.survey_url or ""
            survey_code: Optional[str] = extraction_result.data.survey_code  # Get survey code from receipt
            actions_log.append(f"Extracted URL: {survey_url}")
            if survey_code:
                actions_log.append(f"Extracted survey code: {survey_code}")
            self.progress.success(f"Found survey URL: {survey_url[:50]}...")
            
            # =================================================================
            # STEP 2: SUPERVISOR - Configure persona
            # =================================================================
            self.progress.update("persona", f"Setting up {mood.capitalize()} persona...")
            actions_log.append(f"Configuring {mood} persona")
            
            persona = self.supervisor_agent.get_persona(mood)
            actions_log.append(f"Persona configured: {persona.mood.value}")
            self.progress.success(f"Persona ready: {persona.mood.value} (rating preference: {persona.rating_preference})")
            
            # =================================================================
            # STEP 3: NAVIGATOR - Complete the survey
            # =================================================================
            self.progress.update("navigate", "Navigating survey...")
            actions_log.append("Starting survey navigation")
            
            # Use browser manager context
            browser_manager = BrowserManager(
                headless=self.headless,
            )
            
            coupon_result = None
            nav_state: Optional[dict] = None

            async with browser_manager as page:
                # Choose navigation method
                if self.use_vision:
                    # Use optimized Qwen3-VL vision-based navigation
                    self.progress.update("navigate", "Using AI vision for navigation...")
                    nav_state = await run_optimized_navigation(
                        page=page,
                        survey_url=survey_url,
                        mood=mood,
                        email=email,
                        survey_code=survey_code,
                        max_steps=max_steps,
                    )
                else:
                    # Use rule-based navigation
                    nav_state = await run_survey_navigation(
                        page=page,
                        survey_url=survey_url,
                        persona=persona,
                        max_steps=max_steps,
                        survey_code=survey_code,
                        email=email,
                    )

                # Log actions taken
                if nav_state and nav_state.get('actions_taken'):
                    actions_log.extend(nav_state.get('actions_taken', []))
                
                # Check if survey was completed
                if nav_state and nav_state.get('is_complete') and not nav_state.get('error'):
                    actions_log.append("Survey navigation completed")
                    self.progress.success(f"Survey completed in {nav_state.get('current_step', 0)} steps")
                    
                    # =================================================================
                    # STEP 4: FULFILLMENT - Extract coupon and send email
                    # =================================================================
                    self.progress.update("email", "Extracting coupon code...")

                    # PRIORITY 1: Check if navigation already found a coupon code
                    # This is more reliable than text pattern matching
                    if nav_state and nav_state.get('coupon_code'):
                        from .models.survey_result import CouponPattern, ExtractionMethod
                        coupon_code_from_nav: str = nav_state.get('coupon_code', '')
                        if coupon_code_from_nav:
                            logger.info(f"Using coupon from navigation: {coupon_code_from_nav}")
                            coupon_result = CouponResult(
                                code=coupon_code_from_nav,
                                confidence=0.95,
                                pattern_type=CouponPattern.NUMERIC_ONLY if coupon_code_from_nav.isdigit() else CouponPattern.GENERIC,
                                extraction_method=ExtractionMethod.ELEMENT_SEARCH,
                            )
                            # Still take a screenshot for proof
                            screenshot_path = await self.fulfillment_agent._take_coupon_screenshot(page, coupon_code_from_nav)
                            coupon_result.screenshot_path = screenshot_path
                        else:
                            coupon_result = None
                    else:
                        # PRIORITY 2: Only use fulfillment agent if navigation didn't find a code
                        coupon_result = await self.fulfillment_agent.detect_coupon(
                            page,
                            take_screenshot=True,
                        )
                    
                    if coupon_result:
                        actions_log.append(f"Coupon extracted: {coupon_result.code}")
                        self.progress.success(f"Coupon found: {coupon_result.code}")

                        # Check email sending method
                        formspree_endpoint = os.environ.get("FORMSPREE_ENDPOINT", "")
                        send_via_smtp = os.environ.get("SEND_EMAIL_VIA_SMTP", "false").lower() == "true"

                        logger.info(f"[EMAIL CONFIG] Formspree endpoint: {'SET' if formspree_endpoint else 'NOT SET'}")
                        logger.info(f"[EMAIL CONFIG] Email address: {'SET' if email else 'NOT SET'} (value: {email})")
                        logger.info(f"[EMAIL CONFIG] SMTP enabled: {send_via_smtp}")

                        if formspree_endpoint and email:
                            # Send coupon via Formspree (simpler than SMTP!)
                            logger.info(f"[FORMSPREE] Attempting to send email via Formspree to {email}")
                            logger.info(f"[FORMSPREE] Endpoint: {formspree_endpoint}")
                            logger.info(f"[FORMSPREE] Coupon code: {coupon_result.code}")
                            self.progress.update("email", "Sending coupon via Formspree...")
                            try:
                                formspree = FormspreeSender(formspree_endpoint)
                                logger.debug(f"[FORMSPREE] FormspreeSender initialized")
                                result = await formspree.send_coupon_email(
                                    recipient_email=email,
                                    coupon_code=coupon_result.code,
                                    store_name=extraction_result.data.store_name if extraction_result.data else None,
                                    survey_url=survey_url,
                                    screenshot_path=coupon_result.screenshot_path,
                                )
                                logger.info(f"[FORMSPREE] Send result: {result}")
                                if result["success"]:
                                    email_sent = True
                                    actions_log.append(f"Coupon sent to {email} via Formspree")
                                    self.progress.success(f"Email sent to {email}!")
                                else:
                                    email_sent = False
                                    actions_log.append(f"Formspree failed: {result['message']}")
                                    self.progress.warning(f"Email failed: {result['message']}")
                            except Exception as e:
                                email_sent = False
                                actions_log.append(f"Formspree error: {str(e)}")
                                self.progress.warning(f"Email error: {str(e)}")
                        elif send_via_smtp and email:
                            # Send coupon via SMTP
                            self.progress.update("email", "Sending coupon via email...")
                            try:
                                email_result = await self.fulfillment_agent.send_email(
                                    recipient=email,
                                    coupon=coupon_result,
                                    receipt_data=extraction_result.data,
                                    attach_screenshot=True,
                                )
                                if email_result.success:
                                    email_sent = True
                                    actions_log.append(f"Coupon sent to {email} via SMTP")
                                    self.progress.success(f"Email sent to {email}!")
                                else:
                                    email_sent = False
                                    actions_log.append(f"Email failed: {email_result.error_message}")
                                    self.progress.warning(f"Email failed: {email_result.error_message}")
                            except Exception as e:
                                email_sent = False
                                actions_log.append(f"Email error: {str(e)}")
                                self.progress.warning(f"Email error: {str(e)}")
                        else:
                            # Survey website handles email delivery
                            logger.info(f"[EMAIL] No Formspree endpoint or SMTP configured, relying on survey website to send email")
                            if not formspree_endpoint:
                                logger.warning(f"[EMAIL WARNING] FORMSPREE_ENDPOINT is not set in .env")
                            if not email:
                                logger.warning(f"[EMAIL WARNING] No email address provided")
                            email_sent = True
                            actions_log.append(f"Coupon will be sent to {email} by the survey website")
                            self.progress.success(f"Survey completed! Check {email} for your coupon.")
                    else:
                        # No coupon displayed on page - some surveys send via email only
                        # Check if the page looks like a thank you page
                        page_text = await page.inner_text("body")
                        page_lower = page_text.lower()
                        
                        is_thank_you_page = any(phrase in page_lower for phrase in [
                            "thank you", "thanks for", "appreciate your feedback",
                            "feedback has been", "survey complete"
                        ])
                        
                        if is_thank_you_page and email:
                            # Survey completed but coupon sent via email (common for Panda Express)
                            email_sent = True
                            actions_log.append(f"Survey completed! Coupon will be sent to {email}")
                            self.progress.success(f"Survey completed! Coupon will be emailed to {email}")
                            
                            # Create a placeholder result
                            from .models.survey_result import CouponPattern, ExtractionMethod
                            coupon_result = CouponResult(
                                code="EMAILED",
                                confidence=0.5,
                                pattern_type=CouponPattern.GENERIC,
                                extraction_method=ExtractionMethod.LLM_EXTRACTION,
                                raw_text="Coupon sent via email",
                            )
                        else:
                            email_sent = False
                            actions_log.append("No coupon code found on completion page")
                            self.progress.warning("No coupon code found")
                else:
                    email_sent = False
                    error_msg = (nav_state.get('error') if nav_state else None) or "Survey not completed"
                    actions_log.append(f"Navigation failed: {error_msg}")
                    self.progress.error(f"Navigation failed: {error_msg}")
            
            # Build final result
            return SurveyResult(
                success=coupon_result is not None,
                coupon=coupon_result,
                email_sent=email_sent,
                survey_url=survey_url,
                receipt_image_path=str(image_path),
                start_time=start_time,
                end_time=datetime.now(),
                steps_taken=nav_state.get('current_step', 0) if nav_state else 0,
                actions_log=actions_log,
                error_message=nav_state.get('error') if nav_state else None,
            )
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.progress.error(error_msg)
            logger.exception("Survey automation failed")
            
            return SurveyResult(
                success=False,
                survey_url=survey_url,
                receipt_image_path=str(image_path),
                start_time=start_time,
                end_time=datetime.now(),
                actions_log=actions_log,
                error_message=error_msg,
            )
    
    def _extract_store_name(self, url: str) -> Optional[str]:
        """
        Extract store name from survey URL.
        
        Args:
            url: Survey URL.
            
        Returns:
            Store name if detected, None otherwise.
        """
        url_lower = url.lower()
        
        store_patterns = {
            "walmart": ["walmart", "tellwm"],
            "McDonald's": ["mcdonald", "mcdvoice"],
            "Wendy's": ["wendy", "talktowendys"],
            "Taco Bell": ["tacobell", "tellthebell"],
            "Burger King": ["burgerking", "mybkexperience"],
            "Subway": ["subway", "tellsubway"],
            "Starbucks": ["starbucks", "mystarbucksvisit"],
            "Chick-fil-A": ["chickfila", "mycfavisit"],
            "Target": ["target", "informtarget"],
        }
        
        for store_name, patterns in store_patterns.items():
            if any(p in url_lower for p in patterns):
                return store_name
        
        return None
    
    async def run_with_url(
        self,
        survey_url: str,
        mood: str,
        email: str,
        max_steps: int = 200,
    ) -> SurveyResult:
        """
        Run survey automation with a direct URL (skip ingestion).
        
        Useful for testing or when URL is already known.
        
        Args:
            survey_url: Direct survey URL.
            mood: User mood.
            email: Email address.
            max_steps: Maximum navigation steps.
            
        Returns:
            SurveyResult with completion status.
        """
        start_time = datetime.now()
        actions_log = [f"Starting with direct URL: {survey_url}"]
        
        try:
            # Configure persona
            self.progress.update("persona", f"Setting up {mood.capitalize()} persona...")
            persona = self.supervisor_agent.get_persona(mood)
            actions_log.append(f"Persona configured: {persona.mood.value}")
            
            # Navigate survey
            self.progress.update("navigate", "Navigating survey...")
            
            browser_manager = BrowserManager(
                headless=self.headless,
            )
            
            coupon_result = None
            nav_state = None
            email_sent = False
            
            async with browser_manager as page:
                # Choose navigation method
                if self.use_vision:
                    # Use optimized Qwen3-VL vision-based navigation
                    self.progress.update("navigate", "Using AI vision for navigation...")
                    nav_state = await run_optimized_navigation(
                        page=page,
                        survey_url=survey_url,
                        mood=mood,
                        email=email,
                        max_steps=max_steps,
                    )
                else:
                    # Use rule-based navigation
                    nav_state = await run_survey_navigation(
                        page=page,
                        survey_url=survey_url,
                        persona=persona,
                        max_steps=max_steps,
                        email=email,
                    )

                if nav_state and nav_state.get('actions_taken'):
                    actions_log.extend(nav_state.get('actions_taken', []))

                if nav_state and nav_state.get('is_complete') and not nav_state.get('error'):
                    self.progress.update("email", "Extracting coupon...")

                    coupon_result = await self.fulfillment_agent.detect_coupon(
                        page,
                        take_screenshot=True,
                    )

                    if not coupon_result and nav_state.get('coupon_code'):
                        from .models.survey_result import CouponPattern, ExtractionMethod
                        coupon_code_from_nav_2: str = nav_state.get('coupon_code', '')
                        if coupon_code_from_nav_2:
                            coupon_result = CouponResult(
                                code=coupon_code_from_nav_2,
                                confidence=0.9,
                            )
                    
                    if coupon_result:
                        # Survey website sends coupon directly to email entered in form
                        email_sent = True
                        self.progress.success(f"Survey completed! Check {email} for your coupon.")
            
            return SurveyResult(
                success=coupon_result is not None,
                coupon=coupon_result,
                email_sent=email_sent,
                survey_url=survey_url,
                start_time=start_time,
                end_time=datetime.now(),
                steps_taken=nav_state.get('current_step', 0) if nav_state else 0,
                actions_log=actions_log,
                error_message=nav_state.get('error') if nav_state else None,
            )
            
        except Exception as e:
            self.progress.error(f"Error: {e}")
            return SurveyResult(
                success=False,
                survey_url=survey_url,
                start_time=start_time,
                end_time=datetime.now(),
                actions_log=actions_log,
                error_message=str(e),
            )


# =============================================================================
# CLI INTERFACE
# =============================================================================

@click.group()
@click.version_option(version="0.1.0", prog_name="survey-bot")
def cli():
    """
    Survey Bot - Automated Survey Completion System.
    
    Complete customer satisfaction surveys automatically based on
    receipt images and user-specified mood preferences.
    """
    pass


@cli.command()
@click.option(
    "-i", "--image",
    type=click.Path(exists=True),
    required=True,
    help="Path to receipt image (JPG, PNG).",
)
@click.option(
    "-m", "--mood",
    type=click.Choice(["happy", "neutral", "angry"], case_sensitive=False),
    default="happy",
    help="Survey response mood (default: happy).",
)
@click.option(
    "-e", "--email",
    required=True,
    help="Email address for coupon delivery.",
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    default=True,
    help="Show progress messages.",
)
@click.option(
    "--headless/--no-headless",
    default=True,
    help="Run browser in headless mode (default: yes).",
)
@click.option(
    "--max-steps",
    type=int,
    default=50,
    help="Maximum navigation steps (default: 50).",
)
@click.option(
    "--screenshot-dir",
    type=click.Path(),
    default="screenshots",
    help="Directory for screenshots (default: screenshots).",
)
def run(
    image: str,
    mood: str,
    email: str,
    verbose: bool,
    headless: bool,
    max_steps: int,
    screenshot_dir: str,
):
    """
    Run complete survey automation from receipt image.
    
    Example:
    
        $ python -m survey_bot run -i receipt.jpg -m happy -e user@email.com
        
        $ python -m survey_bot run --image photo.png --mood angry --email test@test.com --no-headless
    """
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Validate email format
    if not _is_valid_email(email):
        click.echo(click.style(f"[ERROR] Invalid email address: {email}", fg="red"))
        sys.exit(1)
    
    click.echo()
    click.echo(click.style("=" * 50, fg="blue"))
    click.echo(click.style("  SURVEY BOT - Automated Survey Completion", fg="blue", bold=True))
    click.echo(click.style("=" * 50, fg="blue"))
    click.echo()
    
    # Show configuration
    if verbose:
        click.echo(f"[Image] Image: {image}")
        click.echo(f"[Mood]Mood: {mood}")
        click.echo(f"Email: {email}")
        click.echo(f"[Mode] Headless: {headless}")
        click.echo()
    
    # Create and run bot
    bot = SurveyBot(
        verbose=verbose,
        headless=headless,
        screenshot_dir=Path(screenshot_dir),
    )
    
    try:
        result = asyncio.run(bot.run(
            image_path=image,
            mood=mood,
            email=email,
            max_steps=max_steps,
        ))
        
        # Show results
        click.echo()
        click.echo(click.style("=" * 50, fg="blue"))
        click.echo(click.style("  RESULTS", fg="blue", bold=True))
        click.echo(click.style("=" * 50, fg="blue"))
        click.echo()
        
        if result.success:
            click.echo(click.style("[SUCCESS] Survey completed successfully!", fg="green", bold=True))
            click.echo()
            
            if result.coupon:
                click.echo(f"[Coupon] Coupon Code: {click.style(result.coupon.code, fg='yellow', bold=True)}")
                click.echo(f"[Score] Confidence: {result.coupon.confidence:.0%}")
                if result.coupon.screenshot_path:
                    click.echo(f"[Screenshot] Screenshot: {result.coupon.screenshot_path}")
            
            click.echo(f"[Time] Duration: {result.duration_seconds:.1f}s")
            click.echo(f"[Steps] Steps: {result.steps_taken}")
        else:
            click.echo(click.style("[FAILED] Survey failed", fg="red", bold=True))
            click.echo()
            
            if result.error_message:
                click.echo(f"Error: {result.error_message}")
        
        click.echo()
        
        # Exit with appropriate code
        sys.exit(0 if result.success else 1)
        
    except KeyboardInterrupt:
        click.echo()
        click.echo(click.style("Interrupted by user", fg="yellow"))
        sys.exit(130)
    except Exception as e:
        click.echo()
        click.echo(click.style(f"[ERROR] Fatal error: {e}", fg="red"))
        logger.exception("Fatal error")
        sys.exit(1)


@cli.command()
@click.option(
    "-u", "--url",
    required=True,
    help="Direct survey URL to complete.",
)
@click.option(
    "-m", "--mood",
    type=click.Choice(["happy", "neutral", "angry"], case_sensitive=False),
    default="happy",
    help="Survey response mood.",
)
@click.option(
    "-e", "--email",
    required=True,
    help="Email address for coupon delivery.",
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    default=True,
    help="Show progress messages.",
)
@click.option(
    "--headless/--no-headless",
    default=True,
    help="Run browser in headless mode.",
)
def direct(url: str, mood: str, email: str, verbose: bool, headless: bool):
    """
    Run survey with direct URL (skip receipt image processing).
    
    Example:
    
        $ python -m survey_bot direct -u https://survey.example.com/abc -m happy -e user@email.com
    """
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    
    click.echo()
    click.echo(click.style("Running survey with direct URL...", fg="blue"))
    click.echo()
    
    bot = SurveyBot(verbose=verbose, headless=headless)
    
    result = asyncio.run(bot.run_with_url(
        survey_url=url,
        mood=mood,
        email=email,
    ))
    
    if result.success:
        click.echo(click.style(f"[SUCCESS] Coupon: {result.coupon.code if result.coupon else 'N/A'}", fg="green"))
    else:
        click.echo(click.style(f"[FAILED] {result.error_message}", fg="red"))
    
    sys.exit(0 if result.success else 1)


@cli.command()
def version():
    """Show version information."""
    click.echo("Survey Bot v0.1.0")
    click.echo("Automated Survey Completion System")
    click.echo()
    click.echo("Modules:")
    click.echo("  - Ingestion Agent: Receipt image processing")
    click.echo("  - Supervisor Agent: Persona management")
    click.echo("  - Navigation Graph: Survey completion")
    click.echo("  - Fulfillment Agent: Coupon extraction & email")


@cli.command()
@click.option(
    "-p", "--port",
    type=int,
    default=5000,
    help="Port to run on (default: 5000).",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug mode.",
)
def web(port: int, debug: bool):
    """
    Start the web UI server.
    
    Example:
    
        $ python -m survey_bot web
        
        $ python -m survey_bot web --port 8080 --debug
    """
    try:
        from .web.app import app
        
        click.echo()
        click.echo(click.style("=" * 63, fg="cyan"))
        click.echo(click.style("           SURVEY BOT WEB UI                              ", fg="cyan"))
        click.echo(click.style("=" * 63, fg="cyan"))
        click.echo(click.style(f"  Open in browser: http://localhost:{port}                      ", fg="cyan"))
        click.echo(click.style("  Press Ctrl+C to stop                                     ", fg="cyan"))
        click.echo(click.style("=" * 63, fg="cyan"))
        click.echo()
        
        app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)
        
    except ImportError as e:
        click.echo(click.style(f"Error: {e}", fg="red"))
        click.echo("Make sure Flask is installed: pip install flask")
        sys.exit(1)


def _is_valid_email(email: str) -> bool:
    """Validate email format."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def run_cli():
    """Entry point for CLI."""
    cli()


# =============================================================================
# MODULE ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_cli()
