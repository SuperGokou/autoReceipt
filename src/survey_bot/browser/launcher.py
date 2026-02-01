"""
Browser launcher and manager for survey automation.

This module provides the BrowserManager class which handles:
- Playwright browser lifecycle (launch/close)
- Page creation and configuration
- Navigation with error handling
- Screenshot capture on errors

Configuration is done via environment variables:
- BROWSER_HEADLESS: Run browser without GUI (default: False for debugging)
- BROWSER_SLOW_MO: Milliseconds to wait between actions (default: 0)
- SCREENSHOT_ON_ERROR: Save screenshot when errors occur (default: True)
- BROWSER_TIMEOUT: Default timeout in milliseconds (default: 30000)

Example Usage:
    >>> from survey_bot.browser.launcher import BrowserManager
    >>> 
    >>> async with BrowserManager() as page:
    ...     await page.goto("https://example.com")
    ...     print(await page.title())
    >>> 
    >>> # With custom config
    >>> async with BrowserManager(headless=True, slow_mo=100) as page:
    ...     await page.goto("https://survey.example.com")
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Literal, cast

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
    TimeoutError as PlaywrightTimeout,
    ViewportSize,
)


if TYPE_CHECKING:
    from types import TracebackType


__all__ = ["BrowserManager"]

logger = logging.getLogger(__name__)


# Default configuration
DEFAULT_VIEWPORT = {"width": 1280, "height": 720}
DEFAULT_TIMEOUT = 30000  # 30 seconds
SCREENSHOT_DIR = Path("screenshots")


class BrowserManager:
    """
    Async context manager for Playwright browser automation.
    
    Handles the complete browser lifecycle including launch, page creation,
    navigation, and cleanup. Designed for use with async with statement.
    
    Attributes:
        headless: Whether to run browser without GUI.
        slow_mo: Delay between actions in milliseconds.
        screenshot_on_error: Whether to capture screenshot on errors.
        timeout: Default timeout for operations in milliseconds.
        viewport: Browser viewport dimensions.
    
    Example:
        >>> async with BrowserManager() as page:
        ...     await page.goto("https://example.com")
        ...     title = await page.title()
    """
    
    def __init__(
        self,
        headless: Optional[bool] = None,
        slow_mo: Optional[int] = None,
        screenshot_on_error: Optional[bool] = None,
        timeout: Optional[int] = None,
        viewport: Optional[ViewportSize] = None,
        user_agent: Optional[str] = None,
    ) -> None:
        """
        Initialize the BrowserManager.
        
        Args:
            headless: Run without GUI. Env: BROWSER_HEADLESS (default: False)
            slow_mo: Ms delay between actions. Env: BROWSER_SLOW_MO (default: 0)
            screenshot_on_error: Save screenshot on error. Env: SCREENSHOT_ON_ERROR (default: True)
            timeout: Default timeout in ms. Env: BROWSER_TIMEOUT (default: 30000)
            viewport: Viewport dimensions (default: 1280x720)
            user_agent: Custom user agent string (default: realistic Chrome UA)
        """
        # Load from environment with fallback to parameters and defaults
        self.headless = self._get_bool_config(
            headless, "BROWSER_HEADLESS", default=False
        )
        self.slow_mo = self._get_int_config(
            slow_mo, "BROWSER_SLOW_MO", default=0
        )
        self.screenshot_on_error = self._get_bool_config(
            screenshot_on_error, "SCREENSHOT_ON_ERROR", default=True
        )
        self.timeout = self._get_int_config(
            timeout, "BROWSER_TIMEOUT", default=DEFAULT_TIMEOUT
        )
        self.viewport = viewport or cast(ViewportSize, DEFAULT_VIEWPORT)
        
        # Realistic user agent to avoid bot detection
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        
        # Internal state
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._error_occurred: bool = False
        
        logger.debug(
            f"BrowserManager initialized: headless={self.headless}, "
            f"slow_mo={self.slow_mo}, timeout={self.timeout}"
        )
    
    @staticmethod
    def _get_bool_config(
        value: Optional[bool],
        env_var: str,
        default: bool
    ) -> bool:
        """Get boolean config from parameter, env var, or default."""
        if value is not None:
            return value
        env_value = os.environ.get(env_var, "").lower()
        if env_value in ("true", "1", "yes"):
            return True
        elif env_value in ("false", "0", "no"):
            return False
        return default
    
    @staticmethod
    def _get_int_config(
        value: Optional[int],
        env_var: str,
        default: int
    ) -> int:
        """Get integer config from parameter, env var, or default."""
        if value is not None:
            return value
        env_value = os.environ.get(env_var, "")
        if env_value.isdigit():
            return int(env_value)
        return default
    
    async def __aenter__(self) -> Page:
        """
        Enter the async context manager.
        
        Launches Playwright, creates browser and page with configured settings.
        
        Returns:
            Configured Playwright Page object.
            
        Raises:
            RuntimeError: If browser fails to launch.
        """
        logger.info("Launching browser...")
        
        try:
            # Start Playwright
            self._playwright = await async_playwright().start()
            
            # Launch browser
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless,
                slow_mo=self.slow_mo,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-infobars",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ]
            )
            
            # Create context with settings to avoid bot detection
            self._context = await self._browser.new_context(
                viewport=self.viewport,
                user_agent=self.user_agent,
                locale="en-US",
                timezone_id="America/New_York",
                # Permissions to avoid popups
                permissions=["geolocation"],
            )
            
            # Set default timeout
            self._context.set_default_timeout(self.timeout)
            
            # Create page
            self._page = await self._context.new_page()
            
            # Add script to mask automation flags
            await self._page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """)
            
            logger.info(
                f"Browser launched: viewport={self.viewport}, "
                f"headless={self.headless}"
            )
            
            return self._page
            
        except Exception as e:
            logger.error(f"Failed to launch browser: {e}")
            await self._cleanup()
            raise RuntimeError(f"Browser launch failed: {e}") from e
    
    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        """
        Exit the async context manager.
        
        Handles cleanup and optional screenshot on error.
        
        Args:
            exc_type: Exception type if error occurred.
            exc_val: Exception value if error occurred.
            exc_tb: Exception traceback if error occurred.
            
        Returns:
            False (don't suppress exceptions).
        """
        # Check if error occurred
        if exc_type is not None:
            self._error_occurred = True
            logger.error(f"Error during browser session: {exc_val}")
            
            # Take screenshot on error if configured
            if self.screenshot_on_error and self._page:
                await self._save_error_screenshot(str(exc_val))
        
        # Clean up resources
        await self._cleanup()
        
        return False  # Don't suppress exceptions
    
    async def _cleanup(self) -> None:
        """Clean up browser resources."""
        logger.debug("Cleaning up browser resources...")
        
        try:
            if self._page:
                await self._page.close()
                self._page = None
        except Exception as e:
            logger.warning(f"Error closing page: {e}")
        
        try:
            if self._context:
                await self._context.close()
                self._context = None
        except Exception as e:
            logger.warning(f"Error closing context: {e}")
        
        try:
            if self._browser:
                await self._browser.close()
                self._browser = None
        except Exception as e:
            logger.warning(f"Error closing browser: {e}")
        
        try:
            if self._playwright:
                await self._playwright.stop()
                self._playwright = None
        except Exception as e:
            logger.warning(f"Error stopping playwright: {e}")
        
        logger.info("Browser resources cleaned up")
    
    async def _save_error_screenshot(self, error_msg: str) -> Optional[Path]:
        """
        Save a screenshot when an error occurs.
        
        Args:
            error_msg: Error message to include in filename.
            
        Returns:
            Path to saved screenshot or None if failed.
        """
        if not self._page:
            return None
        
        try:
            # Create screenshots directory
            SCREENSHOT_DIR.mkdir(exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_error = "".join(c if c.isalnum() else "_" for c in error_msg[:30])
            filename = f"error_{timestamp}_{safe_error}.png"
            filepath = SCREENSHOT_DIR / filename
            
            # Save screenshot
            await self._page.screenshot(path=str(filepath), full_page=True)
            logger.info(f"Error screenshot saved: {filepath}")
            
            return filepath
            
        except Exception as e:
            logger.warning(f"Failed to save error screenshot: {e}")
            return None
    
    async def navigate_to(
        self,
        url: str,
        wait_until: Literal["commit", "domcontentloaded", "load", "networkidle"] = "networkidle",
        handle_popups: bool = True,
    ) -> None:
        """
        Navigate to a URL with proper waiting and popup handling.
        
        Args:
            url: URL to navigate to.
            wait_until: Wait condition ('load', 'domcontentloaded', 'networkidle').
            handle_popups: Whether to dismiss common popups/modals.
            
        Raises:
            RuntimeError: If page is not initialized.
            PlaywrightTimeout: If navigation times out.
        """
        if not self._page:
            raise RuntimeError("Page not initialized. Use within 'async with' block.")
        
        logger.info(f"Navigating to: {url}")
        
        try:
            # Navigate with wait condition
            await self._page.goto(url, wait_until=wait_until)
            logger.debug(f"Navigation complete, waiting for {wait_until}")
            
            # Handle common popups if enabled
            if handle_popups:
                await self._dismiss_common_popups()
            
            logger.info(f"Successfully loaded: {await self._page.title()}")
            
        except PlaywrightTimeout as e:
            logger.error(f"Navigation timeout for {url}: {e}")
            if self.screenshot_on_error:
                await self._save_error_screenshot("navigation_timeout")
            raise
        except Exception as e:
            logger.error(f"Navigation error for {url}: {e}")
            raise
    
    async def _dismiss_common_popups(self) -> None:
        """
        Attempt to dismiss common popup types.
        
        Handles cookie consent, newsletter modals, and other common popups
        that might interfere with survey automation.
        """
        if not self._page:
            return
        
        # Common popup dismissal selectors
        popup_selectors = [
            # Cookie consent buttons
            "button:has-text('Accept')",
            "button:has-text('Accept All')",
            "button:has-text('I Accept')",
            "button:has-text('Got it')",
            "button:has-text('OK')",
            "[aria-label='Accept cookies']",
            "#cookie-accept",
            ".cookie-accept",
            
            # Close buttons for modals
            "button:has-text('Close')",
            "button:has-text('No Thanks')",
            "button:has-text('Maybe Later')",
            "[aria-label='Close']",
            ".modal-close",
            ".popup-close",
            
            # Newsletter/signup dismissals
            "button:has-text('No thanks')",
            "button:has-text('Skip')",
        ]
        
        for selector in popup_selectors:
            try:
                # Quick check if element exists (with short timeout)
                element = self._page.locator(selector).first
                if await element.is_visible(timeout=500):
                    await element.click()
                    logger.debug(f"Dismissed popup: {selector}")
                    # Wait briefly for popup to close
                    await self._page.wait_for_timeout(300)
            except Exception:
                # Silently continue - popup might not exist
                pass
    
    async def take_screenshot(self, name: str = "screenshot") -> Path:
        """
        Take a screenshot of the current page.
        
        Args:
            name: Base name for the screenshot file.
            
        Returns:
            Path to the saved screenshot.
            
        Raises:
            RuntimeError: If page is not initialized.
        """
        if not self._page:
            raise RuntimeError("Page not initialized.")
        
        SCREENSHOT_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.png"
        filepath = SCREENSHOT_DIR / filename
        
        await self._page.screenshot(path=str(filepath), full_page=True)
        logger.info(f"Screenshot saved: {filepath}")
        
        return filepath
    
    async def wait_for_navigation(self, timeout: Optional[int] = None) -> None:
        """
        Wait for navigation to complete.
        
        Args:
            timeout: Custom timeout in milliseconds.
        """
        if not self._page:
            return
        
        await self._page.wait_for_load_state(
            "networkidle",
            timeout=timeout or self.timeout
        )
    
    @property
    def page(self) -> Optional[Page]:
        """Get the current page object."""
        return self._page
    
    @property
    def is_open(self) -> bool:
        """Check if browser is currently open."""
        return self._browser is not None and self._browser.is_connected()
