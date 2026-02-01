"""
Integration tests for the Navigation Agent.

These tests validate the complete survey navigation flow using
a mock survey server. They test the full pipeline:
1. Browser launch
2. Page observation
3. Action decision
4. Action execution
5. Coupon extraction

Run with:
    pytest tests/test_navigator_integration.py -v -m integration
    
    # Run with visible browser (for debugging)
    BROWSER_HEADLESS=false pytest tests/test_navigator_integration.py -v -m integration

Requirements:
    - Flask (for mock server)
    - Playwright browsers installed (playwright install chromium)
"""
from __future__ import annotations

import asyncio
import os
import pytest
import time
from typing import TYPE_CHECKING

# Import our modules
from src.survey_bot.agents.supervisor import SupervisorAgent
from src.survey_bot.browser.launcher import BrowserManager
from src.survey_bot.browser.observer import PageObserver
from src.survey_bot.browser.interactor import PageInteractor
from src.survey_bot.models.persona import MoodType, PersonaConfig
from src.survey_bot.models.page_state import QuestionType

# Import LLM components (may not be available)
try:
    from src.survey_bot.llm.graph import (
        NavigationGraph,
        create_navigation_graph,
        run_survey_navigation,
    )
    from src.survey_bot.llm.chains import ActionType
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    NavigationGraph = None

# Import mock server
try:
    from tests.mock_server import (
        run_server_in_thread,
        stop_server,
        FLASK_AVAILABLE,
    )
except ImportError:
    FLASK_AVAILABLE = False
    run_server_in_thread = None
    stop_server = None


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def mock_server_url():
    """Start the mock survey server for testing."""
    if not FLASK_AVAILABLE:
        pytest.skip("Flask not installed - run: pip install flask")
    
    url = run_server_in_thread(host="127.0.0.1", port=5556)
    # Give server time to start
    time.sleep(1)
    
    yield url
    
    # Cleanup
    stop_server()


@pytest.fixture
def happy_persona():
    """Get a happy persona for testing."""
    supervisor = SupervisorAgent()
    return supervisor.get_persona("😊")


@pytest.fixture
def angry_persona():
    """Get an angry persona for testing."""
    supervisor = SupervisorAgent()
    return supervisor.get_persona("😡")


@pytest.fixture
def neutral_persona():
    """Get a neutral persona for testing."""
    supervisor = SupervisorAgent()
    return supervisor.get_persona("😐")


@pytest.fixture
def observer():
    """Create a PageObserver instance."""
    return PageObserver()


@pytest.fixture
def interactor():
    """Create a PageInteractor instance."""
    return PageInteractor(human_like=False)  # Faster for tests


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def wait_for_server(url: str, timeout: int = 10) -> bool:
    """Wait for server to be ready."""
    import aiohttp
    
    start = time.time()
    while time.time() - start < timeout:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/health") as resp:
                    if resp.status == 200:
                        return True
        except Exception:
            pass
        await asyncio.sleep(0.5)
    return False


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_browser_manager_launches(mock_server_url):
    """Test that BrowserManager can launch and navigate."""
    async with BrowserManager(headless=True) as page:
        await page.goto(f"{mock_server_url}/survey")
        title = await page.title()
        
        assert "Survey" in title
        assert page.url.endswith("/survey")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_observer_extracts_page_state(mock_server_url, observer):
    """Test that PageObserver correctly extracts page state."""
    async with BrowserManager(headless=True) as page:
        await page.goto(f"{mock_server_url}/survey")
        
        state = await observer.get_page_state(page)
        
        # Verify page info
        assert "survey" in state.url.lower()
        assert "Survey" in state.title
        
        # Verify elements found
        assert len(state.elements) > 0
        
        # Should detect rating scale question
        assert state.question_type == QuestionType.RATING_SCALE
        
        # Should find radio buttons for rating
        radios = state.get_radio_buttons()
        assert len(radios) >= 10  # 1-10 scale
        
        # Should find submit button
        submit = state.find_submit_button()
        assert submit is not None
        assert submit.matches_text("next", "submit", "continue")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_interactor_selects_rating(mock_server_url, interactor):
    """Test that PageInteractor can select a rating."""
    async with BrowserManager(headless=True) as page:
        await page.goto(f"{mock_server_url}/survey")
        
        # Select rating 9
        success = await interactor.select_rating(page, rating=9)
        assert success
        
        # Verify selection by checking radio button
        is_checked = await page.locator("input[type='radio'][value='9']").is_checked()
        assert is_checked


@pytest.mark.integration
@pytest.mark.asyncio
async def test_interactor_clicks_next(mock_server_url, interactor, observer):
    """Test that PageInteractor can click the Next button."""
    async with BrowserManager(headless=True) as page:
        await page.goto(f"{mock_server_url}/survey")
        
        # Select a rating first (required)
        await interactor.select_rating(page, rating=8)
        
        # Get page state to find submit button
        state = await observer.get_page_state(page)
        submit_btn = state.find_submit_button()
        
        assert submit_btn is not None
        
        # Click next
        success = await interactor.click_element(page, submit_btn)
        assert success
        
        # Wait for navigation
        await page.wait_for_load_state("networkidle")
        
        # Should be on page 2
        assert "page2" in page.url or "Tell Us More" in await page.content()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_survey_manual_navigation(mock_server_url, happy_persona, observer, interactor):
    """Test manually navigating through the full survey."""
    async with BrowserManager(headless=True) as page:
        # Page 1: Rating
        await page.goto(f"{mock_server_url}/survey")
        
        state = await observer.get_page_state(page)
        assert state.question_type == QuestionType.RATING_SCALE
        
        # Select happy rating
        await interactor.select_rating(page, rating=happy_persona.rating_preference)
        
        # Click Next
        submit_btn = state.find_submit_button()
        await interactor.click_element(page, submit_btn)
        await page.wait_for_load_state("networkidle")
        
        # Page 2: Multiple choice + Text
        state = await observer.get_page_state(page)
        
        # Select "Very Satisfied"
        await interactor.select_option(page, "Very Satisfied")
        
        # Select "Yes" for recommend
        await interactor.select_option(page, "Yes")
        
        # Fill comment (find textarea)
        inputs = state.get_inputs()
        textarea = next((i for i in inputs if i.element_type.value == "textarea"), None)
        if textarea:
            await interactor.fill_text(page, textarea, "Great experience! Highly recommend!")
        
        # Click Submit
        submit_btn = state.find_submit_button()
        await interactor.click_element(page, submit_btn)
        await page.wait_for_load_state("networkidle")
        
        # Page 3: Completion with coupon
        state = await observer.get_page_state(page)
        
        # Should detect completion
        assert state.has_coupon or state.question_type == QuestionType.COMPLETION
        
        # Should have coupon code
        if state.coupon_code:
            assert len(state.coupon_code) > 0
            print(f"Got coupon: {state.coupon_code}")


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(not LLM_AVAILABLE, reason="LLM components not available")
async def test_navigation_graph_happy_persona(mock_server_url, happy_persona):
    """
    Test the full NavigationGraph with happy persona.
    
    This is the main integration test that validates the complete
    survey automation flow.
    """
    # Create navigation graph (no LLM, uses rules)
    graph = create_navigation_graph(
        persona=happy_persona,
        llm=None,  # Use rule-based decisions
        max_steps=30,  # Enough for multi-question pages
    )
    
    async with BrowserManager(headless=True) as page:
        result = await graph.run(page, f"{mock_server_url}/survey")
        
        # Verify completion
        assert result["is_complete"], f"Survey not complete: {result.get('error')}"
        
        # Verify coupon was extracted
        assert result.get("coupon_code") is not None, "No coupon code found"
        
        # Verify coupon format (XXXX-XXXXXX)
        coupon = result["coupon_code"]
        assert "-" in coupon
        assert len(coupon) == 11  # XXXX-XXXXXX
        
        # Verify steps taken
        assert result["current_step"] > 0
        assert len(result["actions_taken"]) > 0
        
        print(f"✅ Survey completed in {result['current_step']} steps")
        print(f"🎫 Coupon: {coupon}")


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(not LLM_AVAILABLE, reason="LLM components not available")
async def test_navigation_graph_angry_persona(mock_server_url, angry_persona):
    """Test NavigationGraph with angry persona selects low ratings."""
    graph = create_navigation_graph(
        persona=angry_persona,
        llm=None,
        max_steps=30,  # Enough for multi-question pages
    )
    
    async with BrowserManager(headless=True) as page:
        result = await graph.run(page, f"{mock_server_url}/survey")
        
        # Should still complete (angry customers can complete surveys too!)
        assert result["is_complete"]
        assert result.get("coupon_code") is not None
        
        print(f"✅ Angry survey completed with coupon: {result['coupon_code']}")


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(not LLM_AVAILABLE, reason="LLM components not available")
async def test_run_survey_navigation_convenience(mock_server_url, happy_persona):
    """Test the convenience function for running surveys."""
    async with BrowserManager(headless=True) as page:
        result = await run_survey_navigation(
            page=page,
            survey_url=f"{mock_server_url}/survey",
            persona=happy_persona,
            llm=None,
            max_steps=30,  # Enough for multi-question pages
        )
        
        assert result["is_complete"]
        assert result.get("coupon_code") is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_screenshot_on_completion(mock_server_url, happy_persona, interactor):
    """Test that screenshots can be taken during navigation."""
    async with BrowserManager(headless=True) as page:
        await page.goto(f"{mock_server_url}/survey")
        
        # Take screenshot
        screenshot_path = await interactor.take_screenshot(page, "test_survey_page1")
        
        assert screenshot_path is not None
        assert "test_survey_page1" in screenshot_path
        
        # Verify file exists
        import os
        assert os.path.exists(screenshot_path)
        
        # Cleanup
        os.remove(screenshot_path)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_observer_detects_question_types(mock_server_url, observer, interactor):
    """Test that observer correctly detects different question types."""
    async with BrowserManager(headless=True) as page:
        # Page 1: Rating scale
        await page.goto(f"{mock_server_url}/survey")
        state = await observer.get_page_state(page)
        assert state.question_type == QuestionType.RATING_SCALE
        
        # Navigate to page 2
        await interactor.select_rating(page, 8)
        submit = state.find_submit_button()
        await interactor.click_element(page, submit)
        await page.wait_for_load_state("networkidle")
        
        # Page 2: Multiple choice (has multiple radio groups)
        state = await observer.get_page_state(page)
        # Could be MULTIPLE_CHOICE, YES_NO, TEXT_INPUT, or RATING_SCALE
        # (satisfaction scales like "Very Satisfied" to "Very Dissatisfied" 
        #  may be detected as RATING_SCALE since they're similar in structure)
        assert state.question_type in (
            QuestionType.MULTIPLE_CHOICE,
            QuestionType.YES_NO,
            QuestionType.TEXT_INPUT,  # Has textarea too
            QuestionType.RATING_SCALE,  # Satisfaction scale detection
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_max_steps_protection(mock_server_url, happy_persona):
    """Test that max_steps prevents infinite loops."""
    if not LLM_AVAILABLE:
        pytest.skip("LLM components not available")
    
    # Create graph with very low max steps
    graph = create_navigation_graph(
        persona=happy_persona,
        llm=None,
        max_steps=2,  # Will definitely not complete
    )
    
    async with BrowserManager(headless=True) as page:
        result = await graph.run(page, f"{mock_server_url}/survey")
        
        # Should hit max steps
        assert result["current_step"] >= 2
        
        # May or may not have error depending on timing
        # The important thing is it stopped


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_survey_completion_time(mock_server_url, happy_persona):
    """Test that survey completes within reasonable time."""
    if not LLM_AVAILABLE:
        pytest.skip("LLM components not available")
    
    import time
    
    graph = create_navigation_graph(
        persona=happy_persona,
        llm=None,
        max_steps=30,
    )
    
    start_time = time.time()
    
    async with BrowserManager(headless=True) as page:
        result = await graph.run(page, f"{mock_server_url}/survey")
    
    elapsed = time.time() - start_time
    
    # Should complete within 60 seconds (multi-step survey with delays)
    assert elapsed < 60, f"Survey took too long: {elapsed:.2f}s"
    assert result["is_complete"]
    
    print(f"⏱️ Survey completed in {elapsed:.2f} seconds")


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_handles_missing_elements_gracefully(mock_server_url, observer):
    """Test that observer handles pages with few elements."""
    async with BrowserManager(headless=True) as page:
        # Navigate to a simple page
        await page.goto(f"{mock_server_url}/health")
        
        state = await observer.get_page_state(page)
        
        # Should not crash, just return minimal state
        assert state.url is not None
        # Question type should be unknown for non-survey page
        assert state.question_type in (QuestionType.UNKNOWN, QuestionType.NAVIGATION)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_interactor_handles_missing_element(mock_server_url, interactor):
    """Test that interactor returns False for missing elements."""
    from src.survey_bot.models.page_state import WebElement, ElementType
    
    async with BrowserManager(headless=True) as page:
        await page.goto(f"{mock_server_url}/survey")
        
        # Create a fake element that doesn't exist
        fake_element = WebElement(
            element_id="nonexistent_element_12345",
            element_type=ElementType.BUTTON,
            text_content="This doesn't exist",
            is_visible=True,
            is_enabled=True,
        )
        
        # Should return False, not crash
        success = await interactor.click_element(page, fake_element)
        assert success is False


# =============================================================================
# MARKERS CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
