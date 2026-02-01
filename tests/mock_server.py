"""
Mock Survey Server for Integration Testing.

This Flask application simulates a 3-page customer satisfaction survey
for testing the Navigation Agent without needing real survey sites.

Survey Flow:
1. Page 1: Rating question (1-10 scale)
2. Page 2: Multiple choice + Text feedback
3. Page 3: Completion page with coupon code

Usage:
    # Run the server
    python -m tests.mock_server
    
    # Or from tests
    from tests.mock_server import create_app, run_server_in_thread

Requirements:
    pip install flask
"""
from __future__ import annotations

import random
import string
import threading
import time
from typing import Any, Optional

try:
    from flask import Flask, render_template_string, request, redirect, url_for, session
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None


__all__ = [
    "create_app",
    "run_server_in_thread",
    "stop_server",
    "FLASK_AVAILABLE",
]


# =============================================================================
# HTML TEMPLATES
# =============================================================================

BASE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 600px;
            margin: 50px auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .survey-container {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 { color: #333; font-size: 24px; }
        h2 { color: #555; font-size: 18px; margin-top: 20px; }
        .question { margin: 20px 0; }
        .question label { display: block; margin: 8px 0; cursor: pointer; }
        .rating-scale {
            display: flex;
            gap: 10px;
            margin: 15px 0;
        }
        .rating-scale label {
            display: flex;
            flex-direction: column;
            align-items: center;
            cursor: pointer;
        }
        .rating-scale input[type="radio"] {
            margin-bottom: 5px;
        }
        textarea {
            width: 100%;
            height: 100px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-family: inherit;
        }
        .btn {
            background: #007bff;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 20px;
        }
        .btn:hover { background: #0056b3; }
        .coupon-code {
            background: #28a745;
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 8px;
            margin: 20px 0;
        }
        .coupon-code .code {
            font-size: 32px;
            font-weight: bold;
            letter-spacing: 3px;
            margin: 10px 0;
        }
        .progress {
            background: #e9ecef;
            height: 8px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        .progress-bar {
            background: #007bff;
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s;
        }
        .error { color: red; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="survey-container">
        {{ content | safe }}
    </div>
</body>
</html>
"""

PAGE1_CONTENT = """
<div class="progress"><div class="progress-bar" style="width: 33%"></div></div>
<h1>Customer Satisfaction Survey</h1>
<p>Thank you for visiting our store! Please take a moment to share your experience.</p>

<form method="POST" action="/survey/page1">
    <div class="question">
        <h2>How would you rate your overall experience today?</h2>
        <div class="rating-scale">
            {% for i in range(1, 11) %}
            <label>
                <input type="radio" name="rating" value="{{ i }}" required>
                {{ i }}
            </label>
            {% endfor %}
        </div>
    </div>
    
    <button type="submit" class="btn">Next</button>
</form>
"""

PAGE2_CONTENT = """
<div class="progress"><div class="progress-bar" style="width: 66%"></div></div>
<h1>Tell Us More</h1>

<form method="POST" action="/survey/page2">
    <div class="question">
        <h2>How satisfied were you with the staff?</h2>
        <label><input type="radio" name="staff_satisfaction" value="very_satisfied" required> Very Satisfied</label>
        <label><input type="radio" name="staff_satisfaction" value="satisfied"> Satisfied</label>
        <label><input type="radio" name="staff_satisfaction" value="neutral"> Neutral</label>
        <label><input type="radio" name="staff_satisfaction" value="dissatisfied"> Dissatisfied</label>
        <label><input type="radio" name="staff_satisfaction" value="very_dissatisfied"> Very Dissatisfied</label>
    </div>
    
    <div class="question">
        <h2>Would you recommend us to a friend?</h2>
        <label><input type="radio" name="recommend" value="yes" required> Yes</label>
        <label><input type="radio" name="recommend" value="no"> No</label>
    </div>
    
    <div class="question">
        <h2>Please share any additional comments:</h2>
        <textarea name="comments" placeholder="Tell us about your experience..."></textarea>
    </div>
    
    <button type="submit" class="btn">Submit</button>
</form>
"""

PAGE3_CONTENT = """
<div class="progress"><div class="progress-bar" style="width: 100%"></div></div>
<h1>Thank You!</h1>

<p>We appreciate your feedback. Your responses help us improve our service.</p>

<div class="coupon-code">
    <p>Your Validation Code:</p>
    <div class="code">{{ coupon_code }}</div>
    <p>Show this code on your next visit for a special offer!</p>
</div>

<p>This code expires in 30 days.</p>
"""

ERROR_CONTENT = """
<h1>Error</h1>
<p class="error">{{ message }}</p>
<a href="/survey" class="btn">Start Over</a>
"""


# =============================================================================
# FLASK APPLICATION
# =============================================================================

def generate_coupon_code() -> str:
    """Generate a random coupon code."""
    letters = ''.join(random.choices(string.ascii_uppercase, k=4))
    numbers = ''.join(random.choices(string.digits, k=6))
    return f"{letters}-{numbers}"


def create_app(debug: bool = False) -> Flask:
    """
    Create and configure the Flask application.
    
    Args:
        debug: Enable debug mode.
        
    Returns:
        Configured Flask app.
    """
    if not FLASK_AVAILABLE:
        raise ImportError("Flask not installed. Run: pip install flask")
    
    app = Flask(__name__)
    app.secret_key = 'test-secret-key-for-sessions'
    app.config['DEBUG'] = debug
    
    @app.route('/')
    def index():
        """Redirect to survey start."""
        return redirect(url_for('survey_start'))
    
    @app.route('/survey')
    def survey_start():
        """Start the survey - Page 1."""
        session.clear()
        session['step'] = 1
        content = render_template_string(PAGE1_CONTENT)
        return render_template_string(BASE_TEMPLATE, title="Survey - Page 1", content=content)
    
    @app.route('/survey/page1', methods=['POST'])
    def survey_page1_submit():
        """Handle Page 1 submission."""
        rating = request.form.get('rating')
        
        if not rating:
            content = render_template_string(ERROR_CONTENT, message="Please select a rating.")
            return render_template_string(BASE_TEMPLATE, title="Error", content=content)
        
        session['rating'] = rating
        session['step'] = 2
        
        content = render_template_string(PAGE2_CONTENT)
        return render_template_string(BASE_TEMPLATE, title="Survey - Page 2", content=content)
    
    @app.route('/survey/page2', methods=['POST'])
    def survey_page2_submit():
        """Handle Page 2 submission - Show completion."""
        staff_satisfaction = request.form.get('staff_satisfaction')
        recommend = request.form.get('recommend')
        comments = request.form.get('comments', '')
        
        if not staff_satisfaction or not recommend:
            content = render_template_string(ERROR_CONTENT, message="Please answer all required questions.")
            return render_template_string(BASE_TEMPLATE, title="Error", content=content)
        
        # Store responses
        session['staff_satisfaction'] = staff_satisfaction
        session['recommend'] = recommend
        session['comments'] = comments
        session['step'] = 3
        
        # Generate coupon code
        coupon_code = generate_coupon_code()
        session['coupon_code'] = coupon_code
        
        content = render_template_string(PAGE3_CONTENT, coupon_code=coupon_code)
        return render_template_string(BASE_TEMPLATE, title="Survey Complete", content=content)
    
    @app.route('/health')
    def health_check():
        """Health check endpoint."""
        return {'status': 'ok'}
    
    @app.route('/api/responses')
    def get_responses():
        """API endpoint to get current session responses (for testing)."""
        return {
            'step': session.get('step'),
            'rating': session.get('rating'),
            'staff_satisfaction': session.get('staff_satisfaction'),
            'recommend': session.get('recommend'),
            'comments': session.get('comments'),
            'coupon_code': session.get('coupon_code'),
        }
    
    return app


# =============================================================================
# SERVER MANAGEMENT
# =============================================================================

_server_thread: Optional[threading.Thread] = None
_server_running: bool = False


def run_server_in_thread(
    host: str = "127.0.0.1",
    port: int = 5555,
    debug: bool = False,
) -> str:
    """
    Run the mock server in a background thread.
    
    Args:
        host: Host to bind to.
        port: Port to bind to.
        debug: Enable debug mode.
        
    Returns:
        Base URL of the server.
    """
    global _server_thread, _server_running
    
    if _server_running:
        return f"http://{host}:{port}"
    
    app = create_app(debug=debug)
    
    def run():
        global _server_running
        _server_running = True
        # Use werkzeug directly for cleaner shutdown
        from werkzeug.serving import make_server
        server = make_server(host, port, app, threaded=True)
        server.serve_forever()
    
    _server_thread = threading.Thread(target=run, daemon=True)
    _server_thread.start()
    
    # Wait for server to start
    time.sleep(1)
    
    return f"http://{host}:{port}"


def stop_server():
    """Stop the mock server."""
    global _server_running
    _server_running = False
    # Thread is daemon, so it will stop when main program exits


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if not FLASK_AVAILABLE:
        print("Flask not installed. Run: pip install flask")
        exit(1)
    
    app = create_app(debug=True)
    print("Starting mock survey server on http://127.0.0.1:5555")
    print("Survey URL: http://127.0.0.1:5555/survey")
    print("Press Ctrl+C to stop")
    app.run(host="127.0.0.1", port=5555, debug=True)
