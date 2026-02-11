"""
Survey Bot Web Application.

A Flask-based web UI for the Survey Bot automation system.
Provides a modern interface for uploading receipts, selecting mood,
and receiving coupons via email.

Run with:
    python -m survey_bot.web.app
    
Or with Flask:
    cd src/survey_bot/web && flask run --port 5000
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from threading import Thread

from flask import Flask, render_template, request, jsonify, Response, stream_with_context

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY') or os.urandom(32).hex()

# Store for active jobs and their progress
jobs: dict[str, dict] = {}


def get_bot():
    """Lazy import of SurveyBot to avoid circular imports."""
    from ..main import SurveyBot
    return SurveyBot


# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
    })


@app.route('/api/submit', methods=['POST'])
def submit_survey():
    """
    Submit a survey automation job.
    
    Accepts either:
    - Form data with 'receipt' file upload
    - JSON with 'survey_url' for direct URL mode
    
    Returns a job_id for tracking progress.
    """
    try:
        job_id = str(uuid.uuid4())[:8]
        
        # Initialize job status
        jobs[job_id] = {
            'status': 'pending',
            'progress': [],
            'result': None,
            'error': None,
            'created_at': datetime.now().isoformat(),
        }
        
        # Check if it's a direct URL submission
        if request.is_json:
            data = request.get_json()
            survey_url = data.get('survey_url')
            mood = data.get('mood', 'happy')
            email = data.get('email')
            # Allow environment override for debugging
            env_headless = os.environ.get('BROWSER_HEADLESS', '').lower()
            if env_headless in ('false', '0', 'no'):
                headless = False
            else:
                headless = data.get('headless', True)
            
            if not survey_url:
                return jsonify({'error': 'survey_url is required'}), 400
            if not email:
                return jsonify({'error': 'email is required'}), 400
            
            # Start async job in background thread
            thread = Thread(
                target=run_direct_url_job,
                args=(job_id, survey_url, mood, email, headless)
            )
            thread.daemon = True
            thread.start()
            
        else:
            # Form data with file upload
            if 'receipt' not in request.files:
                return jsonify({'error': 'No receipt file provided'}), 400
            
            receipt_file = request.files['receipt']
            if receipt_file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            mood = request.form.get('mood', 'happy')
            email = request.form.get('email')
            # Allow environment override for debugging
            env_headless = os.environ.get('BROWSER_HEADLESS', '').lower()
            if env_headless in ('false', '0', 'no'):
                headless = False
            else:
                headless = request.form.get('headless', 'true').lower() == 'true'
            
            if not email:
                return jsonify({'error': 'email is required'}), 400
            
            # Save uploaded file to temp location
            temp_dir = tempfile.mkdtemp()
            file_ext = Path(receipt_file.filename).suffix or '.jpg'
            temp_path = Path(temp_dir) / f"receipt_{job_id}{file_ext}"
            receipt_file.save(str(temp_path))
            
            # Start async job in background thread
            thread = Thread(
                target=run_receipt_job,
                args=(job_id, str(temp_path), mood, email, headless)
            )
            thread.daemon = True
            thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'started',
            'message': 'Survey automation job started',
        })
        
    except Exception as e:
        logger.exception("Error submitting job")
        return jsonify({'error': str(e)}), 500


@app.route('/api/status/<job_id>')
def job_status(job_id: str):
    """Get the status of a job."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    return jsonify({
        'job_id': job_id,
        'status': job['status'],
        'progress': job['progress'],
        'result': job['result'],
        'error': job['error'],
    })


@app.route('/api/stream/<job_id>')
def stream_progress(job_id: str):
    """Stream job progress via Server-Sent Events."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    def generate():
        last_progress_count = 0
        
        while True:
            job = jobs.get(job_id)
            if not job:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Job not found'})}\n\n"
                break
            
            # Send new progress updates
            current_progress = job['progress']
            if len(current_progress) > last_progress_count:
                for msg in current_progress[last_progress_count:]:
                    yield f"data: {json.dumps({'type': 'progress', 'message': msg})}\n\n"
                last_progress_count = len(current_progress)
            
            # Check if job is complete
            if job['status'] in ('completed', 'failed'):
                result_data = {
                    'type': 'complete',
                    'status': job['status'],
                    'result': job['result'],
                    'error': job['error'],
                }
                yield f"data: {json.dumps(result_data)}\n\n"
                break
            
            # Small delay to prevent busy-waiting
            time.sleep(0.5)
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


# Maximum number of finished jobs to keep in memory
MAX_FINISHED_JOBS = 100


# =============================================================================
# BACKGROUND JOB RUNNERS
# =============================================================================

def update_job_progress(job_id: str, message: str):
    """Add a progress message to a job."""
    if job_id in jobs:
        jobs[job_id]['progress'].append(message)
        logger.info(f"[{job_id}] {message}")


def _cleanup_old_jobs():
    """Remove oldest finished jobs when the limit is exceeded."""
    finished = [
        (jid, j) for jid, j in jobs.items()
        if j['status'] in ('completed', 'failed')
    ]
    if len(finished) > MAX_FINISHED_JOBS:
        finished.sort(key=lambda x: x[1].get('created_at', ''))
        for jid, _ in finished[:len(finished) - MAX_FINISHED_JOBS]:
            del jobs[jid]


def _finalize_job(job_id: str, result):
    """Process a SurveyResult and update the job dict."""
    if result.success:
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['result'] = {
            'success': True,
            'coupon_code': result.coupon.code if result.coupon else None,
            'email_sent': result.email_sent,
            'steps_taken': result.steps_taken,
            'duration': result.duration_seconds,
            'survey_url': result.survey_url,
        }
        update_job_progress(job_id, f"[SUCCESS] Coupon: {result.coupon.code if result.coupon else 'N/A'}")
    else:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = result.error_message
        update_job_progress(job_id, f"[FAILED] {result.error_message}")
    _cleanup_old_jobs()


def _run_job(job_id: str, coro, init_message: str, cleanup_path: Optional[str] = None):
    """Generic job runner for background threads."""
    try:
        jobs[job_id]['status'] = 'running'
        update_job_progress(job_id, init_message)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            SurveyBot = get_bot()
            bot = SurveyBot(
                verbose=True,
                headless=True,
                use_vision=True,
                progress_callback=lambda stage, msg: update_job_progress(job_id, f"{msg}"),
            )

            update_job_progress(job_id, "Starting survey automation...")
            result = loop.run_until_complete(coro(bot))
            _finalize_job(job_id, result)
        finally:
            loop.close()
            if cleanup_path:
                try:
                    Path(cleanup_path).unlink(missing_ok=True)
                    Path(cleanup_path).parent.rmdir()
                except Exception:
                    pass

    except Exception as e:
        logger.exception(f"Error in job {job_id}")
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)
        update_job_progress(job_id, f"[ERROR] {str(e)}")


def run_receipt_job(job_id: str, image_path: str, mood: str, email: str, headless: bool):
    """Run survey automation from receipt image in background thread."""
    _run_job(
        job_id,
        coro=lambda bot: bot.run(image_path=image_path, mood=mood, email=email),
        init_message="Processing receipt image...",
        cleanup_path=image_path,
    )


def run_direct_url_job(job_id: str, survey_url: str, mood: str, email: str, headless: bool):
    """Run survey automation with direct URL in background thread."""
    _run_job(
        job_id,
        coro=lambda bot: bot.run_with_url(survey_url=survey_url, mood=mood, email=email),
        init_message=f"Using direct URL: {survey_url[:50]}...",
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║           SURVEY BOT WEB UI                            ║
╠═══════════════════════════════════════════════════════════╣
║  Open in browser: http://localhost:{port}                  ║
║  API Health:      http://localhost:{port}/api/health       ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)
