#!/usr/bin/env python
"""Run Survey Bot with detailed logging to console and file."""
import os
import sys
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f'survey_bot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("=" * 70)
    logger.info("Starting Survey Bot Web UI")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 70)

    try:
        from survey_bot.web.app import app

        print("\n" + "=" * 70)
        print("           SURVEY BOT WEB UI")
        print("=" * 70)
        print(f"  URL: http://localhost:5000")
        print(f"  Logs: {log_file}")
        print("  Press Ctrl+C to stop")
        print("=" * 70 + "\n")

        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error starting server: {e}", exc_info=True)
        sys.exit(1)
