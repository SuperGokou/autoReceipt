import argparse
import os
import sys
import webbrowser
from threading import Timer


def open_browser(port: int):
    """Open the browser after a short delay."""
    webbrowser.open(f'http://localhost:{port}')


def main():
    parser = argparse.ArgumentParser(description='Run Survey Bot Web UI')
    parser.add_argument('--port', type=int, default=5000, help='Port to run on (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-browser', action='store_true', help='Don\'t open browser automatically')
    args = parser.parse_args()
    
    # Set environment variables
    if args.debug:
        os.environ['FLASK_DEBUG'] = 'true'
    os.environ['PORT'] = str(args.port)
    
    # Open browser after 1.5 seconds
    if not args.no_browser:
        Timer(1.5, open_browser, [args.port]).start()
    
    # Import and run the app
    try:
        from src.survey_bot.web.app import app
        
        print(f"""
===============================================================
           SURVEY BOT WEB UI
===============================================================
  Open in browser: http://localhost:{args.port}
  Press Ctrl+C to stop
===============================================================
        """)
        
        app.run(host='0.0.0.0', port=args.port, debug=args.debug, threaded=True)
        
    except ImportError as e:
        print(f"Error: {e}")
        print("\nMake sure you're running from the project root directory:")
        print("  cd J:\\Project Files\\MyPython\\autoReceipt")
        print("  python run_web.py")
        sys.exit(1)


if __name__ == '__main__':
    main()
