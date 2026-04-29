"""
Manuscript OCR — macOS launcher.
Starts the uvicorn server and opens the browser.
Shows a menu bar status while running.
"""

import os
import sys
import time
import signal
import threading
import webbrowser
import subprocess
from pathlib import Path

# When bundled by PyInstaller, sys._MEIPASS points to the temp extraction dir
if getattr(sys, 'frozen', False):
    APP_DIR = Path(sys._MEIPASS)
    # Also set the working directory so app.py can find its modules
    os.chdir(APP_DIR)
else:
    APP_DIR = Path(__file__).resolve().parent

HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"


def wait_for_server(timeout=30):
    """Poll until the server responds or timeout."""
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(URL, timeout=2)
            return True
        except Exception:
            time.sleep(0.3)
    return False


def run_server():
    """Start uvicorn serving the app."""
    import uvicorn
    # Suppress uvicorn's default logging to keep the console clean
    uvicorn.run(
        "app:app",
        host=HOST,
        port=PORT,
        log_level="warning",
    )


def main():
    # Start server in a background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Wait for it to come up
    print("Manuscript OCR — starting server...")
    if wait_for_server():
        print(f"Server ready at {URL}")
        webbrowser.open(URL)
    else:
        print("ERROR: Server failed to start within 30 seconds.")
        sys.exit(1)

    # Keep the main thread alive until Ctrl+C or window close
    try:
        print("Press Ctrl+C to quit.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)


if __name__ == "__main__":
    main()
