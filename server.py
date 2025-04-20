#!/usr/bin/env python3
"""
Unified server script for DSAN Facial Expression Recognition

This script serves both the static landing page and the Flask application.
It automatically redirects users to the right interface based on the URL.
"""
import os
import sys
import logging
import subprocess
from threading import Thread
import time
from flask import Flask, send_from_directory, redirect, request

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask
app = Flask(__name__)

# Create a variable to track the Streamlit process
streamlit_process = None

@app.route('/')
def index():
    """Serve the landing page"""
    return send_from_directory('.', 'index.html')

@app.route('/flask')
def flask_app():
    """Redirect to the Flask application"""
    # Since this is the Flask app itself, just redirect to the main app
    return redirect('/app')

@app.route('/app')
def main_app():
    """Import and run the main Flask application"""
    try:
        from main import app as main_app
        return main_app.index()
    except Exception as e:
        logger.error(f"Error importing main Flask app: {e}")
        return f"Error: {str(e)}"

@app.route('/streamlit')
def streamlit_app():
    """Start the Streamlit application if it's not running and redirect to it"""
    global streamlit_process
    
    # Check if Streamlit is already running
    if streamlit_process is None or streamlit_process.poll() is not None:
        # Start Streamlit in a separate process
        try:
            logger.info("Starting Streamlit application...")
            streamlit_process = subprocess.Popen(
                ["python", "start_streamlit.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Give it a moment to start
            time.sleep(2)
            
            if streamlit_process.poll() is not None:
                # Process has already exited
                stdout, stderr = streamlit_process.communicate()
                logger.error(f"Streamlit failed to start: {stderr}")
                return "Failed to start Streamlit application. Check logs for details."
            
            logger.info("Streamlit application started successfully")
        except Exception as e:
            logger.error(f"Error starting Streamlit: {e}")
            return f"Error starting Streamlit: {str(e)}"
    
    # Redirect to the Streamlit application
    return redirect("http://localhost:8501")

@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

def start_streamlit():
    """Start the Streamlit server in a separate thread"""
    global streamlit_process
    
    logger.info("Pre-starting Streamlit application...")
    try:
        streamlit_process = subprocess.Popen(
            ["python", "start_streamlit.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info("Streamlit pre-start initiated")
    except Exception as e:
        logger.error(f"Error pre-starting Streamlit: {e}")

if __name__ == "__main__":
    # Start Streamlit in a separate thread
    streamlit_thread = Thread(target=start_streamlit)
    streamlit_thread.daemon = True
    streamlit_thread.start()
    
    # Start the Flask server
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)