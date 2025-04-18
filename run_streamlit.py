#!/usr/bin/env python
"""
Script to run the Streamlit application for DSAN Facial Expression Recognition
"""
import os
import subprocess
import sys

def main():
    # Print header
    print("=" * 70)
    print("  Dual Stream Attention Network (DSAN)")
    print("  Facial Expression Recognition - Streamlit Interface")
    print("=" * 70)
    print()
    
    # Set environment variables for Streamlit
    os.environ["STREAMLIT_SERVER_PORT"] = "8501"
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"
    os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "false"
    
    print("Starting Streamlit application...")
    print("URL: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    print()
    
    try:
        # Run the Streamlit app
        subprocess.run([
            "streamlit", "run", "streamlit_app.py",
            "--server.headless=true",
            "--server.enableCORS=false",
            "--server.enableXsrfProtection=false"
        ], check=True)
    except KeyboardInterrupt:
        print("\nStreamlit server stopped.")
    except Exception as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()