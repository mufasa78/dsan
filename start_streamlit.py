#!/usr/bin/env python3
"""
Starter script for the Streamlit interface of DSAN Facial Expression Recognition
This script ensures proper environment setup before launching Streamlit
"""
import os
import subprocess
import sys

def main():
    """Main function to start the Streamlit app"""
    
    # Print header
    print("="*80)
    print("DSAN Facial Expression Recognition - Streamlit Interface")
    print("="*80)
    
    # Check if streamlit is installed
    try:
        subprocess.run(["streamlit", "--version"], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE, 
                      check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Streamlit not found. Installing...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"],
                          check=True)
        except subprocess.SubprocessError as e:
            print(f"Failed to install Streamlit: {e}")
            sys.exit(1)
    
    # Set Streamlit configuration
    os.environ["STREAMLIT_SERVER_PORT"] = "8501"
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"
    os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "false"
    
    print("\nStarting Streamlit application...")
    print("Access the application at: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    print("="*80)
    
    # Run the Streamlit app
    try:
        subprocess.run(["streamlit", "run", "streamlit_app.py", 
                       "--server.headless=true",
                       "--server.enableCORS=false",
                       "--server.enableXsrfProtection=false",
                       "--server.port=8501",
                       "--server.address=0.0.0.0"],
                      check=True)
    except KeyboardInterrupt:
        print("\nStreamlit server stopped")
    except subprocess.SubprocessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()