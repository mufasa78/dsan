#!/bin/bash

# Print header
echo "=========================================================================="
echo "  Dual Stream Attention Network (DSAN)"
echo "  Facial Expression Recognition - Streamlit Interface"
echo "=========================================================================="
echo

echo "Starting Streamlit application..."
echo "URL: http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo

# Run the Streamlit app
streamlit run streamlit_app.py --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false