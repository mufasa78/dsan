#!/bin/bash

# DSAN Facial Expression Recognition Application Runner

print_header() {
  echo "========================================================"
  echo "  Dual Stream Attention Network (DSAN)"
  echo "  Facial Expression Recognition"
  echo "========================================================"
  echo
}

print_usage() {
  echo "Usage: ./run.sh [server|flask|streamlit]"
  echo
  echo "Options:"
  echo "  server     Run the unified server with both interfaces (default)"
  echo "  flask      Run only the Flask web application"
  echo "  streamlit  Run only the Streamlit interactive application"
  echo
  echo "Examples:"
  echo "  ./run.sh           # Run unified server by default"
  echo "  ./run.sh server    # Run unified server explicitly"
  echo "  ./run.sh flask     # Run Flask app only"
  echo "  ./run.sh streamlit # Run Streamlit app only"
  echo
}

run_server() {
  echo "Starting Unified Server (Flask + Streamlit)..."
  echo "Main URL: http://localhost:5000"
  echo "Flask Interface: http://localhost:5000/flask"
  echo "Streamlit Interface: http://localhost:5000/streamlit"
  echo "Press Ctrl+C to stop the server"
  echo
  python server.py
}

run_flask() {
  echo "Starting Flask Web Application..."
  echo "URL: http://localhost:5000"
  echo "Press Ctrl+C to stop the server"
  echo
  gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app
}

run_streamlit() {
  echo "Starting Streamlit Interactive Application..."
  echo "URL: http://localhost:8501"
  echo "Press Ctrl+C to stop the server"
  echo
  ./start_streamlit.py
}

# Main execution
print_header

# Parse arguments
if [ $# -eq 0 ]; then
  # No arguments, default to unified server
  run_server
elif [ "$1" = "server" ]; then
  run_server
elif [ "$1" = "flask" ]; then
  run_flask
elif [ "$1" = "streamlit" ]; then
  run_streamlit
else
  echo "Error: Unknown option '$1'"
  print_usage
  exit 1
fi