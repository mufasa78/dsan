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
  echo "Usage: ./run.sh [flask|streamlit]"
  echo
  echo "Options:"
  echo "  flask      Run the Flask web application (default)"
  echo "  streamlit  Run the Streamlit interactive application"
  echo
  echo "Examples:"
  echo "  ./run.sh           # Run Flask app by default"
  echo "  ./run.sh flask     # Run Flask app explicitly"
  echo "  ./run.sh streamlit # Run Streamlit app"
  echo
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
  streamlit run streamlit_app.py
}

# Main execution
print_header

# Parse arguments
if [ $# -eq 0 ]; then
  # No arguments, default to Flask
  run_flask
elif [ "$1" = "flask" ]; then
  run_flask
elif [ "$1" = "streamlit" ]; then
  run_streamlit
else
  echo "Error: Unknown option '$1'"
  print_usage
  exit 1
fi