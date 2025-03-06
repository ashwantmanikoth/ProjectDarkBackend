# filepath: /Users/ashwantmanikoth/Desktop/programming/ProjectDark/Backend/src/utils/error_handling.py
from flask import jsonify

def handle_exception(e):
    print(f"An error occurred: {e}")
    return jsonify({"error": "An internal error occurred. Please try again later."}), 500