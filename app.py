# filepath: /Users/ashwantmanikoth/Desktop/programming/ProjectDark/Backend/src/app.py
from flask import Flask
from flask_cors import CORS
from routes import configure_routes

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

configure_routes(app)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
