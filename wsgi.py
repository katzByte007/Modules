"""
PythonAnywhere WSGI entry for a manual Flask web app.

In PA: Web → your app → WSGI configuration file
Point it at this file, or paste the same imports into the default wsgi file.
"""
import os
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

os.environ.setdefault('VISION_LITE', '1')

# app.py defines: app = Flask(__name__)
from app import app

# uWSGI on PythonAnywhere also accepts this alias:
application = app
