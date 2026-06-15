"""PythonAnywhere WSGI entry — point your PA web app at this file."""
import os
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

os.environ.setdefault('VISION_LITE', '1')

from app import app as application  # noqa: F401
