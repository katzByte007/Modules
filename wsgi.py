"""
PythonAnywhere WSGI entry — kartik2025 / Modules repo.

Paste this into Web → WSGI configuration file, or import from here.
"""
import sys
import os

# ← Your PA username + clone folder (adjust if different)
project_home = '/home/kartik2025/Modules'

if project_home not in sys.path:
    sys.path.insert(0, project_home)

os.chdir(project_home)
os.environ.setdefault('VISION_LITE', '1')

# app.py contains: app = Flask(__name__)
from app import app

application = app
