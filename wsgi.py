"""
Vercel entrypoint – re-exports the Flask `app` object from backend/app.py.
Vercel auto-detects wsgi.py as a valid Flask entrypoint.
"""
import sys
import os

# Make backend/ importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

from app import app  # noqa: F401  (re-exported for Vercel)
