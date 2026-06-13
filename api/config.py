"""Shared paths/config for the FastAPI backend.

Importing this module guarantees the repo root is on sys.path so that
`from src.ml.predict import EnergySavingsPredictor` resolves when the app is
launched as `uvicorn api.main:app` from the repo root.
"""
import sys
from pathlib import Path

# api/config.py -> parents[1] is the repo root (where models/, *_artifacts/ live)
REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
