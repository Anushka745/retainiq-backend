"""
Global mutable state for the FastAPI application.
Holds the loaded ML model so it isn't reloaded on every request.
"""
from sklearn.pipeline import Pipeline
from typing import Optional

MODEL: Optional[Pipeline] = None
MODEL_BACKEND: str = "none"
