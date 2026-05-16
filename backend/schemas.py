"""Request/response models for the public backend routes."""
from typing import Any, Optional

from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str
    password: str


class OrchestratorRequest(BaseModel):
    action: str
    payload: Optional[dict[str, Any]] = None
