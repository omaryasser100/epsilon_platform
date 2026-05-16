"""JWT bearer-token plumbing.

create_token() encodes user identity + session id into an HS256-signed JWT.
get_current_user() is the FastAPI dependency that decodes the Authorization
header and returns the decoded claim dict for downstream handlers.
"""
import os
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

JWT_SECRET = os.getenv("JWT_SECRET", "changeme-set-in-env")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "8"))

security = HTTPBearer()


def create_token(claims: dict, session_id: str) -> str:
    """Sign and return a JWT carrying the user profile + session id."""
    payload = {**claims}
    payload["session_id"] = session_id
    payload["sub"] = str(claims.get("userid", ""))
    payload["iat"] = datetime.now(timezone.utc)
    payload["exp"] = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """FastAPI dependency: decode the bearer token or raise 401."""
    try:
        return decode_token(credentials.credentials)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")
