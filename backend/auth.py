"""Login flow: verify credentials, open a session row, mint a JWT.

db_login() reads control.users joined to control.channel and returns the
full profile (channel name, RAG channel id, feature list). login_user()
combines that with session creation and JWT minting; end_session() flips
the row's ended_at on logout.
"""
import uuid

from sqlalchemy import text

from db import engine
from jwt_auth import JWT_EXPIRE_HOURS, create_token


def db_login(username: str, password: str):
    """Return the user profile dict on credential match, or None."""
    query = text("""
        SELECT
            u.userid,
            u.name,
            u.username,
            u.channelid,
            u.authorized_features,
            u.authorized_data,
            c.name        AS channel_name,
            c.rag_channel_id
        FROM control.users u
        JOIN control.channel c ON u.channelid = c.channelid
        WHERE u.username = :username
          AND u.password = :password
        LIMIT 1;
    """)

    with engine.connect() as conn:
        row = conn.execute(query, {"username": username, "password": password}) \
                  .mappings().fetchone()

    if row is None:
        return None

    user = dict(row)
    return {
        "userid":              user["userid"],
        "name":                user["name"],
        "username":            user["username"],
        "channelid":           user["channelid"],
        "channel_name":        user["channel_name"],
        "rag_channel_id":      str(user["rag_channel_id"]) if user["rag_channel_id"] else None,
        "authorized_features": user["authorized_features"] or [],
        "authorized_data":     user["authorized_data"] or {},
    }


def _create_session(user_id: int, ip_address: str) -> str:
    """Open a session row and return its UUID. Expiry matches the JWT lifetime."""
    from datetime import datetime, timedelta, timezone
    session_uuid = uuid.uuid4()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO control.sessions
                    (session_id, user_id, ip_address, expires_at)
                VALUES
                    (:session_id, :user_id, :ip_address, :expires_at)
            """),
            {
                "session_id": session_uuid,
                "user_id":    user_id,
                "ip_address": ip_address,
                "expires_at": expires_at,
            },
        )
        conn.commit()
    return str(session_uuid)


def end_session(session_id: str) -> None:
    """Mark a session ended. Audit-only; JWT validity is unaffected."""
    with engine.connect() as conn:
        conn.execute(
            text("""
                UPDATE control.sessions
                SET ended_at = now()
                WHERE session_id = :sid
            """),
            {"sid": uuid.UUID(session_id)},
        )
        conn.commit()


def login_user(username: str, password: str, ip_address: str = "unknown"):
    """End-to-end login: verify credentials, open a session, return a token."""
    user = db_login(username, password)

    if user is None:
        return {
            "success": False,
            "message": "Invalid username or password.",
            "token":   None,
        }

    session_id = _create_session(user["userid"], ip_address)
    token = create_token(user, session_id)

    return {
        "success":             True,
        "message":             f"Welcome {user['name']}",
        "token":               token,
        "name":                user["name"],
        "channel_name":        user["channel_name"],
        "authorized_features": user["authorized_features"],
    }
