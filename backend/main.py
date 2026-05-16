"""Platform backend — FastAPI app the Gradio frontend talks to.

Public surface:
  POST /login        — username/password → JWT bearer token + user profile
  POST /logout       — marks the session ended (token still valid until expiry)
  POST /orchestrate  — routes user actions (query / ingest) to rag_service
  POST /upload       — accepts a PDF and stores it on the shared uploads volume
  GET  /health       — liveness probe

/admin/* routes live in admin.py and require the `admin_panel` feature.
"""
import os

from fastapi import Depends, FastAPI, Request, UploadFile

from admin import router as admin_router
from auth import end_session, login_user
from jwt_auth import get_current_user
from orchestrator import orchestrator
from schemas import LoginRequest, OrchestratorRequest

app = FastAPI(title="Epsilon Platform Backend")
app.include_router(admin_router)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/uploads")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/login")
def login(request_body: LoginRequest, request: Request):
    ip = request.client.host if request.client else "unknown"
    return login_user(request_body.username, request_body.password, ip)


@app.post("/logout")
def logout(user: dict = Depends(get_current_user)):
    session_id = user.get("session_id")
    if session_id:
        end_session(session_id)
    return {"success": True, "message": "Logged out successfully."}


@app.post("/orchestrate")
def orchestrate(request: OrchestratorRequest, user: dict = Depends(get_current_user)):
    return orchestrator(
        user_session=user,
        action=request.action,
        payload=request.payload,
    )


@app.post("/upload")
async def upload_file(file: UploadFile, user: dict = Depends(get_current_user)):
    """Persist an uploaded PDF to the shared volume so rag_service can read it
    by file path during ingestion."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, file.filename)
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)
    return {
        "success":   True,
        "file_path": dest,
        "filename":  file.filename,
    }
