"""Gradio frontend — login, chat, and admin panel for the Epsilon platform.

The app has two screens (login and main). The main screen is a Tabs container
with one Chat tab visible to every user and three admin-only tabs (Channels,
Reports, Ingest PDF) shown when the user's JWT carries the `admin_panel`
feature. All backend calls go through helper functions at the top.

Launch modes:
  - Default: mount Gradio inside a FastAPI app that strips X-Frame-Options so
    Lightning AI's iframe wrapper (web-ui?port=7860) renders the UI.
  - GRADIO_SHARE=true: skip the wrapper and call app.launch(share=True) to get
    a public https://*.gradio.live tunnel URL instead.
"""
import os

import requests

# gradio_client 1.3.0 (shipped with gradio==4.44.1) crashes when walking a
# JSON schema that contains boolean values (Pydantic v2 emits
# `additionalProperties: true` for `dict[str, Any]`). The two functions below
# guard the offending code paths until we can move to a newer gradio.
import gradio_client.utils as _gc_utils

_orig_get_type = _gc_utils.get_type
_orig_jstpt = _gc_utils._json_schema_to_python_type


def _patched_get_type(schema):
    if not isinstance(schema, dict):
        return None
    return _orig_get_type(schema)


def _patched_jstpt(schema, defs=None):
    if isinstance(schema, bool):
        return "Any" if schema else "Never"
    return _orig_jstpt(schema, defs)


_gc_utils.get_type = _patched_get_type
_gc_utils._json_schema_to_python_type = _patched_jstpt

import gradio as gr  # noqa: E402  — must come after the patch above


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Authorized-feature options offered when creating or editing a channel.
FEATURE_OPTIONS = [
    "chatbot",
    "document_search",
    "tables_search",
    "voice_agent",
    "admin_panel",
]


# ── API helpers ───────────────────────────────────────────────────────────────

def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def api_login(username: str, password: str) -> dict:
    r = requests.post(
        f"{BACKEND_URL}/login",
        json={"username": username, "password": password},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def api_logout(token: str) -> None:
    """Best-effort logout. Token still expires naturally if the call fails."""
    try:
        requests.post(
            f"{BACKEND_URL}/logout",
            headers=_auth_headers(token),
            timeout=10,
        )
    except Exception:
        pass


def api_orchestrate(token: str, action: str, payload: dict | None = None) -> dict:
    r = requests.post(
        f"{BACKEND_URL}/orchestrate",
        json={"action": action, "payload": payload or {}},
        headers=_auth_headers(token),
        timeout=600,
    )
    r.raise_for_status()
    return r.json()


def api_upload(token: str, file_path: str, filename: str) -> dict:
    with open(file_path, "rb") as f:
        r = requests.post(
            f"{BACKEND_URL}/upload",
            files={"file": (filename, f, "application/pdf")},
            headers=_auth_headers(token),
            timeout=120,
        )
    r.raise_for_status()
    return r.json()


def api_admin_channels(token: str) -> list[dict]:
    r = requests.get(
        f"{BACKEND_URL}/admin/channels",
        headers=_auth_headers(token),
        timeout=10,
    )
    r.raise_for_status()
    return r.json().get("channels", [])


def api_admin_create_channel(token: str, name: str, description: str, features: list[str]) -> dict:
    r = requests.post(
        f"{BACKEND_URL}/admin/channels",
        json={
            "name": name,
            "description": description,
            "authorized_features": features,
        },
        headers=_auth_headers(token),
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def api_admin_create_rag_channel(token: str, channelid: int, name: str, description: str) -> dict:
    r = requests.post(
        f"{BACKEND_URL}/admin/channels/{channelid}/rag-channel",
        json={"name": name, "description": description},
        headers=_auth_headers(token),
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def api_admin_list_reports(token: str, channelid: int) -> list[dict]:
    r = requests.get(
        f"{BACKEND_URL}/admin/channels/{channelid}/reports",
        headers=_auth_headers(token),
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("reports", [])


def api_admin_delete_report(token: str, report_id: str) -> dict:
    r = requests.delete(
        f"{BACKEND_URL}/admin/reports/{report_id}",
        headers=_auth_headers(token),
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_admin(session: dict | None) -> bool:
    return (
        session is not None
        and "admin_panel" in session.get("authorized_features", [])
    )


def _err_msg(exc: Exception) -> str:
    """Render a user-facing error string from any exception."""
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        if exc.response.status_code == 401:
            return "Your session has expired. Please log out and log in again."
        try:
            detail = exc.response.json().get("detail")
        except Exception:
            detail = exc.response.text
        return f"Backend error ({exc.response.status_code}): {detail}"
    return f"Error: {exc}"


def format_query_response(result: dict) -> str:
    """Render a /orchestrate?action=query response as Markdown for the chat."""
    if not result.get("success"):
        return f"Query failed: {result.get('message', 'Unknown error.')}"

    count = result.get("result_count", 0)
    if count == 0:
        return "No relevant results found for your question."

    lines = [f"Found **{count}** relevant result(s):\n"]
    for i, r in enumerate(result.get("results_metadata", []), 1):
        score = r.get("rerank_score") if r.get("rerank_score") is not None else r.get("rrf_score")
        score_str = f"{score:.3f}" if score is not None else "N/A"
        section = (r.get("section_title") or "").strip()
        section_part = f" | Section: *{section}*" if section else ""
        lines.append(f"**{i}.** Page {r['page_number']}{section_part} | Score: {score_str}")

    return "\n".join(lines)


def format_ingest_response(result: dict) -> str:
    """Render a /orchestrate?action=ingest response as a Markdown summary."""
    if not result.get("success"):
        return f"Ingestion failed: {result.get('message', 'Unknown error.')}"

    return (
        f"Ingestion complete.\n\n"
        f"- **Report ID:** {result.get('report_id', 'N/A')}\n"
        f"- **Pages processed:** {result.get('pages_processed', '?')} / {result.get('pages_total', '?')}\n"
        f"- **Chunks inserted:** {result.get('chunks_inserted', '?')}\n"
        f"- **Latency:** {result.get('total_latency_ms', '?')} ms"
    )


def _channels_table(channels: list[dict]) -> list[list]:
    return [
        [
            c["channelid"],
            c["name"],
            c.get("rag_channel_id") or "— not linked —",
            ", ".join(c.get("authorized_features") or []),
        ]
        for c in channels
    ]


def _unlinked_choices(channels: list[dict]) -> list[tuple[str, int]]:
    return [(c["name"], c["channelid"]) for c in channels if not c.get("rag_channel_id")]


def _linked_choices(channels: list[dict]) -> list[tuple[str, int]]:
    return [(c["name"], c["channelid"]) for c in channels if c.get("rag_channel_id")]


def _ingest_choices(channels: list[dict]) -> list[tuple[str, str]]:
    return [
        (c["name"], c["rag_channel_id"])
        for c in channels
        if c.get("rag_channel_id")
    ]


def _reports_table(reports: list[dict]) -> list[list]:
    return [
        [
            r["report_id"],
            r["filename"],
            r.get("title") or "",
            r.get("page_count", 0),
            r.get("chunk_count", 0),
            (r.get("created_at") or "")[:19].replace("T", " "),
        ]
        for r in reports
    ]


def _report_delete_choices(reports: list[dict]) -> list[tuple[str, str]]:
    return [
        (f"{r['filename']} ({r['report_id'][:8]}…)", r["report_id"])
        for r in reports
    ]


# ── Gradio app ────────────────────────────────────────────────────────────────

with gr.Blocks(title="Epsilon AI") as app:

    # session holds the bearer token and the user profile after login.
    # channels_state caches the channel list for admin tab refreshes.
    session_state  = gr.State(value=None)
    channels_state = gr.State(value=[])

    # ── Login screen ──────────────────────────────────────────────────────────
    with gr.Column(visible=True) as login_screen:
        gr.Markdown("# EPSILON AI")
        gr.Markdown("### Login")
        username_input = gr.Textbox(label="Username", placeholder="Enter username")
        password_input = gr.Textbox(label="Password", placeholder="Enter password", type="password")
        login_btn      = gr.Button("Login", variant="primary")
        login_msg      = gr.Markdown("")

    # ── Main screen ───────────────────────────────────────────────────────────
    with gr.Column(visible=False) as main_screen:
        gr.Markdown("# EPSILON AI")
        session_label = gr.Markdown("")

        with gr.Tabs():

            # Chat tab — visible to every authenticated user.
            with gr.Tab("Chat"):
                chatbot = gr.Chatbot(
                    label="Epsilon AI",
                    type="messages",
                    height=460,
                )
                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Ask anything...",
                        show_label=False,
                        scale=8,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

            # Channels tab — admin only. Three sections:
            #   1. Read-only overview table of every control channel.
            #   2. Form to add a brand-new control channel.
            #   3. Form to create a RAG channel and link it to a control channel.
            with gr.Tab("Channels", visible=False) as channels_tab:
                gr.Markdown("## Channels overview")
                channels_table = gr.Dataframe(
                    headers=["channelid", "name", "rag_channel_id", "authorized_features"],
                    datatype=["number", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                )
                refresh_channels_btn = gr.Button("Refresh channels")

                gr.Markdown("---")
                gr.Markdown("## Add a new channel")
                gr.Markdown(
                    "Creates a tenant row in `control.channel`. RAG linking is a "
                    "separate step below — fill that in next to enable ingestion."
                )
                add_channel_name = gr.Textbox(
                    label="Channel name (required, unique)",
                    placeholder="e.g. Acme Industries",
                )
                add_channel_description = gr.Textbox(
                    label="Description (optional)",
                    placeholder="Stored in metadata.description",
                )
                add_channel_features = gr.CheckboxGroup(
                    label="Authorized features",
                    choices=FEATURE_OPTIONS,
                    value=["chatbot", "document_search"],
                )
                add_channel_btn = gr.Button("Add channel", variant="primary")
                add_channel_status = gr.Markdown("")

                gr.Markdown("---")
                gr.Markdown("## Create & link a RAG channel")
                gr.Markdown(
                    "Pick a control channel that is **not yet linked**, then click "
                    "**Create & link**. A new RAG channel will be created in the RAG "
                    "schema and its UUID will be saved to `control.channel.rag_channel_id`."
                )
                link_channel_dropdown = gr.Dropdown(
                    label="Unlinked control channel",
                    choices=[],
                    interactive=True,
                )
                link_name_input = gr.Textbox(
                    label="RAG channel name (optional — defaults to control channel name)",
                    placeholder="Leave empty to reuse the control channel name",
                )
                link_description_input = gr.Textbox(
                    label="Description (optional)",
                    placeholder="Free-form description for the RAG channel",
                )
                link_btn = gr.Button("Create & link", variant="primary")
                link_status = gr.Markdown("")

            # Reports tab — admin only. Lists reports for a chosen channel and
            # provides a per-report delete control.
            with gr.Tab("Reports", visible=False) as reports_tab:
                gr.Markdown("## Uploaded reports")
                reports_channel_dropdown = gr.Dropdown(
                    label="Channel (linked channels only)",
                    choices=[],
                    interactive=True,
                )
                refresh_reports_btn = gr.Button("Load reports")
                reports_table = gr.Dataframe(
                    headers=["report_id", "filename", "title", "pages", "chunks", "created_at"],
                    datatype=["str", "str", "str", "number", "number", "str"],
                    interactive=False,
                    wrap=True,
                )

                gr.Markdown("---")
                gr.Markdown("### Delete a report")
                delete_report_dropdown = gr.Dropdown(
                    label="Report to delete",
                    choices=[],
                    interactive=True,
                )
                delete_report_btn = gr.Button("Delete report", variant="stop")
                delete_report_status = gr.Markdown("")

            # Ingest tab — admin only. Upload a PDF and run the full pipeline
            # against the chosen channel (must already be RAG-linked).
            with gr.Tab("Ingest PDF", visible=False) as ingest_tab:
                gr.Markdown("## Ingest a PDF into a channel")
                ingest_channel_dropdown = gr.Dropdown(
                    label="Channel (linked channels only)",
                    choices=[],
                    interactive=True,
                )
                pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
                title_input = gr.Textbox(label="Document title (optional)")
                ingest_btn = gr.Button("Ingest document", variant="primary")
                ingest_status = gr.Markdown("")

        logout_btn = gr.Button("Logout", variant="secondary")

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _refresh_dropdowns(channels: list[dict]):
        """Compute fresh values for the channels table and the three dropdowns
        that depend on it. Returned in the order the callers wire to outputs."""
        return (
            _channels_table(channels),
            gr.update(choices=_unlinked_choices(channels), value=None),
            gr.update(choices=_linked_choices(channels), value=None),
            gr.update(choices=_ingest_choices(channels), value=None),
        )

    def handle_send(message: str, history: list, session: dict | None):
        if not message.strip():
            return history, ""
        if not session:
            history = history + [
                {"role": "user",      "content": message},
                {"role": "assistant", "content": "You are not logged in."},
            ]
            return history, ""

        history = history + [{"role": "user", "content": message}]

        try:
            result   = api_orchestrate(session["token"], "query", {"question": message})
            response = format_query_response(result)
        except Exception as e:
            response = _err_msg(e)

        history = history + [{"role": "assistant", "content": response}]
        return history, ""

    send_btn.click(
        fn=handle_send,
        inputs=[chat_input, chatbot, session_state],
        outputs=[chatbot, chat_input],
    )
    chat_input.submit(
        fn=handle_send,
        inputs=[chat_input, chatbot, session_state],
        outputs=[chatbot, chat_input],
    )

    def handle_login(username: str, password: str):
        """Authenticate and toggle visibility of the login/main screens and
        the admin tabs. Also preloads the channel list for admin sessions."""
        try:
            result = api_login(username, password)
        except Exception as e:
            return (
                None, [],
                gr.update(visible=True), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                "", _err_msg(e),
                [], gr.update(choices=[], value=None),
                gr.update(choices=[], value=None), gr.update(choices=[], value=None),
            )

        if not result["success"]:
            return (
                None, [],
                gr.update(visible=True), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                "", result["message"],
                [], gr.update(choices=[], value=None),
                gr.update(choices=[], value=None), gr.update(choices=[], value=None),
            )

        session = {
            "token":               result["token"],
            "name":                result["name"],
            "channel_name":        result["channel_name"],
            "authorized_features": result["authorized_features"],
        }
        admin = _is_admin(session)

        channels: list[dict] = []
        if admin:
            try:
                channels = api_admin_channels(session["token"])
            except Exception:
                channels = []

        table_rows, link_update, reports_update, ingest_update = _refresh_dropdowns(channels)

        return (
            session, channels,
            gr.update(visible=False), gr.update(visible=True),
            gr.update(visible=admin), gr.update(visible=admin), gr.update(visible=admin),
            f"Logged in as **{result['name']}** | Channel: **{result['channel_name']}**",
            result["message"],
            table_rows, link_update, reports_update, ingest_update,
        )

    login_btn.click(
        fn=handle_login,
        inputs=[username_input, password_input],
        outputs=[
            session_state, channels_state,
            login_screen, main_screen,
            channels_tab, reports_tab, ingest_tab,
            session_label, login_msg,
            channels_table, link_channel_dropdown,
            reports_channel_dropdown, ingest_channel_dropdown,
        ],
    )

    # ── Channels handlers ─────────────────────────────────────────────────────

    def handle_refresh_channels(session: dict | None):
        if not session:
            return [], [], gr.update(), gr.update(), gr.update()
        try:
            channels = api_admin_channels(session["token"])
        except Exception:
            return [], [], gr.update(), gr.update(), gr.update()

        table_rows, link_update, reports_update, ingest_update = _refresh_dropdowns(channels)
        return channels, table_rows, link_update, reports_update, ingest_update

    refresh_channels_btn.click(
        fn=handle_refresh_channels,
        inputs=[session_state],
        outputs=[
            channels_state, channels_table,
            link_channel_dropdown, reports_channel_dropdown, ingest_channel_dropdown,
        ],
    )

    def handle_add_channel(
        session: dict | None,
        name: str,
        description: str,
        features: list[str] | None,
    ):
        """Create a new control.channel and refresh the dropdowns so the new
        row is immediately available for RAG linking."""
        if not session:
            return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    "Not logged in.")
        if not name.strip():
            return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    "Please enter a channel name.")

        try:
            result = api_admin_create_channel(
                session["token"],
                name.strip(),
                description.strip(),
                features or [],
            )
        except Exception as e:
            return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    _err_msg(e))

        try:
            channels = api_admin_channels(session["token"])
        except Exception:
            channels = []

        table_rows, link_update, reports_update, ingest_update = _refresh_dropdowns(channels)
        msg = f"Created channel **{result['name']}** (channelid `{result['channelid']}`)."
        return channels, table_rows, link_update, reports_update, ingest_update, msg

    add_channel_btn.click(
        fn=handle_add_channel,
        inputs=[session_state, add_channel_name, add_channel_description, add_channel_features],
        outputs=[
            channels_state, channels_table,
            link_channel_dropdown, reports_channel_dropdown, ingest_channel_dropdown,
            add_channel_status,
        ],
    )

    def handle_create_link(
        session: dict | None,
        channelid,
        name: str,
        description: str,
    ):
        """Create a RAG channel and link it to the chosen control channel."""
        if not session:
            return ([], [], gr.update(), gr.update(), gr.update(), "Not logged in.")
        if channelid is None:
            return (
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                "Please select an unlinked channel.",
            )

        try:
            result = api_admin_create_rag_channel(
                session["token"], int(channelid), name.strip(), description.strip(),
            )
        except Exception as e:
            return (
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                _err_msg(e),
            )

        try:
            channels = api_admin_channels(session["token"])
        except Exception:
            channels = []

        table_rows, link_update, reports_update, ingest_update = _refresh_dropdowns(channels)
        msg = (
            f"Created and linked RAG channel **{result['name']}** "
            f"(`{result['rag_channel_id']}`) to control channel `{result['channelid']}`."
        )
        return channels, table_rows, link_update, reports_update, ingest_update, msg

    link_btn.click(
        fn=handle_create_link,
        inputs=[session_state, link_channel_dropdown, link_name_input, link_description_input],
        outputs=[
            channels_state, channels_table,
            link_channel_dropdown, reports_channel_dropdown, ingest_channel_dropdown,
            link_status,
        ],
    )

    # ── Reports handlers ──────────────────────────────────────────────────────

    def handle_refresh_reports(session: dict | None, channelid):
        if not session:
            return [], gr.update(choices=[], value=None), "Not logged in."
        if channelid is None:
            return [], gr.update(choices=[], value=None), "Please select a channel."
        try:
            reports = api_admin_list_reports(session["token"], int(channelid))
        except Exception as e:
            return [], gr.update(choices=[], value=None), _err_msg(e)
        return (
            _reports_table(reports),
            gr.update(choices=_report_delete_choices(reports), value=None),
            f"Loaded {len(reports)} report(s).",
        )

    refresh_reports_btn.click(
        fn=handle_refresh_reports,
        inputs=[session_state, reports_channel_dropdown],
        outputs=[reports_table, delete_report_dropdown, delete_report_status],
    )

    def handle_delete_report(session: dict | None, channelid, report_id: str):
        """Delete a report and refresh the list for the same channel."""
        if not session:
            return [], gr.update(), "Not logged in."
        if not report_id:
            return gr.update(), gr.update(), "Please select a report to delete."
        try:
            result = api_admin_delete_report(session["token"], report_id)
        except Exception as e:
            return gr.update(), gr.update(), _err_msg(e)

        if channelid is None:
            return gr.update(), gr.update(choices=[], value=None), (
                f"Deleted report (removed {result.get('deleted_chunks', 0)} chunks)."
            )

        try:
            reports = api_admin_list_reports(session["token"], int(channelid))
        except Exception:
            reports = []

        return (
            _reports_table(reports),
            gr.update(choices=_report_delete_choices(reports), value=None),
            f"Deleted report (removed {result.get('deleted_chunks', 0)} chunks).",
        )

    delete_report_btn.click(
        fn=handle_delete_report,
        inputs=[session_state, reports_channel_dropdown, delete_report_dropdown],
        outputs=[reports_table, delete_report_dropdown, delete_report_status],
    )

    # ── Ingest handler ────────────────────────────────────────────────────────

    def handle_ingest(file, rag_channel_id: str, title: str, session: dict | None):
        """Upload the PDF to the backend, then trigger the orchestrator's
        ingest action against the chosen RAG channel."""
        if not session:
            return "Not logged in."
        if not rag_channel_id:
            return "Please select a channel."
        if file is None:
            return "Please upload a PDF file."

        file_path = file.name if hasattr(file, "name") else str(file)
        filename  = os.path.basename(file_path)

        try:
            upload_result = api_upload(session["token"], file_path, filename)
        except Exception as e:
            return _err_msg(e)

        try:
            result = api_orchestrate(
                session["token"],
                "ingest",
                {
                    "rag_channel_id": rag_channel_id,
                    "file_path":      upload_result["file_path"],
                    "filename":       upload_result["filename"],
                    "title":          (title or "").strip(),
                },
            )
        except Exception as e:
            return _err_msg(e)

        return format_ingest_response(result)

    ingest_btn.click(
        fn=handle_ingest,
        inputs=[pdf_upload, ingest_channel_dropdown, title_input, session_state],
        outputs=[ingest_status],
    )

    # ── Logout handler ────────────────────────────────────────────────────────

    def handle_logout(session: dict | None):
        """Clear the session and return the UI to the login screen."""
        if session and session.get("token"):
            api_logout(session["token"])
        return (
            None, [],
            gr.update(visible=True), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            "", "",
            "", "",
            [],
            [],
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            "",
            "",
        )

    logout_btn.click(
        fn=handle_logout,
        inputs=[session_state],
        outputs=[
            session_state, channels_state,
            login_screen, main_screen,
            channels_tab, reports_tab, ingest_tab,
            session_label, login_msg,
            username_input, password_input,
            chatbot,
            channels_table,
            link_channel_dropdown, reports_channel_dropdown,
            ingest_channel_dropdown, delete_report_dropdown,
            link_status, delete_report_status,
        ],
    )


# ── Launch ────────────────────────────────────────────────────────────────────
# Two access modes for Lightning AI users:
#   * GRADIO_SHARE=true → app.launch(share=True) opens an https://*.gradio.live
#     public tunnel and prints the URL in the logs.
#   * Otherwise → mount Gradio inside a small FastAPI app that strips the
#     X-Frame-Options header so Lightning AI's iframe wrapper for port 7860
#     can display the UI directly.

if os.getenv("GRADIO_SHARE", "false").lower() == "true":
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
else:
    from fastapi import FastAPI
    from starlette.middleware.base import BaseHTTPMiddleware
    import uvicorn

    class _AllowIframe(BaseHTTPMiddleware):
        """Remove anti-iframe response headers so Lightning AI's wrapper page
        can embed the UI in its own iframe view."""
        async def dispatch(self, request, call_next):
            response = await call_next(request)
            response.headers.pop("X-Frame-Options", None)
            response.headers.pop("x-frame-options", None)
            response.headers["Content-Security-Policy"] = "frame-ancestors *"
            return response

    app.queue()
    _main = FastAPI()
    _main.add_middleware(_AllowIframe)
    gr.mount_gradio_app(_main, app, path="/")

    uvicorn.run(_main, host="0.0.0.0", port=7860, log_level="info")
