"""Gateway adapter: chat-id routing, inbound dispatch, and authorization.

Skipped when Hermes itself is not importable, so the rest of the suite still
runs in a bare checkout.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("gateway.platforms.base", reason="requires a Hermes checkout on sys.path")

from gateway.config import PlatformConfig

from clawtalk import register, wire


class FakeCtx:
    """Captures everything ``register(ctx)`` wires up."""

    def __init__(self) -> None:
        self.tools: list[str] = []
        self.platforms: list[dict[str, Any]] = []
        self.cli: list[str] = []
        self.commands: list[str] = []
        self.skills: list[str] = []

    def get_config(self, _key: str, default: Any = None) -> Any:
        return default

    def register_tool(self, **kwargs: Any) -> None:
        self.tools.append(kwargs["name"])

    def register_platform(self, **kwargs: Any) -> None:
        self.platforms.append(kwargs)
        # Register for real as well: Platform("clawtalk") only resolves for a
        # name the registry knows, and the adapter looks it up in __init__.
        from gateway.platform_registry import PlatformEntry, platform_registry

        entry_kwargs = dict(kwargs)
        entry_kwargs.setdefault("plugin_name", "clawtalk")
        platform_registry.register(
            PlatformEntry(source="plugin", **entry_kwargs),
            scope=platform_registry.current_scope_key(),
        )

    def register_cli_command(self, **kwargs: Any) -> None:
        self.cli.append(kwargs["name"])

    def register_command(self, name: str, **_kwargs: Any) -> None:
        self.commands.append(name)

    def register_skill(self, name: str, _path: Any, **_kwargs: Any) -> None:
        self.skills.append(name)


class FakeWs:
    """Stands in for the live WebSocket, recording outbound frames."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.is_connected = True
        self.fatal_error: str | None = None
        self.last_ping = None
        self.last_pong = None

    async def try_send(self, frame: dict[str, Any]) -> bool:
        self.sent.append(frame)
        return True


@pytest.fixture(scope="module")
def registered() -> FakeCtx:
    """Register the plugin once so ``Platform("clawtalk")`` resolves."""
    ctx = FakeCtx()
    register(ctx)
    return ctx


@pytest.fixture
def adapter(registered, data_dir, monkeypatch):
    from clawtalk.adapter import ClawTalkAdapter

    monkeypatch.setenv("CLAWTALK_API_KEY", "ct_test_key")
    monkeypatch.setattr("clawtalk.runtime.resolve_data_dir", lambda: data_dir)

    instance = ClawTalkAdapter(PlatformConfig(enabled=True, extra={"api_key": "ct_test_key"}))
    instance._ws = FakeWs()

    # Capture dispatched turns instead of running the gateway pipeline.
    dispatched: list[Any] = []

    async def capture(event):
        dispatched.append(event)

    instance.handle_message = capture  # type: ignore[method-assign]
    instance.dispatched = dispatched  # type: ignore[attr-defined]
    return instance


class TestRegistration:
    def test_registers_every_surface(self, registered):
        assert len(registered.tools) == 21
        assert len(registered.platforms) == 1
        assert registered.cli == ["clawtalk"]
        assert registered.commands == ["clawtalk"]
        assert "missions" in registered.skills

    def test_platform_entry_is_wired_for_cron_and_auth(self, registered):
        entry = registered.platforms[0]
        assert entry["name"] == "clawtalk"
        assert entry["allowed_users_env"] == "CLAWTALK_ALLOWED_USERS"
        assert entry["cron_deliver_env_var"] == "CLAWTALK_HOME_CHANNEL"
        assert callable(entry["standalone_sender_fn"])
        assert entry["pii_safe"] is True

    def test_target_parser_accepts_namespaced_ids(self, registered):
        parse = registered.platforms[0]["parse_target_ref_fn"]
        assert parse("sms:+15551234567") == ("sms:+15551234567", None)
        assert parse("#general") is None


class TestAuthorization:
    def test_delegates_upstream_by_default(self, adapter):
        # ClawTalk's server gates callers before anything reaches the gateway,
        # and a phone number is not a platform account an operator allowlists.
        assert adapter.authorization_is_upstream is True

    def test_local_allowlist_takes_authorization_back(self, registered, data_dir, monkeypatch):
        from clawtalk.adapter import ClawTalkAdapter

        monkeypatch.setattr("clawtalk.runtime.resolve_data_dir", lambda: data_dir)
        monkeypatch.setenv("CLAWTALK_ALLOWED_USERS", "+15551234567")

        instance = ClawTalkAdapter(
            PlatformConfig(enabled=True, extra={"api_key": "ct_test_key"})
        )
        assert instance.authorization_is_upstream is False


class TestSharedRuntime:
    def test_tools_and_adapter_share_one_runtime(self, adapter):
        """The approval waiter and the socket that resolves it must agree.

        clawtalk_approve blocks on the runtime's ApprovalManager while the
        adapter's socket resolves the decision on its own. If the adapter
        built a private runtime, every approval would time out.
        """
        from clawtalk.runtime import get_runtime

        shared = get_runtime()
        assert shared is adapter._runtime
        assert shared.approvals is adapter._runtime.approvals

    def test_status_tool_sees_the_live_socket(self, adapter, monkeypatch):
        import json

        from clawtalk.runtime import get_runtime
        from clawtalk.tools.status import clawtalk_status

        adapter._runtime.ws = adapter._ws
        monkeypatch.setattr("clawtalk.tools.common.get_runtime", get_runtime)

        result = json.loads(clawtalk_status({}))
        assert result["connected"] is True
        assert result["websocket_state"] == "open"


class TestSendRouting:
    def test_first_voice_reply_answers_the_deep_tool_request(self, adapter):
        adapter._pending_deep["c1"] = "req_1"
        result = asyncio.run(adapter.send("call:c1", "**Found** three messages"))

        assert result.success is True
        frame = adapter._ws.sent[0]
        assert frame["type"] == "deep_tool_result"
        assert frame["request_id"] == "req_1"
        # Markdown would be read out character by character.
        assert frame["text"] == "Found three messages"

    def test_later_voice_replies_become_spoken_progress(self, adapter):
        adapter._pending_deep["c1"] = "req_1"
        asyncio.run(adapter.send("call:c1", "first"))
        asyncio.run(adapter.send("call:c1", "still working"))

        kinds = [frame["type"] for frame in adapter._ws.sent]
        assert kinds == ["deep_tool_result", "response"]

    def test_walkie_reply_uses_the_request_id(self, adapter):
        adapter._pending_walkie["walkie:default"] = "req_9"
        asyncio.run(adapter.send("walkie:walkie:default", "On it"))

        frame = adapter._ws.sent[0]
        assert frame["type"] == "walkie_response"
        assert frame["request_id"] == "req_9"

    def test_walkie_reply_without_a_request_is_dropped(self, adapter):
        result = asyncio.run(adapter.send("walkie:default", "orphan"))
        assert result.success is True
        assert adapter._ws.sent == []

    def test_sms_reply_is_truncated_and_sent_over_rest(self, adapter, monkeypatch):
        sent: dict[str, Any] = {}

        def fake_send(to: str, message: str, media_urls=None):
            sent.update({"to": to, "message": message})
            return {"id": "msg_1"}

        monkeypatch.setattr(adapter._runtime.client.sms, "send", fake_send)
        adapter._sms_numbers["15551234567"] = "+15551234567"

        result = asyncio.run(adapter.send("sms:15551234567", "x" * 500))

        assert result.success is True
        assert sent["to"] == "+15551234567"
        assert len(sent["message"]) == 300

    def test_sms_reply_reconstructs_an_unseen_number(self, adapter, monkeypatch):
        sent: dict[str, Any] = {}
        monkeypatch.setattr(
            adapter._runtime.client.sms,
            "send",
            lambda to, message, media_urls=None: sent.update({"to": to}) or {"id": "m"},
        )
        asyncio.run(adapter.send("sms:15551234567", "hi"))
        assert sent["to"] == "+15551234567"

    def test_mission_replies_are_not_delivered(self, adapter):
        result = asyncio.run(adapter.send("mission:call-alice", "I'll wait for the reply"))
        assert result.success is True
        assert adapter._ws.sent == []

    def test_unknown_chat_id_fails_loudly(self, adapter):
        result = asyncio.run(adapter.send("nonsense", "hi"))
        assert result.success is False
        assert "Unknown ClawTalk chat id" in result.error


class TestInboundDispatch:
    def test_context_request_answers_with_prompt_and_greeting(self, adapter):
        asyncio.run(adapter._on_context_request({"call_id": "c1"}))

        kinds = [frame["type"] for frame in adapter._ws.sent]
        assert kinds == ["context_response", "response"]
        assert "VOICE RULES" in adapter._ws.sent[0]["context"]["system_prompt"]
        assert adapter._ws.sent[1]["text"] == "Hey there, what's up?"
        # A caller mid-conversation must not be greeted twice.
        assert adapter._greeted["c1"] is True

    def test_inbound_call_started_greets_once(self, adapter):
        asyncio.run(adapter._on_call_started({"call_id": "c1", "direction": "inbound"}))
        asyncio.run(adapter._on_call_started({"call_id": "c1", "direction": "inbound"}))
        assert len(adapter._ws.sent) == 1

    def test_outbound_call_started_does_not_greet(self, adapter):
        asyncio.run(adapter._on_call_started({"call_id": "c1", "direction": "outbound"}))
        assert adapter._ws.sent == []

    def test_deep_tool_request_dispatches_into_the_call_session(self, adapter):
        asyncio.run(
            adapter._on_deep_tool_request(
                {"call_id": "c1", "request_id": "req_1", "query": "check Slack"}
            )
        )
        asyncio.run(asyncio.sleep(0))

        event = adapter.dispatched[0]
        assert event.source.chat_id == "call:c1"
        assert event.text.startswith("[VOICE CALL]")
        assert "check Slack" in event.text
        assert "VOICE RULES" in event.channel_prompt
        # An external caller must never resolve a gateway slash command.
        assert event.allow_gateway_control is False
        assert adapter._pending_deep["c1"] == "req_1"

    def test_sms_dispatches_per_contact_with_the_sms_prompt(self, adapter):
        asyncio.run(
            adapter._on_sms_received(
                {"from": "+15551234567", "body": "you around?", "message_id": "m1"}
            )
        )
        asyncio.run(asyncio.sleep(0))

        event = adapter.dispatched[0]
        assert event.source.chat_id == "sms:15551234567"
        assert event.text == "you around?"
        assert "MAX 300 CHARACTERS" in event.channel_prompt
        assert adapter._sms_numbers["15551234567"] == "+15551234567"

    def test_mms_attachments_are_offered_to_the_agent(self, adapter):
        asyncio.run(
            adapter._on_sms_received(
                {
                    "from": "+15551234567",
                    "body": "what is this",
                    "media_urls": ["https://example.com/a.png"],
                }
            )
        )
        asyncio.run(asyncio.sleep(0))
        assert "https://example.com/a.png" in adapter.dispatched[0].text

    def test_walkie_dispatch_records_the_pending_request(self, adapter):
        asyncio.run(
            adapter._on_walkie_request({"request_id": "req_9", "transcript": "status?"})
        )
        asyncio.run(asyncio.sleep(0))

        event = adapter.dispatched[0]
        assert event.source.chat_id == "walkie:walkie:default"
        assert event.channel_prompt.startswith("[WALKIE-TALKIE]")
        assert adapter._pending_walkie["walkie:default"] == "req_9"

    def test_call_ended_reports_the_outcome(self, adapter):
        asyncio.run(
            adapter._on_call_ended(
                {
                    "call_id": "c1",
                    "direction": "outbound",
                    "to_number": "+15551234567",
                    "reason": "user_hangup",
                    "duration_seconds": 90,
                }
            )
        )
        asyncio.run(asyncio.sleep(0))

        event = adapter.dispatched[0]
        assert event.source.chat_id == "events:calls"
        assert "1 minute 30 seconds" in event.text

    def test_mission_event_routes_to_the_slug_session(self, adapter):
        adapter._runtime.missions.update_slug_state(
            "call-alice", {"mission_id": "m1", "run_id": "r1"}
        )
        asyncio.run(
            adapter._on_mission_event(
                {
                    "type": "event",
                    "event": wire.EVENT_MISSION_CALL_FAILED,
                    "mission_id": "m1",
                    "step_id": "s1",
                    "reason": "no-answer",
                    "from": "+1",
                    "to": "+2",
                }
            )
        )
        asyncio.run(asyncio.sleep(0))

        event = adapter.dispatched[0]
        assert event.source.chat_id == "mission:call-alice"
        assert "Call FAILED" in event.text

    def test_approval_response_reaches_the_manager(self, adapter):
        seen = {}
        adapter._runtime.approvals.handle_ws_response = lambda msg: seen.update(msg)
        adapter._on_approval_responded({"request_id": "req_1", "decision": "approved"})
        assert seen["decision"] == "approved"

    def test_log_request_is_answered(self, adapter):
        adapter._runtime.ws_log.open()
        adapter._runtime.ws_log.lifecycle("connected", "wss://clawdtalk.com/ws")

        asyncio.run(adapter._on_request_logs("req_logs_1"))

        frame = adapter._ws.sent[0]
        assert frame["type"] == "logs_response"
        assert frame["request_id"] == "req_logs_1"
        assert any("connected" in line for line in frame["lines"])
