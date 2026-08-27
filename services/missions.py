"""Mission lifecycle orchestration.

A mission is a multi-step outreach campaign: create mission -> create run ->
create plan -> set up a voice assistant -> schedule calls/SMS -> advance plan
steps -> complete.

This service owns two things:

* the local state file (``missions_state.json``), which maps a URL-safe slug
  to the server IDs and scratch memory for that mission, and
* every server call needed to drive the lifecycle.

Unlike the OpenClaw original, reads and writes are serialised behind a lock
and written atomically, so concurrent updates from the WebSocket handler, the
observer, and a tool call cannot interleave into a corrupt file.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..formatting import slugify
from ..sdk import ClawTalkClient

__all__ = [
    "EVENT_TYPES",
    "STEP_STATUSES",
    "TERMINAL_STEP_STATUSES",
    "MissionService",
]

logger = logging.getLogger(__name__)

STATE_FILENAME = "missions_state.json"

STEP_STATUSES = ("pending", "in_progress", "completed", "failed", "skipped")
TERMINAL_STEP_STATUSES = frozenset({"completed", "failed", "skipped"})

#: Allowed plan-step transitions. Terminal states are absent by design: once
#: a step is terminal it cannot move again.
VALID_TRANSITIONS: dict[str, frozenset] = {
    "pending": frozenset({"in_progress", "skipped"}),
    "in_progress": frozenset({"completed", "failed", "skipped"}),
}

EVENT_TYPES = (
    "step_started",
    "step_completed",
    "step_failed",
    "call_scheduled",
    "call_completed",
    "sms_scheduled",
    "sms_sent",
    "agent_linked",
    "agent_unlinked",
    "note",
    "error",
)

DEFAULT_ASSISTANT_MODEL = "openai/gpt-4o"
DEFAULT_ASSISTANT_FEATURES = ("telephony", "messaging")
DEFAULT_EVENT_AGENT_ID = "hermes-plugin"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


class MissionService:
    """Local mission state plus the server calls that advance a mission."""

    def __init__(self, client: ClawTalkClient, data_dir: str | os.PathLike[str]) -> None:
        self._client = client
        self._state_path = Path(data_dir) / STATE_FILENAME
        self._lock = threading.RLock()

    @property
    def client(self) -> ClawTalkClient:
        """The SDK client, for tools that need direct server access."""
        return self._client

    @property
    def state_path(self) -> Path:
        return self._state_path

    # -- state persistence -------------------------------------------------

    def load_state(self) -> dict[str, dict[str, Any]]:
        """Read the whole state file. Missing or corrupt files read as ``{}``."""
        with self._lock:
            try:
                raw = self._state_path.read_text(encoding="utf-8")
            except (OSError, ValueError):
                return {}
            try:
                parsed = json.loads(raw)
            except ValueError:
                logger.warning(
                    "[clawtalk] mission state at %s is not valid JSON; ignoring",
                    self._state_path,
                )
                return {}
            return parsed if isinstance(parsed, dict) else {}

    def save_state(self, state: Mapping[str, Any]) -> None:
        """Write the state file atomically (temp file + rename)."""
        with self._lock:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            tmp.replace(self._state_path)

    def get_slug_state(self, slug: str) -> dict[str, Any]:
        return dict(self.load_state().get(slug, {}))

    def update_slug_state(self, slug: str, updates: Mapping[str, Any]) -> None:
        with self._lock:
            state = self.load_state()
            entry = dict(state.get(slug, {}))
            entry.update(updates)
            state[slug] = entry
            self.save_state(state)

    def remove_slug_state(self, slug: str) -> None:
        with self._lock:
            state = self.load_state()
            state.pop(slug, None)
            self.save_state(state)

    def list_missions(self) -> list[tuple[str, dict[str, Any]]]:
        """Return ``(slug, state)`` for every locally tracked mission."""
        return list(self.load_state().items())

    def resolve_slug(self, mission_id: str) -> str | None:
        """Map a server mission ID back to its local slug, if we track it."""
        for slug, entry in self.load_state().items():
            if entry.get("mission_id") == mission_id:
                return slug
        return None

    # -- memory ------------------------------------------------------------

    def save_memory(self, slug: str, key: str, value: Any) -> None:
        """Store a value under ``memory[key]`` for this mission."""
        with self._lock:
            state = self.load_state()
            entry = dict(state.get(slug, {}))
            memory = dict(entry.get("memory") or {})
            memory[key] = value
            entry["memory"] = memory
            entry["last_updated"] = _utc_now()
            state[slug] = entry
            self.save_state(state)
        logger.info("[clawtalk] saved memory '%s' for mission '%s'", key, slug)

    def get_memory(self, slug: str, key: str | None = None) -> Any:
        """Return one memory value, or the whole memory dict when *key* is None."""
        memory = self.get_slug_state(slug).get("memory") or {}
        return memory.get(key) if key else memory

    def append_memory(self, slug: str, key: str, item: Any) -> int:
        """Append to a list-valued memory key, coercing scalars to a list.

        Returns the new list length.
        """
        with self._lock:
            state = self.load_state()
            entry = dict(state.get(slug, {}))
            memory = dict(entry.get("memory") or {})

            existing = memory.get(key)
            if isinstance(existing, list):
                items = list(existing)
            elif existing is not None:
                items = [existing]
            else:
                items = []
            items.append(item)

            memory[key] = items
            entry["memory"] = memory
            entry["last_updated"] = _utc_now()
            state[slug] = entry
            self.save_state(state)

        logger.info(
            "[clawtalk] appended to memory '%s' for mission '%s' (now %d items)",
            key,
            slug,
            len(items),
        )
        return len(items)

    # -- lifecycle ---------------------------------------------------------

    def init_mission(
        self,
        name: str,
        instructions: str,
        request: str,
        steps: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Create a mission, its first run, and an optional plan.

        Idempotent: if the slug already has both IDs on file, the existing
        mission is resumed rather than duplicated.
        """
        slug = slugify(name)
        existing = self.get_slug_state(slug)

        if existing.get("mission_id") and existing.get("run_id"):
            logger.info("[clawtalk] resuming existing mission: %s", slug)
            return {
                "mission_id": existing["mission_id"],
                "run_id": existing["run_id"],
                "slug": slug,
                "resumed": True,
            }

        mission = self._client.missions.create(name=name, instructions=instructions)
        mission_id = str(mission.get("id") or "")
        logger.info("[clawtalk] created mission: %s", mission_id)

        self.update_slug_state(
            slug,
            {"mission_name": name, "mission_id": mission_id, "created_at": _utc_now()},
        )

        run = self._client.missions.runs.create(mission_id, {"original_request": request})
        run_id = str(run.get("run_id") or "")
        logger.info("[clawtalk] created run: %s", run_id)
        self.update_slug_state(slug, {"run_id": run_id})

        if steps:
            plan_steps = [
                {
                    "step_id": slugify(str(step.get("title", ""))),
                    "sequence": index + 1,
                    "title": step.get("title"),
                    "description": step.get("description"),
                    "status": "pending",
                }
                for index, step in enumerate(steps)
            ]
            self._client.missions.plans.create(mission_id, run_id, plan_steps)
            logger.info("[clawtalk] created plan with %d steps", len(plan_steps))

        self._client.missions.runs.update(mission_id, run_id, {"status": "running"})

        return {
            "mission_id": mission_id,
            "run_id": run_id,
            "slug": slug,
            "resumed": False,
        }

    def setup_voice_agent(
        self,
        slug: str,
        name: str,
        instructions: str,
        *,
        greeting: str | None = None,
        voice: str | None = None,
        model: str | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        features: Sequence[str] | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Create the assistant, link it to the run, and claim a phone number.

        Idempotent: returns the existing assistant when one is already on file
        for this slug.
        """
        existing = self.get_slug_state(slug)

        if existing.get("assistant_id") and existing.get("agent_phone"):
            logger.info("[clawtalk] using existing assistant: %s", existing["assistant_id"])
            return {
                "assistant_id": existing["assistant_id"],
                "phone": existing["agent_phone"],
            }

        assistant = self._client.assistants.create(
            name=name,
            instructions=instructions,
            greeting=greeting or "",
            voice=voice,
            model=model or DEFAULT_ASSISTANT_MODEL,
            tools=tools,
            enabled_features=list(features or DEFAULT_ASSISTANT_FEATURES),
            description=description,
        )
        assistant_id = str(assistant.get("id") or "")
        logger.info("[clawtalk] created assistant: %s", assistant_id)
        self.update_slug_state(slug, {"assistant_id": assistant_id})

        mission_id = existing.get("mission_id")
        run_id = existing.get("run_id")
        if mission_id and run_id:
            self._client.missions.agents.link(mission_id, run_id, assistant_id)
            logger.info("[clawtalk] linked agent %s to run %s", assistant_id, run_id)

        phone: str | None = None
        try:
            available = self._client.numbers.available()
            phone = available.get("phone_number")
            self.update_slug_state(
                slug,
                {"agent_phone": phone, "phone_number_id": available.get("id")},
            )
            logger.info("[clawtalk] assigned phone: %s", phone)
        except Exception as exc:  # noqa: BLE001 - a missing number is not fatal
            logger.warning(
                "[clawtalk] no available phone number (%s). Outbound calls/SMS "
                "cannot be scheduled without a dedicated number.",
                exc,
            )

        return {"assistant_id": assistant_id, "phone": phone}

    def complete_mission(
        self,
        slug: str,
        summary: str,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        """Mark the run succeeded and drop local state.

        Refuses while any plan step is still non-terminal, so a mission can
        never be closed with work silently outstanding.
        """
        mission_id, run_id = self._require_ids(slug)

        steps = self._client.missions.plans.get(mission_id, run_id)
        outstanding = [
            step for step in steps if step.get("status") not in TERMINAL_STEP_STATUSES
        ]
        if outstanding:
            listed = ", ".join(
                f"{step.get('step_id')} ({step.get('status')})" for step in outstanding
            )
            raise ValueError(
                f"Cannot complete mission '{slug}': {len(outstanding)} step(s) still "
                f"non-terminal: {listed}. Mark each step as completed, failed, or "
                f"skipped before completing the mission."
            )

        self._client.missions.runs.update(
            mission_id,
            run_id,
            {
                "status": "succeeded",
                "result_summary": summary,
                "result_payload": dict(payload) if payload else None,
            },
        )
        self.remove_slug_state(slug)
        logger.info("[clawtalk] mission '%s' completed successfully", slug)

    # -- scheduling --------------------------------------------------------

    def schedule_call(
        self, slug: str, to: str, scheduled_at: str, step_id: str | None = None
    ) -> str:
        """Schedule an outbound call. Returns the scheduled event ID."""
        assistant_id, agent_phone = self._require_assistant(slug)
        mission_id, run_id = self._require_ids(slug)

        event = self._client.assistants.events.schedule(
            assistant_id=assistant_id,
            to=to,
            from_=agent_phone,
            scheduled_at=scheduled_at,
            mission_id=mission_id,
            run_id=run_id,
            step_id=step_id,
        )
        event_id = str(event.get("id") or "")
        logger.info("[clawtalk] scheduled call: %s", event_id)
        return event_id

    def schedule_sms(
        self,
        slug: str,
        to: str,
        scheduled_at: str,
        text_body: str,
        step_id: str | None = None,
    ) -> str:
        """Schedule an outbound SMS. Returns the scheduled event ID."""
        assistant_id, agent_phone = self._require_assistant(slug)
        mission_id, run_id = self._require_ids(slug)

        event = self._client.assistants.events.schedule(
            assistant_id=assistant_id,
            to=to,
            from_=agent_phone,
            scheduled_at=scheduled_at,
            text_body=text_body,
            mission_id=mission_id,
            run_id=run_id,
            step_id=step_id,
        )
        event_id = str(event.get("id") or "")
        logger.info("[clawtalk] scheduled SMS: %s", event_id)
        return event_id

    def get_scheduled_event(self, slug: str, event_id: str) -> dict[str, Any]:
        assistant_id, _ = self._require_assistant(slug, need_phone=False)
        return self._client.assistants.events.get(assistant_id, event_id)

    def cancel_scheduled_event(self, slug: str, event_id: str) -> None:
        assistant_id, _ = self._require_assistant(slug, need_phone=False)
        self._client.assistants.events.cancel(assistant_id, event_id)
        logger.info("[clawtalk] cancelled scheduled event: %s", event_id)

    # -- events and plan ---------------------------------------------------

    def log_event(
        self,
        slug: str,
        event_type: str,
        summary: str,
        *,
        agent_id: str | None = None,
        step_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> str:
        """Append an event to the mission run's log. Returns the event ID."""
        mission_id, run_id = self._require_ids(slug)
        event = self._client.missions.events.log(
            mission_id,
            run_id,
            {
                "type": event_type,
                "summary": summary,
                "agent_id": agent_id or DEFAULT_EVENT_AGENT_ID,
                "step_id": step_id,
                "payload": dict(payload) if payload else None,
            },
        )
        logger.info("[clawtalk] logged event: %s", summary)
        return str(event.get("id") or "")

    def get_plan(self, slug: str) -> list[dict[str, Any]]:
        mission_id, run_id = self._require_ids(slug)
        return self._client.missions.plans.get(mission_id, run_id)

    def get_events(self, slug: str) -> dict[str, Any]:
        state = self.get_slug_state(slug)
        mission_id = state.get("mission_id")
        if not mission_id:
            raise ValueError(f"No active mission found for slug '{slug}'")
        return self._client.missions.events.aggregate(mission_id)

    def update_plan_step(self, slug: str, step_id: str, status: str) -> dict[str, Any]:
        """Move a plan step, enforcing the transition rules client-side.

        The server enforces these too, but checking here produces a message
        the model can act on instead of a bare 4xx.
        """
        mission_id, run_id = self._require_ids(slug)

        steps = self._client.missions.plans.get(mission_id, run_id)
        step = next((s for s in steps if s.get("step_id") == step_id), None)
        if step is None:
            raise ValueError(f"Step '{step_id}' not found in mission '{slug}'")

        current = str(step.get("status") or "pending")
        if current in TERMINAL_STEP_STATUSES:
            raise ValueError(
                f"Cannot update step '{step_id}': already in terminal state "
                f"'{current}'. Terminal states (completed, failed, skipped) "
                f"cannot be changed."
            )

        allowed = VALID_TRANSITIONS.get(current)
        if allowed is not None and status not in allowed:
            raise ValueError(
                f"Invalid transition for step '{step_id}': '{current}' -> '{status}'. "
                f"Allowed transitions from '{current}': {', '.join(sorted(allowed))}."
            )

        return self._client.missions.plans.update_step(
            mission_id, run_id, step_id, status
        )

    def get_insights(self, conversation_id: str) -> dict[str, Any]:
        return self._client.insights.get(conversation_id)

    # -- internals ---------------------------------------------------------

    def _require_ids(self, slug: str) -> tuple[str, str]:
        state = self.get_slug_state(slug)
        mission_id = state.get("mission_id")
        run_id = state.get("run_id")
        if not mission_id or not run_id:
            raise ValueError(f"No active mission found for slug '{slug}'")
        return str(mission_id), str(run_id)

    def _require_assistant(
        self, slug: str, need_phone: bool = True
    ) -> tuple[str, str]:
        state = self.get_slug_state(slug)
        assistant_id = state.get("assistant_id")
        agent_phone = state.get("agent_phone")

        if not assistant_id:
            raise ValueError(f"Mission '{slug}' has no assistant set up")
        if need_phone and not agent_phone:
            raise ValueError(f"Mission '{slug}' has no assistant/phone set up")

        return str(assistant_id), str(agent_phone or "")
