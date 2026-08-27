"""ClawTalk server endpoint map - single source of truth.

Every endpoint the SDK can call is declared here. Namespaces reference these
paths and the integration tests iterate over :data:`READ_ENDPOINTS` to verify
reachability.

Path params use ``:paramName`` notation; :func:`resolve` interpolates them.

Server route file -> mount point::

    user.js          -> /v1
    calls.js         -> /v1/calls
    messages.js      -> /v1/messages
    approvals.js     -> /v1/approvals
    missions.js      -> /v1/missions
    assistants.js    -> /v1/assistants
    numbers.js       -> /v1/numbers
    conversations.js -> /v1/conversations
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import quote

__all__ = [
    "ENDPOINTS",
    "IMPLEMENTED_ENDPOINTS",
    "READ_ENDPOINTS",
    "UNIMPLEMENTED_ENDPOINTS",
    "Endpoint",
    "resolve",
]

_PARAM_RE = re.compile(r":(\w+)")


@dataclass(frozen=True)
class Endpoint:
    """A single REST endpoint on the ClawTalk server."""

    method: str
    path: str
    #: Which SDK method wraps this endpoint, or ``None`` if unimplemented.
    sdk_method: str | None
    #: Whether calling it creates or modifies data.
    write: bool


def _ep(method: str, path: str, sdk_method: str | None, write: bool) -> Endpoint:
    return Endpoint(method=method, path=path, sdk_method=sdk_method, write=write)


ENDPOINTS: dict[str, Endpoint] = {
    # -- User (/v1 + user.js) ------------------------------------------
    "getMe": _ep("GET", "/v1/me", "user.me", False),
    "updateMe": _ep("PATCH", "/v1/me", "user.update_me", True),
    # -- Voices (/v1 + user.js) ----------------------------------------
    "listVoices": _ep("GET", "/v1/voices", "voices.list", False),
    # -- Calls (/v1/calls + calls.js) ----------------------------------
    "initiateCall": _ep("POST", "/v1/calls", "calls.initiate", True),
    "getCallStatus": _ep("GET", "/v1/calls/:callId", "calls.status", False),
    "endCall": _ep("POST", "/v1/calls/:callId/end", "calls.end", True),
    # -- SMS (/v1/messages + messages.js) ------------------------------
    "sendSms": _ep("POST", "/v1/messages/send", "sms.send", True),
    "listMessages": _ep("GET", "/v1/messages", "sms.list", False),
    "listConversations": _ep(
        "GET", "/v1/messages/conversations", "sms.conversations", False
    ),
    # -- Approvals (/v1/approvals + approvals.js) ----------------------
    "createApproval": _ep("POST", "/v1/approvals", "approvals.create", True),
    "getApprovalStatus": _ep(
        "GET", "/v1/approvals/:requestId", "approvals.status", False
    ),
    # -- Missions (/v1/missions + missions.js) -------------------------
    "createMission": _ep("POST", "/v1/missions", "missions.create", True),
    "getMission": _ep("GET", "/v1/missions/:missionId", "missions.get", False),
    "listMissions": _ep("GET", "/v1/missions", "missions.list", False),
    "cancelMission": _ep("POST", "/v1/missions/:missionId/cancel", None, True),
    # -- Runs ----------------------------------------------------------
    "createRun": _ep(
        "POST", "/v1/missions/:missionId/runs", "missions.runs.create", True
    ),
    "getRun": _ep(
        "GET", "/v1/missions/:missionId/runs/:runId", "missions.runs.get", False
    ),
    "updateRun": _ep(
        "PATCH", "/v1/missions/:missionId/runs/:runId", "missions.runs.update", True
    ),
    "listRuns": _ep(
        "GET", "/v1/missions/:missionId/runs", "missions.runs.list", False
    ),
    # -- Plans ---------------------------------------------------------
    "createPlan": _ep(
        "POST",
        "/v1/missions/:missionId/runs/:runId/plan",
        "missions.plans.create",
        True,
    ),
    "getPlan": _ep(
        "GET", "/v1/missions/:missionId/runs/:runId/plan", "missions.plans.get", False
    ),
    "updateStep": _ep(
        "PATCH",
        "/v1/missions/:missionId/runs/:runId/plan/steps/:stepId",
        "missions.plans.update_step",
        True,
    ),
    # -- Mission events ------------------------------------------------
    "logEvent": _ep(
        "POST",
        "/v1/missions/:missionId/runs/:runId/events",
        "missions.events.log",
        True,
    ),
    "listEvents": _ep(
        "GET",
        "/v1/missions/:missionId/runs/:runId/events",
        "missions.events.list",
        False,
    ),
    "getMissionEvents": _ep(
        "GET", "/v1/missions/:missionId/events", "missions.events.aggregate", False
    ),
    # -- Linked agents -------------------------------------------------
    "linkAgent": _ep(
        "POST",
        "/v1/missions/:missionId/runs/:runId/agents",
        "missions.agents.link",
        True,
    ),
    "listLinkedAgents": _ep(
        "GET",
        "/v1/missions/:missionId/runs/:runId/agents",
        "missions.agents.list",
        False,
    ),
    "unlinkAgent": _ep(
        "DELETE",
        "/v1/missions/:missionId/runs/:runId/agents/:agentId",
        "missions.agents.unlink",
        True,
    ),
    # -- Insights ------------------------------------------------------
    "getInsights": _ep(
        "GET",
        "/v1/missions/conversations/:conversationId/insights",
        "insights.get",
        False,
    ),
    "getRecording": _ep("GET", "/v1/missions/recordings/:recordingId", None, False),
    # -- Assistants (/v1/assistants + assistants.js) -------------------
    "createAssistant": _ep("POST", "/v1/assistants", "assistants.create", True),
    "listAssistants": _ep("GET", "/v1/assistants", "assistants.list", False),
    "getAssistant": _ep(
        "GET", "/v1/assistants/:assistantId", "assistants.get", False
    ),
    "updateAssistant": _ep(
        "PATCH", "/v1/assistants/:assistantId", "assistants.update", True
    ),
    "deleteAssistant": _ep("DELETE", "/v1/assistants/:assistantId", None, True),
    "getConnectionId": _ep(
        "GET",
        "/v1/assistants/:assistantId/connection-id",
        "assistants.connection_id",
        False,
    ),
    "assignPhone": _ep(
        "POST",
        "/v1/assistants/:assistantId/assign-phone",
        "assistants.assign_phone",
        True,
    ),
    # -- Scheduled events ----------------------------------------------
    "scheduleEvent": _ep(
        "POST", "/v1/assistants/:assistantId/events", "assistants.events.schedule", True
    ),
    "listScheduledEvents": _ep(
        "GET", "/v1/assistants/:assistantId/events", None, False
    ),
    "getScheduledEvent": _ep(
        "GET",
        "/v1/assistants/:assistantId/events/:eventId",
        "assistants.events.get",
        False,
    ),
    "cancelScheduledEvent": _ep(
        "DELETE",
        "/v1/assistants/:assistantId/events/:eventId",
        "assistants.events.cancel",
        True,
    ),
    # -- Phone numbers (/v1/numbers + numbers.js) ----------------------
    "getAvailablePhone": _ep(
        "GET", "/v1/numbers/account-phones/available", "numbers.available", False
    ),
    "assignPhoneNumber": _ep(
        "PATCH", "/v1/numbers/account-phones/:phoneId", "numbers.assign", True
    ),
    "listAccountPhones": _ep("GET", "/v1/numbers/account-phones", None, False),
    "searchPhones": _ep("GET", "/v1/numbers/search", None, False),
    "orderPhone": _ep("POST", "/v1/numbers/order", None, True),
    "releasePhone": _ep("POST", "/v1/numbers/release", None, True),
    "listMyPhones": _ep("GET", "/v1/numbers/mine", None, False),
    # -- Doctor (/v1/doctor) -------------------------------------------
    "doctorCritical": _ep("GET", "/v1/doctor/critical", "doctor.critical", False),
    "doctorWarnings": _ep("GET", "/v1/doctor/warnings", "doctor.warnings", False),
    "doctorRecommended": _ep(
        "GET", "/v1/doctor/recommended", "doctor.recommended", False
    ),
    "doctorInfra": _ep("GET", "/v1/doctor/infra", "doctor.infra", False),
}


def resolve(path: str, params: Mapping[str, str] | None = None) -> str:
    """Interpolate ``:name`` segments in *path*.

    ``resolve(ENDPOINTS["getCallStatus"].path, {"callId": "call_123"})``
    -> ``"/v1/calls/call_123"``.

    Raises:
        KeyError: when the path needs a param that *params* does not supply.
    """
    values = dict(params or {})

    def _sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values or values[key] is None:
            raise KeyError(f'Missing path param ":{key}" for path "{path}"')
        return quote(str(values[key]), safe="")

    return _PARAM_RE.sub(_sub, path)


#: Endpoints the SDK has a wrapper method for.
IMPLEMENTED_ENDPOINTS: dict[str, Endpoint] = {
    name: ep for name, ep in ENDPOINTS.items() if ep.sdk_method is not None
}

#: Endpoints the server exposes that the SDK does not wrap yet.
UNIMPLEMENTED_ENDPOINTS: dict[str, Endpoint] = {
    name: ep for name, ep in ENDPOINTS.items() if ep.sdk_method is None
}

#: Read-only endpoints, safe to probe in integration tests with fake IDs.
READ_ENDPOINTS: dict[str, Endpoint] = {
    name: ep for name, ep in ENDPOINTS.items() if not ep.write
}
