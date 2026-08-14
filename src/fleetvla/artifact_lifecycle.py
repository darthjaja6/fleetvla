"""Cross-event lifecycle validation for version-1 artifacts."""

from __future__ import annotations

import math
from typing import Any

from .benchmark import _EVENT_CLOCK_TOLERANCE_S


def _validate_inference_lifecycle(
    events: list[dict[str, Any]],
    *,
    system: bool,
    expected_sessions: set[str],
    duration_s: float,
    max_batch_size: int,
    model_output_horizon: int | None = None,
    execution_horizon: int | None = None,
    require_scheduler_events: bool = False,
) -> None:
    stage: str | None = None
    batch_size: int | None = None
    batch_sessions: tuple[str, ...] = ()
    pending_requests: list[tuple[str, int, int]] = []
    batch_sequences: dict[str, int] = {}
    expected_chunks: dict[str, tuple[int, bool, float]] = {}
    expected_resets: set[str] = set()
    invalidated_sessions: set[str] = set()
    registered: list[str] = []
    registration_ended = False
    started_at_s: float | None = None
    modeled_latency_s: float | None = None
    pending_decision: tuple[tuple[str, ...], bool] | None = None
    scheduler_fallback = False
    saw_scheduler_decision = False
    scheduler_failure_pending = False

    for event in events:
        kind = event["kind"]
        details = event["details"] or {}
        session_id = event["session_id"]
        if not system and event["time_s"] > duration_s:
            raise ValueError("algorithm event occurs after benchmark duration")
        if kind == "session_registered":
            if registration_ended:
                raise ValueError("sessions must be registered at startup")
            registered.append(session_id)
            continue
        registration_ended = True
        if scheduler_failure_pending and kind != "scheduler_decision":
            raise ValueError("scheduler failure must immediately select fallback")
        if kind == "scheduler_failed":
            if (
                not system
                or pending_decision is not None
                or stage is not None
                or expected_chunks
                or expected_resets
            ):
                raise ValueError("scheduler failure overlaps another transition")
            scheduler_fallback = True
            scheduler_failure_pending = True
            continue
        if kind == "scheduler_decision":
            if (
                not system
                or pending_decision is not None
                or stage is not None
                or pending_requests
                or expected_chunks
                or expected_resets
                or details["fallback"] != scheduler_fallback
            ):
                raise ValueError("scheduler decision occurs in an invalid state")
            pending_decision = (
                tuple(details["selected_session_ids"]),
                details["deferred"],
            )
            saw_scheduler_decision = True
            scheduler_failure_pending = False
            continue
        if pending_decision is not None and kind not in {
            "request_dispatched",
            "batch_dispatched",
            "dispatch_deferred",
        }:
            raise ValueError("scheduler decision was not applied atomically")
        if kind == "dispatch_deferred" and (
            stage is not None or pending_requests or expected_chunks or expected_resets
        ):
            raise ValueError("dispatch deferral overlaps an inference lifecycle")
        if kind == "dispatch_deferred" and pending_decision is not None:
            if not pending_decision[1]:
                raise ValueError("dispatch deferral does not match its decision")
            pending_decision = None
        elif kind == "dispatch_deferred" and require_scheduler_events:
            raise ValueError("dispatch deferral lacks a scheduler decision")
        if system and kind in {"session_reset", "session_disconnected"}:
            if session_id in batch_sessions:
                invalidated_sessions.add(session_id)
            if session_id in expected_chunks:
                sequence, _, produced_at_s = expected_chunks[session_id]
                expected_chunks[session_id] = (sequence, True, produced_at_s)

        if kind == "request_dispatched":
            if stage is not None or expected_chunks or expected_resets:
                raise ValueError("request dispatched before the prior batch ended")
            pending_requests.append(
                (session_id, details["batch_size"], details["sequence"])
            )
        elif kind == "batch_dispatched":
            selected = tuple(details["session_ids"])
            if pending_decision is not None:
                if pending_decision[1] or pending_decision[0] != selected:
                    raise ValueError("batch does not match its scheduler decision")
                pending_decision = None
            elif require_scheduler_events:
                raise ValueError("batch lacks a scheduler decision")
            if (
                stage is not None
                or expected_chunks
                or expected_resets
                or len(selected) > max_batch_size
            ):
                raise ValueError("a batch was dispatched before the prior batch ended")
            expected_requests = [
                (selected_session, details["batch_size"])
                for selected_session in selected
            ]
            if [request[:2] for request in pending_requests] != expected_requests:
                raise ValueError("batch sessions do not match dispatched requests")
            batch_sequences = {
                session: sequence for session, _, sequence in pending_requests
            }
            pending_requests.clear()
            stage = "dispatched"
            batch_size = details["batch_size"]
            batch_sessions = selected
            invalidated_sessions.clear()
        elif kind == "scheduler_cost_estimate":
            if (
                not system
                or stage != "dispatched"
                or details["batch_size"] != batch_size
            ):
                raise ValueError("scheduler cost event does not match its batch")
            stage = "costed"
        elif kind == "inference_started":
            expected_stage = "costed" if system else "dispatched"
            if stage != expected_stage or details["batch_size"] != batch_size:
                raise ValueError("inference start does not match its batch")
            modeled_latency_s = details.get("latency_s")
            if not system and modeled_latency_s is None:
                raise ValueError("algorithm inference start lacks modeled latency")
            started_at_s = event["time_s"]
            stage = "started"
        elif kind == "inference_completed":
            if stage != "started" or details["batch_size"] != batch_size:
                raise ValueError("inference completion does not match its start")
            if system:
                measured_latency_s = details.get("latency_s")
                backend_latency_s = details.get("backend_reported_latency_s")
                shapes = details.get("output_shapes")
                horizons = details.get("execution_horizons")
                if (
                    measured_latency_s is None
                    or backend_latency_s is None
                    or started_at_s is None
                    or not math.isclose(
                        measured_latency_s,
                        event["time_s"] - started_at_s,
                        rel_tol=0,
                        abs_tol=_EVENT_CLOCK_TOLERANCE_S,
                    )
                    or backend_latency_s > measured_latency_s + _EVENT_CLOCK_TOLERANCE_S
                    or not isinstance(shapes, list)
                    or any(
                        len(shape) != 3 or shape[1] != model_output_horizon
                        for shape in shapes
                    )
                    or not isinstance(horizons, list)
                    or any(horizon != execution_horizon for horizon in horizons)
                ):
                    raise ValueError(
                        "inference timing or output does not match the system trace"
                    )
                expected_chunks = {
                    session: (
                        sequence,
                        session in invalidated_sessions,
                        started_at_s + backend_latency_s,
                    )
                    for session, sequence in batch_sequences.items()
                }
            stage = None
            batch_size = None
            batch_sessions = ()
            batch_sequences = {}
            invalidated_sessions.clear()
            started_at_s = None
            modeled_latency_s = None
        elif kind == "inference_failed":
            expected_stage = "costed" if details["phase"] == "prepare" else "started"
            if (
                not system
                or stage != expected_stage
                or tuple(details["session_ids"]) != batch_sessions
            ):
                raise ValueError("inference failure does not match its batch")
            expected_resets = set(batch_sessions)
            stage = None
            batch_size = None
            batch_sessions = ()
            batch_sequences = {}
            invalidated_sessions.clear()
            started_at_s = None
            modeled_latency_s = None
        elif system and kind in {
            "chunk_accepted",
            "chunk_rejected_stale",
            "chunk_rejected_unexpected",
            "chunk_rejected_horizon",
        }:
            if session_id not in expected_chunks:
                raise ValueError("chunk outcome does not match a completed batch")
            expected_sequence, invalidated, produced_at_s = expected_chunks[session_id]
            if invalidated != (kind == "chunk_rejected_stale"):
                raise ValueError("chunk outcome does not match session invalidation")
            if kind == "chunk_accepted" and (
                details["sequence"] != expected_sequence
                or details["actions"] != execution_horizon
                or not math.isclose(
                    details["action_age_s"],
                    event["time_s"] - produced_at_s,
                    rel_tol=0,
                    abs_tol=_EVENT_CLOCK_TOLERANCE_S,
                )
            ):
                raise ValueError("accepted chunk does not match its request")
            if kind == "chunk_rejected_horizon" and (
                details["expected_actions"] != execution_horizon
            ):
                raise ValueError("horizon rejection does not match system config")
            del expected_chunks[session_id]
        elif kind == "session_reset" and session_id in expected_resets:
            expected_resets.remove(session_id)

    if (
        len(registered) != len(expected_sessions)
        or set(registered) != expected_sessions
    ):
        raise ValueError("artifact registrations do not match configured sessions")
    if pending_requests:
        raise ValueError("artifact contains requests without a batch")
    if pending_decision is not None:
        raise ValueError("artifact contains an incomplete scheduler decision")
    if scheduler_failure_pending:
        raise ValueError("artifact contains an incomplete scheduler failure")
    if require_scheduler_events and not saw_scheduler_decision:
        raise ValueError("system artifact lacks scheduler decision evidence")
    if system and (stage is not None or expected_chunks or expected_resets):
        raise ValueError("system artifact contains an incomplete inference lifecycle")
    if not system and stage not in {None, "started"}:
        raise ValueError("algorithm artifact contains an incomplete dispatch")
    if not system and stage == "started":
        if (
            modeled_latency_s is None
            or started_at_s is None
            or started_at_s + modeled_latency_s <= duration_s
        ):
            raise ValueError("algorithm artifact omits a due inference completion")


def _validate_system_session_lifecycle(
    events: list[dict[str, Any]],
    expected_sessions: set[str],
    *,
    duration_s: float,
    control_hz: float,
    action_execution: str,
) -> None:
    buffers: dict[str, list[tuple[int, int]]] = {
        session_id: [] for session_id in expected_sessions
    }
    outstanding: dict[str, tuple[int, int] | None] = {
        session_id: None for session_id in expected_sessions
    }
    action_sent = {session_id: False for session_id in expected_sessions}
    action_accepted = {session_id: False for session_id in expected_sessions}
    connected = {session_id: True for session_id in expected_sessions}
    generations = {session_id: 0 for session_id in expected_sessions}
    next_sequences = {session_id: 0 for session_id in expected_sessions}
    ready: dict[str, int | None] = {
        session_id: None for session_id in expected_sessions
    }
    observation_times: dict[str, dict[int, float]] = {
        session_id: {} for session_id in expected_sessions
    }
    observation_action_starts: dict[str, dict[int, int]] = {
        session_id: {} for session_id in expected_sessions
    }
    next_action_indices = {session_id: 0 for session_id in expected_sessions}
    ready_times: dict[str, float | None] = {
        session_id: None for session_id in expected_sessions
    }
    dispatched_times: dict[str, float | None] = {
        session_id: None for session_id in expected_sessions
    }
    fallback_due = {session_id: False for session_id in expected_sessions}
    task_step_due = {session_id: False for session_id in expected_sessions}
    boundary_due = {session_id: False for session_id in expected_sessions}
    reset_due = {session_id: False for session_id in expected_sessions}
    failure_reset_due = {session_id: False for session_id in expected_sessions}
    action_failure_due = {session_id: False for session_id in expected_sessions}
    disconnect_due = {session_id: False for session_id in expected_sessions}
    ticks = {session_id: 0 for session_id in expected_sessions}

    for event in events:
        kind = event["kind"]
        details = event["details"] or {}
        session_id = event["session_id"]

        if kind == "inference_failed":
            for failed_session in details["session_ids"]:
                failure_reset_due[failed_session] = True
            continue
        if kind == "batch_dispatched":
            for state in details["selected_state"]:
                selected_session = state["session_id"]
                captured_at_s = dispatched_times[selected_session]
                if captured_at_s is None:
                    raise ValueError("selected session has no dispatched observation")
                expected_age_s = event["time_s"] - captured_at_s
                expected_horizon_s = len(buffers[selected_session]) / control_hz
                if not math.isclose(
                    state["request_age_s"],
                    expected_age_s,
                    rel_tol=0,
                    abs_tol=_EVENT_CLOCK_TOLERANCE_S,
                ) or not math.isclose(
                    state["buffer_horizon_s"],
                    expected_horizon_s,
                    rel_tol=0,
                    abs_tol=1e-9,
                ):
                    raise ValueError("selected_state does not match session history")
                dispatched_times[selected_session] = None
            continue
        if session_id is None:
            continue
        if dispatched_times[session_id] is not None:
            raise ValueError(
                "dispatched request must enter its batch before local work"
            )
        if outstanding[session_id] is not None and kind not in {
            "action_sent_endpoint",
            "action_accepted_endpoint",
            "action_executed",
            "action_rejected_endpoint",
        }:
            raise ValueError(
                "dequeued action must receive its outcome before local work"
            )
        if failure_reset_due[session_id] and kind != "session_reset":
            raise ValueError("inference failure must be followed by a session reset")
        if fallback_due[session_id] and kind not in {
            "endpoint_fallback",
            "endpoint_fallback_failed",
        }:
            raise ValueError("starved or reset session must complete its fallback")
        if task_step_due[session_id] and kind != "endpoint_task_step":
            raise ValueError("executed action or fallback must report its task step")
        if action_failure_due[session_id] and kind != "endpoint_action_failed":
            raise ValueError("rejected action must report its endpoint failure")
        if disconnect_due[session_id] and kind != "session_disconnected":
            raise ValueError("endpoint failure must be followed by disconnect")
        if boundary_due[session_id] and kind != "endpoint_episode_boundary":
            raise ValueError(
                "terminal task step must be followed by an episode boundary"
            )
        if reset_due[session_id] and kind != "session_reset":
            raise ValueError("episode boundary must be followed by a session reset")

        if kind == "observation_ready":
            if (
                not connected[session_id]
                or ready[session_id] is not None
                or details["sequence"] != next_sequences[session_id]
            ):
                raise ValueError("observation sequence does not match session state")
            ready[session_id] = details["sequence"]
            observation_times[session_id][details["sequence"]] = event["time_s"]
            observation_action_starts[session_id][details["sequence"]] = (
                next_action_indices[session_id]
            )
            ready_times[session_id] = event["time_s"]
            next_sequences[session_id] += 1
        elif kind == "request_dispatched":
            if ready[session_id] != details["sequence"]:
                raise ValueError("request does not match a ready observation")
            dispatched_times[session_id] = ready_times[session_id]
            ready[session_id] = None
            ready_times[session_id] = None
        elif kind == "chunk_accepted":
            if not connected[session_id]:
                raise ValueError("a disconnected session accepted a chunk")
            sequence = details["sequence"]
            if action_execution == "latest-indexed":
                action_start = observation_action_starts[session_id].get(sequence)
                if action_start is None:
                    raise ValueError("accepted chunk has no observation action index")
                first_action = max(0, next_action_indices[session_id] - action_start)
                buffers[session_id].clear()
                buffers[session_id].extend(
                    (sequence, action_start + index)
                    for index in range(first_action, details["actions"])
                )
            else:
                buffers[session_id].extend(
                    (sequence, index) for index in range(details["actions"])
                )
        elif kind == "action_dequeued":
            key = (details["sequence"], details["action_index"])
            captured_at_s = observation_times[session_id].get(details["sequence"])
            if (
                not connected[session_id]
                or outstanding[session_id] is not None
                or not buffers[session_id]
                or buffers[session_id][0] != key
                or captured_at_s is None
                or not math.isclose(
                    details["action_age_s"],
                    event["time_s"] - captured_at_s,
                    rel_tol=0,
                    abs_tol=_EVENT_CLOCK_TOLERANCE_S,
                )
            ):
                raise ValueError("dequeued action is not the next buffered action")
            buffers[session_id].pop(0)
            if action_execution == "latest-indexed":
                next_action_indices[session_id] = details["action_index"] + 1
            if details["remaining_steps"] != len(buffers[session_id]):
                raise ValueError("dequeued action reports an invalid remaining count")
            outstanding[session_id] = key
            action_sent[session_id] = False
            action_accepted[session_id] = False
        elif kind == "action_sent_endpoint":
            key = (details["sequence"], details["action_index"])
            if (
                outstanding[session_id] != key
                or action_sent[session_id]
                or details["deadline_s"] < event["time_s"]
            ):
                raise ValueError("sent action does not match a dequeued action")
            action_sent[session_id] = True
        elif kind == "action_accepted_endpoint":
            key = (details["sequence"], details["action_index"])
            if (
                outstanding[session_id] != key
                or not action_sent[session_id]
                or action_accepted[session_id]
            ):
                raise ValueError("accepted action does not match a sent action")
            action_accepted[session_id] = True
        elif kind in {"action_executed", "action_rejected_endpoint"}:
            key = (details["sequence"], details["action_index"])
            captured_at_s = observation_times[session_id].get(details["sequence"])
            if (
                outstanding[session_id] != key
                or captured_at_s is None
                or (
                    kind == "action_executed"
                    and action_sent[session_id]
                    and not action_accepted[session_id]
                )
                or not math.isclose(
                    details["action_age_s"],
                    event["time_s"] - captured_at_s,
                    rel_tol=0,
                    abs_tol=_EVENT_CLOCK_TOLERANCE_S,
                )
            ):
                raise ValueError("action outcome does not match a dequeued action")
            outstanding[session_id] = None
            action_sent[session_id] = False
            action_accepted[session_id] = False
            ticks[session_id] += 1
            if kind == "action_executed":
                task_step_due[session_id] = True
            else:
                action_failure_due[session_id] = True
        elif kind == "action_starved":
            if (
                not connected[session_id]
                or buffers[session_id]
                or outstanding[session_id] is not None
            ):
                raise ValueError("session reported starvation with an action available")
            fallback_due[session_id] = True
            ticks[session_id] += 1
        elif kind == "control_ticks_missed":
            ticks[session_id] += details["count"]
        elif kind == "endpoint_fallback":
            if not fallback_due[session_id]:
                raise ValueError("endpoint fallback has no corresponding control tick")
            fallback_due[session_id] = False
            task_step_due[session_id] = True
        elif kind == "endpoint_fallback_failed":
            if not fallback_due[session_id]:
                raise ValueError("failed endpoint fallback was not requested")
            fallback_due[session_id] = False
            disconnect_due[session_id] = True
        elif kind == "endpoint_action_failed":
            if not action_failure_due[session_id]:
                raise ValueError("endpoint action failure has no rejected action")
            action_failure_due[session_id] = False
            disconnect_due[session_id] = True
        elif kind == "endpoint_observation_failed":
            disconnect_due[session_id] = True
        elif kind == "endpoint_task_step":
            if not task_step_due[session_id]:
                raise ValueError("task step has no executed action or fallback")
            task_step_due[session_id] = False
            if details["terminated"] or details["truncated"]:
                boundary_due[session_id] = True
        elif kind == "endpoint_episode_boundary":
            if not boundary_due[session_id]:
                raise ValueError("episode boundary has no terminal task step")
            boundary_due[session_id] = False
            reset_due[session_id] = True
        elif kind == "session_reset":
            after_failure = failure_reset_due[session_id]
            if not (reset_due[session_id] or after_failure):
                raise ValueError("session reset has no failure or episode boundary")
            buffers[session_id].clear()
            outstanding[session_id] = None
            action_sent[session_id] = False
            action_accepted[session_id] = False
            ready[session_id] = None
            ready_times[session_id] = None
            dispatched_times[session_id] = None
            reset_due[session_id] = False
            failure_reset_due[session_id] = False
            generations[session_id] += 1
            next_action_indices[session_id] = 0
            if details["generation"] != generations[session_id]:
                raise ValueError("session reset reports an invalid generation")
            if after_failure:
                fallback_due[session_id] = True
        elif kind == "session_disconnected":
            if not connected[session_id] or not disconnect_due[session_id]:
                raise ValueError("session disconnect has no endpoint failure")
            disconnect_due[session_id] = False
            connected[session_id] = False
            buffers[session_id].clear()
            outstanding[session_id] = None
            action_sent[session_id] = False
            action_accepted[session_id] = False
            ready[session_id] = None
            ready_times[session_id] = None
            dispatched_times[session_id] = None
            generations[session_id] += 1
            next_action_indices[session_id] = 0
        elif kind == "session_reconnected":
            if connected[session_id]:
                raise ValueError("connected session was reconnected")
            if details["generation"] != generations[session_id]:
                raise ValueError("session reconnect reports an invalid generation")
            connected[session_id] = True

    expected_ticks = math.floor(duration_s * control_hz + 1e-9)
    if any(count != expected_ticks for count in ticks.values()):
        raise ValueError("system event ticks do not match duration and control_hz")
    pending = (
        outstanding,
        action_sent,
        action_accepted,
        fallback_due,
        task_step_due,
        boundary_due,
        reset_due,
        failure_reset_due,
        action_failure_due,
        disconnect_due,
    )
    if any(any(values.values()) for values in pending):
        raise ValueError("system artifact ends with an incomplete session lifecycle")
