"""Wall-clock serving loop using the same runtime and scheduler contract."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from typing import Any

from .backend import BackendResult
from .clock import MonotonicClock
from .endpoints import (
    Endpoint,
    ExecutionOutcome,
    ObservationUnavailable,
    validate_observation,
)
from .runtime import FleetRuntime
from .schedulers import Scheduler
from .types import ActionChunk, Observation


class AsyncServingEngine:
    """Batch endpoint observations while control loops continue asynchronously."""

    def __init__(
        self,
        endpoints: list[Endpoint],
        backend: Any,
        scheduler: Scheduler,
        *,
        max_batch_size: int = 8,
        inference_timeout_s: float | None = None,
    ) -> None:
        if not endpoints:
            raise ValueError("at least one endpoint is required")
        self.endpoints = {
            endpoint.session_config.session_id: endpoint for endpoint in endpoints
        }
        if len(self.endpoints) != len(endpoints):
            raise ValueError("endpoint session IDs must be unique")
        self.backend = backend
        self.scheduler = scheduler
        if inference_timeout_s is not None and (
            not math.isfinite(inference_timeout_s) or inference_timeout_s <= 0
        ):
            raise ValueError("inference_timeout_s must be positive")
        self.inference_timeout_s = inference_timeout_s
        self.clock = MonotonicClock()
        self.runtime = FleetRuntime(self.clock, max_batch_size=max_batch_size)
        for endpoint in endpoints:
            self.runtime.register(endpoint.session_config)
        self._running = False
        self._dispatch_task: asyncio.Task[None] | None = None
        self._quarantined_backend_task: asyncio.Task[Any] | None = None

    async def run(self, duration_s: float) -> tuple:
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if self._running:
            raise RuntimeError("serving engine is already running")
        self._running = True
        start_s = self.clock.now()
        end_s = start_s + duration_s
        next_tick = {
            session_id: start_s + 1.0 / endpoint.session_config.control_hz
            for session_id, endpoint in self.endpoints.items()
        }
        try:
            for session_id in self.endpoints:
                self._observe(session_id)
            while self.clock.now() - start_s < duration_s:
                self._maybe_dispatch()
                if self.clock.now() >= end_s:
                    break
                for session_id, due_s in tuple(next_tick.items()):
                    now_s = self.clock.now()
                    if now_s >= end_s:
                        break
                    if now_s >= due_s:
                        period_s = 1.0 / self.endpoints[
                            session_id
                        ].session_config.control_hz
                        missed = math.floor((now_s - due_s) / period_s)
                        if missed:
                            self._record_missed_ticks(session_id, missed)
                        next_tick[session_id] = due_s + (missed + 1) * period_s
                        self._control_tick(session_id)
                sleep_s = max(0.0005, min(next_tick.values()) - self.clock.now())
                await asyncio.sleep(min(sleep_s, 0.005))
            for session_id, due_s in next_tick.items():
                if due_s <= end_s:
                    period_s = 1.0 / self.endpoints[
                        session_id
                    ].session_config.control_hz
                    missed = math.floor((end_s - due_s) / period_s) + 1
                    self._record_missed_ticks(session_id, missed)
            if self._dispatch_task is not None:
                await self._dispatch_task
            return self.runtime.events.events
        finally:
            self._running = False

    def _observe(self, session_id: str) -> None:
        if self.runtime.has_pending_request(session_id):
            self.runtime.events.append(
                self.clock.now(),
                "observation_suppressed_backpressure",
                session_id,
            )
            return
        endpoint = self.endpoints[session_id]
        try:
            payload = endpoint.observe()
            validate_observation(payload, endpoint.observation_schema)
            self.runtime.observe(session_id, payload)
        except ObservationUnavailable:
            self.runtime.events.append(
                self.clock.now(), "endpoint_observation_unavailable", session_id
            )
        except Exception as error:
            self.runtime.events.append(
                self.clock.now(),
                "endpoint_observation_failed",
                session_id,
                error=str(error),
            )
            self._disconnect(session_id)

    def _control_tick(self, session_id: str) -> None:
        endpoint = self.endpoints[session_id]
        snapshot_before = next(
            session
            for session in self.runtime.snapshot().sessions
            if session.session_id == session_id
        )
        if not snapshot_before.connected:
            self._record_missed_ticks(session_id, 1)
            return
        command = self.runtime.dequeue_action(session_id)
        if command is None:
            try:
                outcome = endpoint.fallback()
                self.runtime.events.append(
                    self.clock.now(), "endpoint_fallback", session_id
                )
                self._handle_execution_outcome(session_id, outcome)
                self._observe(session_id)
            except Exception as error:
                self.runtime.events.append(
                    self.clock.now(),
                    "endpoint_fallback_failed",
                    session_id,
                    error=str(error),
                )
                self._disconnect(session_id)
            return
        try:
            outcome = endpoint.execute(command.value)
        except Exception as error:
            self.runtime.acknowledge(command, accepted=False)
            self.runtime.events.append(
                self.clock.now(),
                "endpoint_action_failed",
                session_id,
                error=str(error),
            )
            self._disconnect(session_id)
            return
        self.runtime.acknowledge(command, accepted=True)
        self._handle_execution_outcome(session_id, outcome)
        snapshot = next(
            session
            for session in self.runtime.snapshot().sessions
            if session.session_id == session_id
        )
        if (
            snapshot.buffer_horizon_s
            <= endpoint.session_config.request_threshold_s
        ):
            self._observe(session_id)

    def _record_missed_ticks(self, session_id: str, count: int) -> None:
        self.runtime.events.append(
            self.clock.now(), "control_ticks_missed", session_id, count=count
        )

    def _maybe_dispatch(self) -> None:
        if self._quarantined_backend_task is not None:
            if not self._quarantined_backend_task.done():
                return
            # Retrieve a quarantined worker's exception without using its stale result.
            try:
                self._quarantined_backend_task.result()
            except Exception:
                pass
            self._quarantined_backend_task = None
        if self._dispatch_task is not None and not self._dispatch_task.done():
            return
        if self._dispatch_task is not None:
            exception = self._dispatch_task.exception()
            self._dispatch_task = None
            if exception is not None:
                raise exception
        snapshot = self.runtime.snapshot()
        if not snapshot.ready_sessions:
            return
        costs = self.backend.cost_model
        decision = self.scheduler.schedule(snapshot, costs)
        batch = self.runtime.prepare_batch(decision)
        self.runtime.events.append(
            self.clock.now(),
            "scheduler_cost_estimate",
            batch_size=len(batch),
            base_latency_s=costs.base_latency_s,
            per_item_latency_s=costs.per_item_latency_s,
            estimated_latency_s=costs.estimate(len(batch)),
        )
        prepare_batch = getattr(self.backend, "prepare_batch", None)
        try:
            backend_request = prepare_batch(batch) if prepare_batch else batch
        except Exception as error:
            self._handle_batch_failure(batch, error, phase="prepare")
            return
        self._dispatch_task = asyncio.create_task(
            self._infer(batch, backend_request)
        )

    async def _infer(self, batch: tuple, backend_request: Any) -> None:
        started_at_s = self.clock.now()
        self.runtime.events.append(
            started_at_s, "inference_started", batch_size=len(batch)
        )
        infer_async = getattr(self.backend, "infer_async", None)
        try:
            if infer_async is not None:
                worker = asyncio.create_task(
                    infer_async(backend_request, started_at_s)
                )
            else:
                infer = getattr(
                    self.backend, "infer_prepared", self.backend.infer
                )
                worker = asyncio.create_task(
                    asyncio.to_thread(infer, backend_request, started_at_s)
                )
            if self.inference_timeout_s is None:
                result = await worker
            else:
                try:
                    result = await asyncio.wait_for(
                        asyncio.shield(worker), timeout=self.inference_timeout_s
                    )
                except TimeoutError:
                    self._quarantined_backend_task = worker
                    raise
            self._validate_backend_result(batch, result)
        except Exception as error:
            self._handle_batch_failure(batch, error, phase="inference")
            return
        latency_s = self.clock.now() - started_at_s
        self.runtime.events.append(
            self.clock.now(),
            "inference_completed",
            batch_size=len(result.chunks),
            latency_s=latency_s,
            backend_reported_latency_s=result.latency_s,
            output_shapes=tuple(
                chunk.auxiliary.get("output_shape")
                for chunk in result.chunks
                if "output_shape" in chunk.auxiliary
            ),
            execution_horizons=tuple(
                chunk.auxiliary.get("execution_horizon")
                for chunk in result.chunks
                if "execution_horizon" in chunk.auxiliary
            ),
        )
        deliveries = [
            self._deliver(
                chunk,
                self.endpoints[chunk.session_id].session_config.network_latency_s,
            )
            for chunk in result.chunks
        ]
        await asyncio.gather(*deliveries)

    async def _deliver(self, chunk: Any, delay_s: float) -> None:
        if delay_s:
            await asyncio.sleep(delay_s)
        accepted = self.runtime.accept(chunk)
        commit_chunk = getattr(self.backend, "commit_chunk", None)
        if commit_chunk is not None:
            commit_chunk(chunk, accepted)

    def _disconnect(self, session_id: str) -> None:
        snapshot = next(
            session
            for session in self.runtime.snapshot().sessions
            if session.session_id == session_id
        )
        if snapshot.connected:
            self.runtime.disconnect(session_id)
        reset_session = getattr(self.backend, "reset_session", None)
        if reset_session is not None:
            reset_session(session_id)
        try:
            self.endpoints[session_id].close()
        except Exception as error:
            self.runtime.events.append(
                self.clock.now(),
                "endpoint_close_failed",
                session_id,
                error=str(error),
            )

    def reset_session(self, session_id: str) -> None:
        self.runtime.reset(session_id)
        reset_session = getattr(self.backend, "reset_session", None)
        if reset_session is not None:
            reset_session(session_id)

    def reconnect_session(self, session_id: str) -> None:
        endpoint = self.endpoints[session_id]
        endpoint.reconnect()
        self.runtime.reconnect(session_id)
        reset_session = getattr(self.backend, "reset_session", None)
        if reset_session is not None:
            reset_session(session_id)
        self._observe(session_id)

    def _handle_execution_outcome(
        self, session_id: str, outcome: ExecutionOutcome | None
    ) -> None:
        if outcome is not None and outcome.task_reward is not None:
            self.runtime.events.append(
                self.clock.now(),
                "endpoint_task_step",
                session_id,
                reward=outcome.task_reward,
                success=outcome.task_success,
                terminated=outcome.terminated,
                truncated=outcome.truncated,
            )
        if outcome is not None and outcome.episode_boundary:
            self.runtime.events.append(
                self.clock.now(), "endpoint_episode_boundary", session_id
            )
            self.reset_session(session_id)

    def _validate_backend_result(
        self, batch: tuple[Observation, ...], result: Any
    ) -> None:
        if not isinstance(result, BackendResult):
            raise TypeError("backend must return BackendResult")
        if len(result.chunks) != len(batch):
            raise ValueError(
                "backend result must contain exactly one chunk per observation"
            )
        expected = {
            (observation.session_id, observation.generation, observation.sequence)
            for observation in batch
        }
        actual = set()
        for chunk in result.chunks:
            if not isinstance(chunk, ActionChunk):
                raise TypeError("backend result contains a non-ActionChunk value")
            key = (
                chunk.session_id,
                chunk.generation,
                chunk.observation_sequence,
            )
            if key in actual:
                raise ValueError("backend result contains a duplicate chunk")
            actual.add(key)
        if actual != expected:
            raise ValueError(
                "backend chunks do not match dispatched session/generation/sequence"
            )

    def _handle_batch_failure(
        self, batch: tuple, error: Exception, *, phase: str
    ) -> None:
        self.runtime.events.append(
            self.clock.now(),
            "inference_failed",
            error=str(error),
            session_ids=tuple(observation.session_id for observation in batch),
            phase=phase,
        )
        for observation in batch:
            session_id = observation.session_id
            self.reset_session(session_id)
            try:
                outcome = self.endpoints[session_id].fallback()
                self.runtime.events.append(
                    self.clock.now(), "endpoint_fallback", session_id
                )
                self._handle_execution_outcome(session_id, outcome)
            except Exception as fallback_error:
                self.runtime.events.append(
                    self.clock.now(),
                    "endpoint_fallback_failed",
                    session_id,
                    error=str(fallback_error),
                )
                self._disconnect(session_id)

    def close(self) -> None:
        for session_id in self.endpoints:
            self._disconnect(session_id)


def serving_metrics(events: tuple, endpoints: list[Endpoint], duration_s: float):
    """Compute the same system metrics used by virtual-time benchmarks."""

    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("metrics duration must be finite and positive")

    from .benchmark import compute_metrics
    from .simulation import RobotSpec, SimulationResult

    robots = [
        RobotSpec(
            session_id=endpoint.session_config.session_id,
            control_hz=endpoint.session_config.control_hz,
            chunk_size=endpoint.session_config.chunk_size,
            request_threshold_s=endpoint.session_config.request_threshold_s,
            network_latency_s=endpoint.session_config.network_latency_s,
            latency_budget_s=endpoint.session_config.latency_budget_s,
        )
        for endpoint in endpoints
    ]
    return compute_metrics(SimulationResult(duration_s, events), robots)
