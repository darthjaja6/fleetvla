"""Wall-clock serving loop using the same runtime and scheduler contract."""

from __future__ import annotations

import asyncio
import math
from typing import Any

from .backend import BackendResult
from .clock import MonotonicClock
from .endpoints import (
    Endpoint,
    ExecutionOutcome,
    ObservationUnavailable,
    validate_observation,
)
from .remote import RemoteActionReceipt
from .runtime import FleetRuntime
from .scheduler_execution import SchedulerExecutionError, SchedulerRunner
from .schedulers import EDFScheduler, Scheduler
from .types import (
    ActionChunk,
    FleetSnapshot,
    InferenceCostModel,
    Observation,
    ScheduleDecision,
)


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
        scheduler_timeout_s: float = 0.01,
        endpoint_timeout_s: float = 0.1,
        action_execution: str = "sequential-buffer",
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
        if not math.isfinite(scheduler_timeout_s) or scheduler_timeout_s <= 0:
            raise ValueError("scheduler_timeout_s must be positive")
        self.scheduler_timeout_s = scheduler_timeout_s
        if not math.isfinite(endpoint_timeout_s) or endpoint_timeout_s <= 0:
            raise ValueError("endpoint_timeout_s must be positive")
        self.endpoint_timeout_s = endpoint_timeout_s
        self._scheduler_runner: SchedulerRunner | None = None
        self._fallback_scheduler = EDFScheduler()
        self._using_scheduler_fallback = False
        self.clock = MonotonicClock()
        self.runtime = FleetRuntime(
            self.clock,
            max_batch_size=max_batch_size,
            action_execution=action_execution,
        )
        for endpoint in endpoints:
            self.runtime.register(endpoint.session_config)
        self._running = False
        self._dispatch_task: asyncio.Task[None] | None = None
        self._scheduler_task: asyncio.Task[None] | None = None
        self._quarantined_backend_task: asyncio.Task[Any] | None = None
        self._dispatch_not_before_s: float | None = None
        self._deferred_ready: tuple[tuple[str, int | None], ...] | None = None
        self._dispatch_deadline_s: float | None = None
        self._endpoint_tasks: dict[str, asyncio.Task[None]] = {}
        self._background_endpoint_tasks: set[asyncio.Task[None]] = set()
        self._quarantined_endpoint_tasks: set[asyncio.Task[Any]] = set()
        self._endpoint_locks = {
            session_id: asyncio.Lock() for session_id in self.endpoints
        }

    async def run(self, duration_s: float) -> tuple:
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if self._running:
            raise RuntimeError("serving engine is already running")
        self._running = True
        self._scheduler_runner = SchedulerRunner(self.scheduler)
        self._using_scheduler_fallback = False
        start_s = self.clock.now()
        end_s = start_s + duration_s
        self._dispatch_deadline_s = end_s
        next_tick = {
            session_id: start_s + 1.0 / endpoint.session_config.control_hz
            for session_id, endpoint in self.endpoints.items()
        }
        try:
            for session_id in self.endpoints:
                self._start_endpoint_task(session_id, self._observe(session_id))
            while self.clock.now() - start_s < duration_s:
                self._harvest_endpoint_tasks()
                self._maybe_dispatch()
                if self.clock.now() >= end_s:
                    break
                for session_id, due_s in tuple(next_tick.items()):
                    now_s = self.clock.now()
                    if now_s >= end_s:
                        break
                    if now_s >= due_s:
                        period_s = (
                            1.0 / self.endpoints[session_id].session_config.control_hz
                        )
                        missed = math.floor((now_s - due_s) / period_s)
                        if missed:
                            self._record_missed_ticks(session_id, missed)
                        next_tick[session_id] = due_s + (missed + 1) * period_s
                        self._start_control_tick(session_id)
                wake_s = min(next_tick.values())
                if self._dispatch_not_before_s is not None:
                    wake_s = min(wake_s, self._dispatch_not_before_s)
                sleep_s = max(0.0005, wake_s - self.clock.now())
                await asyncio.sleep(min(sleep_s, 0.005))
            for session_id, due_s in next_tick.items():
                if due_s <= end_s:
                    period_s = (
                        1.0 / self.endpoints[session_id].session_config.control_hz
                    )
                    missed = math.floor((end_s - due_s) / period_s) + 1
                    self._record_missed_ticks(session_id, missed)
            if self._scheduler_task is not None:
                await self._scheduler_task
            if self._dispatch_task is not None:
                await self._dispatch_task
            await self._drain_endpoint_tasks()
            return self.runtime.events.events
        finally:
            if self._scheduler_runner is not None:
                self._scheduler_runner.close()
                self._scheduler_runner = None
            self._dispatch_deadline_s = None
            self._running = False

    async def _observe(self, session_id: str) -> None:
        if self.runtime.has_pending_request(session_id):
            self.runtime.events.append(
                self.clock.now(),
                "observation_suppressed_backpressure",
                session_id,
            )
            return
        endpoint = self.endpoints[session_id]
        try:
            payload = await self._call_endpoint(session_id, endpoint.observe)
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
            await self._disconnect(session_id)

    def _start_control_tick(self, session_id: str) -> None:
        task = self._endpoint_tasks.get(session_id)
        if task is not None and not task.done():
            self._record_missed_ticks(session_id, 1)
            return
        if task is not None:
            task.result()
        self._start_endpoint_task(session_id, self._control_tick(session_id))

    async def _control_tick(self, session_id: str) -> None:
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
                outcome = await self._call_endpoint(
                    session_id, endpoint.fallback
                )
                self.runtime.events.append(
                    self.clock.now(), "endpoint_fallback", session_id
                )
                self._handle_execution_outcome(session_id, outcome)
                await self._observe(session_id)
            except Exception as error:
                self.runtime.events.append(
                    self.clock.now(),
                    "endpoint_fallback_failed",
                    session_id,
                    error=str(error),
                )
                await self._disconnect(session_id)
            return
        try:
            self.runtime.events.append(
                self.clock.now(),
                "action_sent_endpoint",
                session_id,
                sequence=command.observation_sequence,
                action_index=command.action_index,
                deadline_s=command.deadline_s,
            )
            execute_command = getattr(endpoint, "execute_command", None)
            if execute_command is None:
                outcome = await self._call_endpoint(
                    session_id, endpoint.execute, command.value
                )
            else:
                receipt = await self._call_endpoint(
                    session_id, execute_command, command
                )
                if not isinstance(receipt, RemoteActionReceipt):
                    raise TypeError(
                        "command-aware endpoint must return RemoteActionReceipt"
                    )
                if receipt.accepted:
                    self._record_action_accepted(command)
                if not receipt.executed:
                    raise RuntimeError("remote robot rejected the action command")
                outcome = None
            if execute_command is None:
                self._record_action_accepted(command)
        except Exception as error:
            self.runtime.acknowledge(command, accepted=False)
            self.runtime.events.append(
                self.clock.now(),
                "endpoint_action_failed",
                session_id,
                error=str(error),
            )
            await self._disconnect(session_id)
            return
        if not self.runtime.acknowledge(command, accepted=True):
            return
        self._handle_execution_outcome(session_id, outcome)
        snapshot = next(
            session
            for session in self.runtime.snapshot().sessions
            if session.session_id == session_id
        )
        if snapshot.buffer_horizon_s <= endpoint.session_config.request_threshold_s:
            await self._observe(session_id)

    def _record_action_accepted(self, command: Any) -> None:
        self.runtime.events.append(
            self.clock.now(),
            "action_accepted_endpoint",
            command.session_id,
            sequence=command.observation_sequence,
            action_index=command.action_index,
        )

    async def _call_endpoint(self, session_id: str, method: Any, *args: Any) -> Any:
        async def invoke() -> Any:
            async with self._endpoint_locks[session_id]:
                if asyncio.iscoroutinefunction(method):
                    return await method(*args)
                return await asyncio.to_thread(method, *args)

        operation = asyncio.create_task(invoke())
        try:
            return await asyncio.wait_for(
                asyncio.shield(operation), timeout=self.endpoint_timeout_s
            )
        except TimeoutError:
            # A Python thread cannot be killed safely. Keep the operation and
            # its per-endpoint lock alive so a timed-out driver cannot overlap
            # a later close/reconnect, while other sessions continue.
            self._quarantined_endpoint_tasks.add(operation)
            raise

    def _start_endpoint_task(
        self, session_id: str, operation: Any
    ) -> asyncio.Task[None]:
        task = asyncio.create_task(operation)
        self._endpoint_tasks[session_id] = task
        return task

    def _start_background_endpoint_task(self, operation: Any) -> None:
        task = asyncio.create_task(operation)
        self._background_endpoint_tasks.add(task)

    def _harvest_endpoint_tasks(self) -> None:
        for session_id, task in tuple(self._endpoint_tasks.items()):
            if task.done():
                task.result()
                del self._endpoint_tasks[session_id]
        for task in tuple(self._background_endpoint_tasks):
            if task.done():
                task.result()
                self._background_endpoint_tasks.remove(task)
        for task in tuple(self._quarantined_endpoint_tasks):
            if task.done():
                try:
                    task.result()
                except Exception:
                    pass
                self._quarantined_endpoint_tasks.remove(task)

    async def _drain_endpoint_tasks(self) -> None:
        tasks = tuple(self._endpoint_tasks.values()) + tuple(
            self._background_endpoint_tasks
        )
        if tasks:
            await asyncio.gather(*tasks)
        self._endpoint_tasks.clear()
        self._background_endpoint_tasks.clear()

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
        if self._scheduler_task is not None and not self._scheduler_task.done():
            return
        if self._scheduler_task is not None:
            exception = self._scheduler_task.exception()
            self._scheduler_task = None
            if exception is not None:
                raise exception
        snapshot = self.runtime.snapshot()
        if not snapshot.ready_sessions:
            self._dispatch_not_before_s = None
            self._deferred_ready = None
            return
        ready = tuple(
            (session.session_id, session.ready_sequence)
            for session in snapshot.ready_sessions
        )
        if (
            self._dispatch_not_before_s is not None
            and self.clock.now() < self._dispatch_not_before_s
            and ready == self._deferred_ready
        ):
            return
        costs = self.backend.cost_model
        if self._using_scheduler_fallback:
            decision = self._fallback_scheduler.schedule(snapshot, costs)
            self._apply_decision(snapshot, costs, decision, 0.0, fallback=True)
            return
        self._scheduler_task = asyncio.create_task(
            self._schedule_and_dispatch(snapshot, costs)
        )

    async def _schedule_and_dispatch(
        self, snapshot: FleetSnapshot, costs: InferenceCostModel
    ) -> None:
        fallback = False
        try:
            if self._scheduler_runner is None:
                raise RuntimeError("scheduler runner is not active")
            result = await self._scheduler_runner.decide(
                snapshot, costs, self.scheduler_timeout_s
            )
            decision = result.decision
            latency_s = result.latency_s
        except SchedulerExecutionError as error:
            if (
                self._dispatch_deadline_s is None
                or self.clock.now() >= self._dispatch_deadline_s
            ):
                return
            fallback = True
            self._using_scheduler_fallback = True
            latency_s = self.scheduler_timeout_s
            self.runtime.events.append(
                self.clock.now(),
                "scheduler_failed",
                error=str(error),
                fallback="edf",
            )
            decision = self._fallback_scheduler.schedule(snapshot, costs)
        if (
            self._dispatch_deadline_s is None
            or self.clock.now() >= self._dispatch_deadline_s
            or self._ready_identity(self.runtime.snapshot())
            != self._ready_identity(snapshot)
        ):
            return
        try:
            self._apply_decision(
                snapshot, costs, decision, latency_s, fallback=fallback
            )
        except Exception as error:
            if fallback:
                raise
            self._using_scheduler_fallback = True
            self.runtime.events.append(
                self.clock.now(),
                "scheduler_failed",
                error=f"invalid decision: {error}",
                fallback="edf",
            )
            fallback_decision = self._fallback_scheduler.schedule(snapshot, costs)
            self._apply_decision(
                snapshot, costs, fallback_decision, latency_s, fallback=True
            )

    @staticmethod
    def _ready_identity(
        snapshot: FleetSnapshot,
    ) -> tuple[tuple[str, int, int | None], ...]:
        return tuple(
            (session.session_id, session.generation, session.ready_sequence)
            for session in snapshot.ready_sessions
        )

    def _apply_decision(
        self,
        snapshot: FleetSnapshot,
        costs: InferenceCostModel,
        decision: ScheduleDecision,
        latency_s: float,
        *,
        fallback: bool,
    ) -> None:
        self._validate_decision(snapshot, decision)
        ready = tuple(
            (session.session_id, session.ready_sequence)
            for session in snapshot.ready_sessions
        )
        self.runtime.events.append(
            self.clock.now(),
            "scheduler_decision",
            latency_s=latency_s,
            selected_session_ids=decision.session_ids,
            deferred=decision.defer_until_s is not None,
            fallback=fallback,
        )
        if decision.defer_until_s is not None:
            self._dispatch_not_before_s = decision.defer_until_s
            self._deferred_ready = ready
            self.runtime.events.append(
                self.clock.now(),
                "dispatch_deferred",
                defer_until_s=decision.defer_until_s,
                reason=decision.reason,
            )
            return
        self._dispatch_not_before_s = None
        self._deferred_ready = None
        batch = self.runtime.prepare_batch(decision)
        self.runtime.events.append(
            self.clock.now(),
            "scheduler_cost_estimate",
            batch_size=len(batch),
            base_latency_s=costs.base_latency_s,
            per_item_latency_s=costs.per_item_latency_s,
            batch_latency_s=costs.batch_latency_s,
            estimated_latency_s=costs.estimate(len(batch)),
        )
        prepare_batch = getattr(self.backend, "prepare_batch", None)
        try:
            backend_request = prepare_batch(batch) if prepare_batch else batch
        except Exception as error:
            self._handle_batch_failure(batch, error, phase="prepare")
            return
        self._dispatch_task = asyncio.create_task(self._infer(batch, backend_request))

    def _validate_decision(
        self, snapshot: FleetSnapshot, decision: ScheduleDecision
    ) -> None:
        if not isinstance(decision, ScheduleDecision):
            raise TypeError("scheduler must return a ScheduleDecision")
        if decision.defer_until_s is not None:
            if decision.defer_until_s <= self.clock.now():
                raise ValueError("scheduler deferral must be in the future")
            return
        if not decision.session_ids:
            raise ValueError("scheduler returned an empty batch")
        if len(decision.session_ids) > snapshot.max_batch_size:
            raise ValueError("scheduler exceeded max_batch_size")
        ready = {session.session_id for session in snapshot.ready_sessions}
        if not set(decision.session_ids) <= ready:
            raise ValueError("scheduler selected a session that is not ready")

    async def _infer(self, batch: tuple, backend_request: Any) -> None:
        started_at_s = self.clock.now()
        self.runtime.events.append(
            started_at_s, "inference_started", batch_size=len(batch)
        )
        infer_async = getattr(self.backend, "infer_async", None)
        try:
            if infer_async is not None:
                worker = asyncio.create_task(infer_async(backend_request, started_at_s))
            else:
                infer = getattr(self.backend, "infer_prepared", self.backend.infer)
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

    async def _disconnect(self, session_id: str) -> None:
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
            await self._call_endpoint(
                session_id, self.endpoints[session_id].close
            )
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
            self._start_background_endpoint_task(
                self._fallback_after_failure(session_id)
            )

    async def _fallback_after_failure(self, session_id: str) -> None:
        try:
            outcome = await self._call_endpoint(
                session_id, self.endpoints[session_id].fallback
            )
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
            await self._disconnect(session_id)

    def close(self) -> None:
        if self._scheduler_runner is not None:
            self._scheduler_runner.close()
            self._scheduler_runner = None
        for session_id, endpoint in self.endpoints.items():
            snapshot = next(
                session
                for session in self.runtime.snapshot().sessions
                if session.session_id == session_id
            )
            if snapshot.connected:
                self.runtime.disconnect(session_id)
            endpoint.close()


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
            service_weight=endpoint.session_config.service_weight,
        )
        for endpoint in endpoints
    ]
    return compute_metrics(SimulationResult(duration_s, events), robots)
