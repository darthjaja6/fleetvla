"""Minimal external scheduler used by the contributor guide."""

from fleetvla import FleetSnapshot, InferenceCostModel, ScheduleDecision


class SmallestBufferFirst:
    def schedule(
        self, fleet: FleetSnapshot, costs: InferenceCostModel
    ) -> ScheduleDecision:
        del costs
        ready = sorted(
            fleet.ready_sessions,
            key=lambda session: (session.buffer_horizon_s, session.session_id),
        )
        selected = tuple(
            session.session_id for session in ready[: fleet.max_batch_size]
        )
        return ScheduleDecision(selected, "smallest action buffer first")
