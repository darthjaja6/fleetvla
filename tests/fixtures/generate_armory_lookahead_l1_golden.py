"""Print the pinned Armory L=1 fixtures used by FleetVLA's differential test."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ARMORY_COMMIT = "e876202ede99723f4be40d8d7cab31847bbd14a9"
SCENARIOS = (
    {
        "name": "latest-indexed-prefix",
        "latency_profile_s": [0.2],
        "max_batch_size": 1,
        "sessions": [
            {
                "session_id": "buffered",
                "control_hz": 20.0,
                "buffer_steps": 10,
                "chunk_size": 10,
                "service_weight": 1.0,
            },
            {
                "session_id": "empty",
                "control_hz": 20.0,
                "buffer_steps": 0,
                "chunk_size": 8,
                "service_weight": 0.9,
            },
        ],
    },
    {
        "name": "priority-tiers-batch-three",
        "latency_profile_s": [0.073, 0.1098, 0.1424],
        "max_batch_size": 3,
        "sessions": [
            {
                "session_id": "priority-a",
                "control_hz": 20.0,
                "buffer_steps": 0,
                "chunk_size": 6,
                "service_weight": 2.0,
            },
            {
                "session_id": "priority-b",
                "control_hz": 20.0,
                "buffer_steps": 0,
                "chunk_size": 6,
                "service_weight": 2.0,
            },
            {
                "session_id": "regular-a",
                "control_hz": 20.0,
                "buffer_steps": 0,
                "chunk_size": 10,
                "service_weight": 1.0,
            },
            {
                "session_id": "regular-b",
                "control_hz": 20.0,
                "buffer_steps": 0,
                "chunk_size": 10,
                "service_weight": 1.0,
            },
        ],
    },
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("armory_checkout", type=Path)
    args = parser.parse_args()
    checkout = args.armory_checkout.resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != ARMORY_COMMIT:
        raise SystemExit(f"expected Armory {ARMORY_COMMIT}, got {commit}")
    sys.path[:0] = [
        str(checkout / "src"),
        str(checkout / "armory-client" / "src"),
    ]

    import armory.scheduling.mirror as mirror_module
    from armory.scheduling.latency import LatencyTracker
    from armory.scheduling.lookahead_actions import HORIZON, IncrementalSearch
    from armory.scheduling.mirror import Mirror
    from armory.serving.rtc import InferType
    from armory.serving.schemas import ActionChunk, SlotRequest

    class FixedLatency(LatencyTracker):
        def __init__(self, profile):
            super().__init__()
            self.profile = profile

        def _update_measurement(self, values, key, value):
            del values, key, value

        def observation_latency(self, robot_id):
            del robot_id
            return 0.0

        def infer_latency(self, batch_size):
            return self.profile[batch_size - 1]

        def action_latency(self, robot_id):
            del robot_id
            return 0.0

    def request(session):
        return SlotRequest(
            0,
            session["session_id"],
            0,
            0.0,
            0,
            0,
            0.0,
            0.0,
            0,
            session["chunk_size"],
            InferType.SYNC,
            None,
            None,
            session["control_hz"],
            session["service_weight"],
        )

    def evaluate(scenario):
        latency = FixedLatency(scenario["latency_profile_s"])
        mirror = Mirror(latency)
        for index, session in enumerate(scenario["sessions"]):
            mirror.receive_request(request(session))
            if session["buffer_steps"]:
                mirror.robots[session["session_id"]].queue_chunk(
                    ActionChunk(
                        100 + index,
                        0,
                        0,
                        0,
                        session["buffer_steps"],
                        0.0,
                        0,
                        0,
                        "confirmed",
                    )
                )
        search = IncrementalSearch(
            mirror,
            latency,
            max_depth=1,
            max_batch_size=scenario["max_batch_size"],
        )
        candidates = []
        for batch in search._candidate_batches(search.root_node):
            infer_s = latency.infer_latency(len(batch))
            node = search.root_node.get_twin()
            chunks = node.queue_batch(
                list(batch),
                1,
                origin="searched",
                fast_forward=False,
                dispatch_time=0.0,
            )
            node.fast_forward(infer_s)
            evaluated = node.get_twin()
            evaluated.fast_forward(HORIZON)
            executed = {
                session_id: round(
                    evaluated.robots[session_id].score
                    / evaluated.robots[session_id].control_hz,
                    10,
                )
                for session_id in batch
            }
            objective = (
                sum(
                    executed[session_id] * evaluated.robots[session_id].weight
                    for session_id in batch
                )
                / infer_s
            )
            candidates.append(
                {
                    "batch": list(batch),
                    "first_executed_indices": {
                        session_id: chunk.first_executed_index
                        for session_id, chunk in zip(batch, chunks)
                    },
                    "executed_time_s": executed,
                    "objective": round(objective, 10),
                }
            )
        while not search.is_done():
            search.step(32)
        return {
            **scenario,
            "evaluation_horizon_s": HORIZON,
            "upstream_output": {
                "candidates": candidates,
                "selected_batch": list(search.best()[0]),
            },
        }

    mirror_module.time.time = lambda: 0.0
    output = {
        "source": {
            "repository": "https://github.com/GaTech-RL2/armory",
            "commit": ARMORY_COMMIT,
            "implementation": (
                "src/armory/scheduling/lookahead_actions.py:"
                "IncrementalSearch(max_depth=1)"
            ),
        },
        "scenarios": [evaluate(scenario) for scenario in SCENARIOS],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
