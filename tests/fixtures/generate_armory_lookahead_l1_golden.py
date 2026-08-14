"""Print the pinned Armory L=1 fixture used by FleetVLA's differential test."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ARMORY_COMMIT = "e876202ede99723f4be40d8d7cab31847bbd14a9"


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
        def _update_measurement(self, values, key, value):
            del values, key, value

        def observation_latency(self, robot_id):
            del robot_id
            return 0.0

        def infer_latency(self, batch_size):
            del batch_size
            return 0.2

        def action_latency(self, robot_id):
            del robot_id
            return 0.0

    def request(session_id, chunk_size, weight):
        return SlotRequest(
            0,
            session_id,
            0,
            0.0,
            0,
            0,
            0.0,
            0.0,
            0,
            chunk_size,
            InferType.SYNC,
            None,
            None,
            20.0,
            weight,
        )

    mirror_module.time.time = lambda: 0.0
    latency = FixedLatency()
    mirror = Mirror(latency)
    mirror.receive_request(request("buffered", 10, 1.0))
    mirror.receive_request(request("empty", 8, 0.9))
    mirror.robots["buffered"].queue_chunk(
        ActionChunk(100, 0, 0, 0, 10, 0.0, 0, 0, "confirmed")
    )
    search = IncrementalSearch(mirror, latency, max_depth=1, max_batch_size=1)
    candidates = []
    for batch in search._candidate_batches(search.root_node):
        node = search.root_node.get_twin()
        chunk = node.queue_batch(
            list(batch),
            1,
            origin="searched",
            fast_forward=False,
            dispatch_time=0.0,
        )[0]
        node.fast_forward(0.2)
        evaluated = node.get_twin()
        evaluated.fast_forward(HORIZON)
        session_id = batch[0]
        robot = evaluated.robots[session_id]
        executed_time_s = robot.score / robot.control_hz
        candidates.append(
            {
                "session_id": session_id,
                "first_executed_index": chunk.first_executed_index,
                "executed_time_s": round(executed_time_s, 10),
                "objective": round(executed_time_s * robot.weight / 0.2, 10),
            }
        )
    while not search.is_done():
        search.step(32)
    output = {
        "source": {
            "repository": "https://github.com/GaTech-RL2/armory",
            "commit": ARMORY_COMMIT,
            "implementation": (
                "src/armory/scheduling/lookahead_actions.py:"
                "IncrementalSearch(max_depth=1)"
            ),
        },
        "scenario": {
            "evaluation_horizon_s": HORIZON,
            "inference_latency_s": 0.2,
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
        "upstream_output": {
            "candidates": sorted(candidates, key=lambda item: item["session_id"]),
            "selected_batch": list(search.best()[0]),
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
