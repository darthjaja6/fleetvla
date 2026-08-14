import asyncio
import math

import pytest

from fleetvla import (
    FieldSpec,
    FIFOScheduler,
    Observation,
    SessionConfig,
    SyntheticBackend,
)
from fleetvla.endpoints import (
    LeRobotRobotEndpoint,
    ObservationUnavailable,
    ROS2Endpoint,
)
from fleetvla.integrations import (
    LeRobotPolicyBackend,
    LiberoVectorAdapter,
    PolicyPrediction,
    StatefulPolicyBackend,
)
from fleetvla.serving import AsyncServingEngine


class FakeRobot:
    def __init__(self):
        self.actions = []
        self.disconnected = False
        self.connected = False

    def get_observation(self):
        return {"state": ShapeValue((2,))}

    def send_action(self, action):
        self.actions.append(action)

    def disconnect(self):
        self.disconnected = True
        self.connected = False

    def connect(self):
        self.connected = True


class ShapeValue:
    def __init__(self, shape):
        self.shape = shape


def test_lerobot_robot_endpoint_owns_fallback_and_action_validation() -> None:
    robot = FakeRobot()
    endpoint = LeRobotRobotEndpoint(
        robot,
        SessionConfig("arm", control_hz=10, chunk_size=2),
        observation_schema=(FieldSpec("state", (2,)),),
        action_converter=lambda value: {"joint": value},
        safe_action=lambda: {"joint": 0.0},
    )

    assert endpoint.observe()["state"].shape == (2,)
    endpoint.execute(0.5)
    with pytest.raises(ValueError, match="safety"):
        endpoint.execute(math.nan)
    endpoint.close()

    assert robot.actions == [
        {"joint": 0.5},
        {"joint": 0.0},
        {"joint": 0.0},
    ]
    assert robot.disconnected
    endpoint.reconnect()
    assert robot.connected


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class FakeNode:
    def __init__(self):
        self.publisher = FakePublisher()
        self.callback = None

    def create_publisher(self, message_type, topic, qos):
        return self.publisher

    def create_subscription(self, message_type, topic, callback, qos):
        self.callback = callback
        return object()


def test_ros2_bridge_uses_topic_conversion_and_safe_fallback() -> None:
    node = FakeNode()
    endpoint = ROS2Endpoint(
        node,
        SessionConfig("arm", control_hz=10, chunk_size=1),
        observation_topic="/state",
        observation_message_type=dict,
        observation_converter=lambda message: message,
        action_topic="/command",
        action_message_type=dict,
        action_converter=lambda action: {"command": action},
        fallback_message=lambda: {"command": 0},
        observation_schema=(FieldSpec("state", (1,)),),
    )
    with pytest.raises(RuntimeError, match="no ROS 2 observation"):
        endpoint.observe()
    node.callback({"state": ShapeValue((1,))})
    endpoint.execute(2)
    endpoint.close()

    assert node.publisher.messages == [{"command": 2}, {"command": 0}]


def test_ros2_bridge_rejects_unsafe_or_closed_actions_and_waits_after_reconnect() -> (
    None
):
    node = FakeNode()
    endpoint = ROS2Endpoint(
        node,
        SessionConfig("arm", control_hz=10, chunk_size=1),
        observation_topic="/state",
        observation_message_type=dict,
        observation_converter=lambda message: message,
        action_topic="/command",
        action_message_type=dict,
        action_converter=lambda action: {"joint": action},
        fallback_message=lambda: {"joint": 0},
        observation_schema=(FieldSpec("state", (1,)),),
    )
    node.callback({"state": [1]})
    assert endpoint.observe() == {"state": [1]}
    with pytest.raises(ValueError, match="safety"):
        endpoint.execute(math.nan)
    endpoint.close()
    node.callback({"state": [999]})
    with pytest.raises(RuntimeError, match="closed"):
        endpoint.execute(7)

    endpoint.reconnect()
    with pytest.raises(ObservationUnavailable):
        endpoint.observe()
    node.callback({"state": [2]})
    assert endpoint.observe() == {"state": [2]}
    assert node.publisher.messages == [{"joint": 0}, {"joint": 0}]


def test_ros2_validates_converted_command_and_fallback_message() -> None:
    node = FakeNode()
    endpoint = ROS2Endpoint(
        node,
        SessionConfig("arm", control_hz=10, chunk_size=1),
        observation_topic="/state",
        observation_message_type=dict,
        observation_converter=lambda message: message,
        action_topic="/command",
        action_message_type=dict,
        action_converter=lambda action: {"joint": math.nan},
        fallback_message=lambda: {"joint": math.nan},
    )

    with pytest.raises(ValueError, match="fallback"):
        endpoint.execute(1.0)
    with pytest.raises(ValueError, match="fallback"):
        endpoint.fallback()
    assert node.publisher.messages == []


def test_ros2_default_validator_supports_generated_message_fields() -> None:
    class Message:
        __slots__ = ("position",)

        def __init__(self, position):
            self.position = position

        @classmethod
        def get_fields_and_field_types(cls):
            return {"position": "double"}

    node = FakeNode()
    endpoint = ROS2Endpoint(
        node,
        SessionConfig("arm", control_hz=10, chunk_size=1),
        observation_topic="/state",
        observation_message_type=Message,
        observation_converter=lambda message: {"state": [message.position]},
        action_topic="/command",
        action_message_type=Message,
        action_converter=Message,
        fallback_message=lambda: Message(0.0),
    )

    endpoint.execute(1.0)
    with pytest.raises(ValueError, match="safety"):
        endpoint.execute(math.nan)

    assert [message.position for message in node.publisher.messages] == [1.0, 0.0]


@pytest.mark.parametrize("value", [7, [1], [1, 2, 3], "unsafe"])
def test_shape_schema_fails_closed_for_wrong_python_values(value) -> None:
    with pytest.raises(ValueError):
        FieldSpec("state", (2,)).validate({"state": value})


def test_lerobot_policy_backend_dynamic_batches_action_chunks() -> None:
    torch = pytest.importorskip("torch")

    class Policy:
        def __init__(self):
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

        def predict_action_chunk(self, batch):
            assert not torch.is_grad_enabled()
            assert batch["state"].shape == (2, 2)
            return torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=torch.float32)

    policy = Policy()
    backend = LeRobotPolicyBackend(policy)
    observations = (
        Observation("a", 0, 0, 0, {"state": torch.tensor([[0.0, 1.0]])}),
        Observation("b", 0, 0, 0, {"state": torch.tensor([[2.0, 3.0]])}),
    )

    result = backend.infer(observations, 10.0)

    assert [chunk.actions for chunk in result.chunks] == [
        ([1.0], [2.0]),
        ([3.0], [4.0]),
    ]
    assert result.chunks[0].auxiliary["output_shape"] == (2, 2, 1)
    assert policy.reset_calls == 1

    backend.reset_session("a")
    backend.infer(observations, 11.0)
    assert policy.reset_calls == 2


def test_lerobot_policy_backend_preserves_shared_processor_metadata() -> None:
    torch = pytest.importorskip("torch")

    class Policy:
        def reset(self):
            pass

        def predict_action_chunk(self, batch):
            assert batch["action"] is None
            assert batch["next.reward"] == 0.0
            assert batch["info"] == {}
            return torch.ones((2, 1, 1))

    backend = LeRobotPolicyBackend(
        Policy(),
        preprocessor=lambda item: {
            **item,
            "action": None,
            "next.reward": 0.0,
            "info": {},
        },
    )
    observations = (
        Observation("a", 0, 0, 0, {"state": torch.zeros((1, 1))}),
        Observation("b", 0, 0, 0, {"state": torch.ones((1, 1))}),
    )

    result = backend.infer(observations, 0)

    assert len(result.chunks) == 2


def test_lerobot_policy_backend_preprocesses_the_complete_batch_once() -> None:
    torch = pytest.importorskip("torch")
    calls = []

    def preprocess(batch):
        calls.append(tuple(batch["task"]))
        batch["tokens"] = torch.ones((2, 4))
        return batch

    class Policy:
        def reset(self):
            pass

        def predict_action_chunk(self, batch):
            assert batch["tokens"].shape == (2, 4)
            return torch.ones((2, 1, 1))

    backend = LeRobotPolicyBackend(Policy(), preprocessor=preprocess)
    backend.infer(
        (
            Observation("a", 0, 0, 0, {"task": ["short"]}),
            Observation("b", 0, 0, 0, {"task": ["a longer task"]}),
        ),
        0,
    )

    assert calls == [("short", "a longer task")]


def test_lerobot_policy_backend_updates_its_cost_from_measured_latency(
    monkeypatch,
) -> None:
    torch = pytest.importorskip("torch")
    times = iter((10.0, 12.0))
    monkeypatch.setattr(
        "fleetvla.integrations.lerobot.time.perf_counter", lambda: next(times)
    )

    class Policy:
        def reset(self):
            pass

        def predict_action_chunk(self, batch):
            return torch.ones((2, 1, 1))

    backend = LeRobotPolicyBackend(
        Policy(),
        predicted_base_latency_s=0,
        predicted_per_item_latency_s=0,
        cost_update_alpha=1,
    )
    backend.infer(
        (
            Observation("a", 0, 0, 0, {"state": torch.zeros((1, 1))}),
            Observation("b", 0, 0, 0, {"state": torch.zeros((1, 1))}),
        ),
        0,
    )

    assert backend.cost_model.estimate(2) == 2.0


def test_lerobot_policy_backend_separates_output_and_execution_horizons() -> None:
    torch = pytest.importorskip("torch")

    class Config:
        n_action_steps = 2

    class Policy:
        config = Config()

        def reset(self):
            pass

        def predict_action_chunk(self, batch):
            return torch.arange(5, dtype=torch.float32).reshape(1, 5, 1)

    backend = LeRobotPolicyBackend(Policy())
    result = backend.infer(
        (Observation("a", 0, 0, 0, {"state": torch.zeros((1, 1))}),), 0
    )

    assert result.chunks[0].actions == ([0.0], [1.0])
    assert result.chunks[0].auxiliary["output_shape"] == (1, 5, 1)
    assert result.chunks[0].auxiliary["execution_horizon"] == 2


def test_lerobot_policy_backend_requires_resettable_policy() -> None:
    class Policy:
        def predict_action_chunk(self, batch):
            del batch

    with pytest.raises(TypeError, match="define reset"):
        LeRobotPolicyBackend(Policy())


def test_stateful_policy_state_is_session_local_and_resettable() -> None:
    def predict(payloads, states):
        return [
            PolicyPrediction((payload,), (state or 0) + 1, {"world_token": "opaque"})
            for payload, state in zip(payloads, states)
        ]

    backend = StatefulPolicyBackend(predict, base_latency_s=0, per_item_latency_s=0)
    first = backend.infer(
        (Observation("a", 0, 0, 0, 1), Observation("b", 0, 0, 0, 2)), 0
    )
    for chunk in first.chunks:
        backend.commit_chunk(chunk, True)
    second = backend.infer((Observation("a", 1, 0, 0, 3),), 0)
    backend.commit_chunk(second.chunks[0], True)

    assert backend.state_for_testing("a") == 2
    assert backend.state_for_testing("b") == 1
    backend.reset_session("a")
    assert backend.state_for_testing("a") is None


def test_stateful_reset_invalidates_in_flight_state_write() -> None:
    import threading

    started = threading.Event()
    release = threading.Event()

    def predict(payloads, states):
        started.set()
        assert release.wait(2)
        return [PolicyPrediction((1,), "stale-state")]

    backend = StatefulPolicyBackend(predict, base_latency_s=0, per_item_latency_s=0)
    worker = threading.Thread(
        target=backend.infer,
        args=((Observation("arm", 0, 0, 0, 1),), 0),
    )
    worker.start()
    assert started.wait(2)
    backend.reset_session("arm")
    release.set()
    worker.join(2)

    assert backend.state_for_testing("arm") is None


def test_stateful_request_bound_before_worker_entry_cannot_commit_after_reset() -> None:
    backend = StatefulPolicyBackend(
        lambda payloads, states: [PolicyPrediction((1,), "stale")],
        base_latency_s=0,
        per_item_latency_s=0,
    )
    prepared = backend.prepare_batch((Observation("arm", 0, 0, 0, 1),))
    backend.reset_session("arm")

    result = backend.infer_prepared(prepared, 0)
    backend.commit_chunk(result.chunks[0], True)

    assert backend.state_for_testing("arm") is None


class FakeVectorEnv:
    def __init__(self):
        self.values = [0, 0]

    def reset(self, id):
        self.values[id[0]] = 0
        return [{"state": [self.values[id[0]]]}]

    def step(self, action, id):
        index = id[0]
        self.values[index] += 1
        return (
            [{"state": [self.values[index]]}],
            [1.0],
            [self.values[index] >= 2],
            [{"success": self.values[index] >= 2}],
        )


def test_libero_vector_slots_are_endpoints_but_rewards_stay_in_adapter() -> None:
    adapter = LiberoVectorAdapter(
        FakeVectorEnv(),
        [
            SessionConfig("a", control_hz=10, chunk_size=1),
            SessionConfig("b", control_hz=10, chunk_size=1),
        ],
        observation_converter=lambda observation: observation,
        action_converter=lambda action: action,
        fallback_action=lambda: [0],
    )

    adapter.endpoints[0].execute([1])
    adapter.endpoints[0].execute([1])

    assert adapter.metrics["a"].reward == 2
    assert adapter.metrics["a"].successes == 1
    assert adapter.endpoints[0].observe() == {"state": [0]}
    assert adapter.metrics["a"].episodes == 1


def test_libero_single_addressing_supports_gym_vector_env() -> None:
    np = pytest.importorskip("numpy")

    class GymVectorEnv:
        def reset(self, seed=None):
            assert seed == 7
            return {"state": np.asarray([[0.0]])}, {}

        def step(self, action):
            assert action.shape == (1, 1)
            return (
                {"state": np.asarray([[1.0]])},
                np.asarray([1.0]),
                np.asarray([False]),
                np.asarray([False]),
                {"is_success": np.asarray([False])},
            )

    adapter = LiberoVectorAdapter(
        GymVectorEnv(),
        [SessionConfig("task", control_hz=20, chunk_size=1)],
        observation_converter=lambda observation: observation,
        action_converter=lambda action: action,
        fallback_action=lambda: [0],
        addressing="single",
        reset_kwargs={"seed": 7},
    )

    assert adapter.endpoints[0].observe()["state"].shape == (1, 1)
    adapter.endpoints[0].execute([1])
    assert adapter.metrics["task"].reward == 1


def test_libero_single_addressing_rejects_multiple_sessions() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        LiberoVectorAdapter(
            object(),
            [
                SessionConfig("a", control_hz=20, chunk_size=1),
                SessionConfig("b", control_hz=20, chunk_size=1),
            ],
            observation_converter=lambda observation: observation,
            action_converter=lambda action: action,
            fallback_action=lambda: [0],
            addressing="single",
        )


def test_libero_adapter_enforces_an_explicit_episode_limit() -> None:
    np = pytest.importorskip("numpy")

    class NeverDoneEnv:
        def reset(self):
            return {"state": np.asarray([[0.0]])}, {}

        def step(self, action):
            return (
                {"state": np.asarray([[1.0]])},
                np.asarray([0.0]),
                np.asarray([False]),
                np.asarray([False]),
                {"is_success": np.asarray([False])},
            )

    adapter = LiberoVectorAdapter(
        NeverDoneEnv(),
        [SessionConfig("task", control_hz=20, chunk_size=1)],
        observation_converter=lambda observation: observation,
        action_converter=lambda action: action,
        fallback_action=lambda: [0],
        addressing="single",
        max_episode_steps=2,
    )

    assert adapter.endpoints[0].execute([1]).episode_boundary is False
    assert adapter.endpoints[0].execute([1]).episode_boundary is True
    assert adapter.metrics["task"].episodes == 1
    assert adapter.metrics["task"].truncated == 1


class OneStepEpisodeVectorEnv:
    def __init__(self):
        self.episode = -1
        self.actions = []

    def reset(self, id):
        self.episode += 1
        return [{"state": [0]}]

    def step(self, action, id):
        self.actions.append((self.episode, action[0]))
        return ([{"state": [1]}], [1.0], [True], [{}])


def test_libero_episode_boundary_clears_remainder_of_action_chunk() -> None:
    environment = OneStepEpisodeVectorEnv()
    adapter = LiberoVectorAdapter(
        environment,
        [SessionConfig("task", control_hz=30, chunk_size=3)],
        observation_converter=lambda observation: observation,
        action_converter=lambda action: action,
        fallback_action=lambda: [0],
    )
    engine = AsyncServingEngine(
        adapter.endpoints,
        SyntheticBackend(chunk_size=3, base_latency_s=0, per_item_latency_s=0),
        FIFOScheduler(),
    )

    events = asyncio.run(engine.run(0.2))

    executed_indices = [
        event.details["action_index"]
        for event in events
        if event.kind == "action_executed" and event.details
    ]
    assert executed_indices
    assert set(executed_indices) == {0}
    assert adapter.metrics["task"].successes == len(executed_indices)
    assert any(event.kind == "endpoint_episode_boundary" for event in events)


class FiveValueVectorEnv:
    def __init__(self, terminated, truncated, info):
        self.terminated = terminated
        self.truncated = truncated
        self.info = info

    def reset(self, id):
        return [{"state": [0]}]

    def step(self, action, id):
        return (
            [{"state": [1]}],
            [0.0],
            [self.terminated],
            [self.truncated],
            [self.info],
        )


@pytest.mark.parametrize(
    ("terminated", "truncated", "info", "successes"),
    [
        (True, False, {"is_success": True}, 1),
        (True, False, {"is_success": False}, 0),
        (False, True, {"is_success": False}, 0),
        (True, False, {"success": True}, 1),
    ],
)
def test_libero_five_value_success_uses_upstream_is_success(
    terminated, truncated, info, successes
) -> None:
    adapter = LiberoVectorAdapter(
        FiveValueVectorEnv(terminated, truncated, info),
        [SessionConfig("task", control_hz=10, chunk_size=1)],
        observation_converter=lambda observation: observation,
        action_converter=lambda action: action,
        fallback_action=lambda: [0],
    )

    outcome = adapter.endpoints[0].execute([1])

    assert outcome.episode_boundary == (terminated or truncated)
    assert adapter.metrics["task"].successes == successes
    assert adapter.metrics["task"].terminated == int(terminated)
    assert adapter.metrics["task"].truncated == int(truncated)


def test_libero_four_value_done_cannot_be_overridden_by_info() -> None:
    class ConflictingFourValueEnv:
        def reset(self, id):
            return [{"state": [0]}]

        def step(self, action, id):
            return ([{"state": [1]}], [0.0], [True], [{"is_success": False}])

    adapter = LiberoVectorAdapter(
        ConflictingFourValueEnv(),
        [SessionConfig("task", control_hz=10, chunk_size=1)],
        observation_converter=lambda observation: observation,
        action_converter=lambda action: action,
        fallback_action=lambda: [0],
    )

    adapter.endpoints[0].execute([1])

    assert adapter.metrics["task"].successes == 1
