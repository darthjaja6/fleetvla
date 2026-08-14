"""Optional LeRobot policy integration.

Validated against LeRobot 0.6.2's `PreTrainedPolicy.predict_action_chunk`
contract: batched observations in, `[batch, horizon, action_dim]` out.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from ..backend import BackendResult
from ..types import ActionChunk, InferenceCostModel, Observation


class LeRobotPolicyBackend:
    """Dynamic-batching backend for SmolVLA and other LeRobot policies."""

    def __init__(
        self,
        policy: Any,
        *,
        preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        postprocessor: Callable[[Any], Any] | None = None,
        predicted_base_latency_s: float = 0.05,
        predicted_per_item_latency_s: float = 0.01,
        cost_update_alpha: float = 0.2,
        execution_horizon: int | None = None,
    ) -> None:
        if not hasattr(policy, "predict_action_chunk"):
            raise TypeError("LeRobot policy must define predict_action_chunk")
        if not callable(getattr(policy, "reset", None)):
            raise TypeError("LeRobot policy must define reset")
        self.policy = policy
        self.preprocessor = preprocessor or (lambda batch: batch)
        self.postprocessor = postprocessor or (lambda action: action)
        self.cost_model = InferenceCostModel(
            predicted_base_latency_s, predicted_per_item_latency_s
        )
        if not math.isfinite(cost_update_alpha) or not 0 < cost_update_alpha <= 1:
            raise ValueError("cost_update_alpha must be in (0, 1]")
        self.cost_update_alpha = cost_update_alpha
        if execution_horizon is None:
            execution_horizon = getattr(
                getattr(policy, "config", None), "n_action_steps", None
            )
        if execution_horizon is not None and (
            not isinstance(execution_horizon, int)
            or isinstance(execution_horizon, bool)
            or execution_horizon <= 0
        ):
            raise ValueError("execution_horizon must be positive")
        self.execution_horizon = execution_horizon
        self._reset_lock = threading.Lock()
        self._reset_pending = True

    @classmethod
    def from_smolvla_pretrained(
        cls,
        model_id: str,
        *,
        revision: str | None = None,
        **backend_kwargs: Any,
    ) -> "LeRobotPolicyBackend":
        try:
            from lerobot.policies import make_pre_post_processors
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        except ImportError as error:
            raise ImportError(
                "SmolVLA support requires Python 3.12+ and "
                "`pip install 'fleetvla[smolvla]'`"
            ) from error
        policy = SmolVLAPolicy.from_pretrained(model_id, revision=revision)
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=model_id,
            pretrained_revision=revision,
        )
        return cls(
            policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            **backend_kwargs,
        )

    def infer(
        self, observations: Sequence[Observation], started_at_s: float
    ) -> BackendResult:
        observations = tuple(observations)
        if not observations:
            raise ValueError("cannot infer an empty batch")
        payloads = []
        for observation in observations:
            if not isinstance(observation.payload, Mapping):
                raise TypeError("LeRobot observation payloads must be mappings")
            payloads.append(dict(observation.payload))
        batch = self.preprocessor(_collate(payloads))
        wall_start_s = time.perf_counter()
        try:
            import torch
        except ImportError as error:
            raise ImportError("LeRobot inference requires torch") from error
        with torch.inference_mode():
            with self._reset_lock:
                if self._reset_pending:
                    self.policy.reset()
                    self._reset_pending = False
            try:
                action_batch = self.policy.predict_action_chunk(batch)
            except Exception:
                self.reset()
                raise
        latency_s = time.perf_counter() - wall_start_s
        observed_per_item_s = max(
            0.0,
            (latency_s - self.cost_model.base_latency_s) / len(observations),
        )
        self.cost_model = InferenceCostModel(
            self.cost_model.base_latency_s,
            (1 - self.cost_update_alpha) * self.cost_model.per_item_latency_s
            + self.cost_update_alpha * observed_per_item_s,
        )
        if getattr(action_batch, "ndim", None) != 3:
            raise ValueError(
                "predict_action_chunk must return [batch, horizon, action_dim]"
            )
        if int(action_batch.shape[0]) != len(observations):
            raise ValueError("policy output batch size does not match observations")
        output_horizon = int(action_batch.shape[1])
        execution_horizon = self.execution_horizon or output_horizon
        if execution_horizon > output_horizon:
            raise ValueError("execution horizon exceeds the policy output horizon")
        chunks = []
        for batch_index, observation in enumerate(observations):
            actions = []
            for action_index in range(execution_horizon):
                processed_action = self.postprocessor(
                    action_batch[batch_index : batch_index + 1, action_index]
                )
                actions.append(_to_python(processed_action))
            chunks.append(
                ActionChunk(
                    session_id=observation.session_id,
                    observation_sequence=observation.sequence,
                    generation=observation.generation,
                    actions=tuple(actions),
                    produced_at_s=started_at_s + latency_s,
                    auxiliary={
                        "policy_type": type(self.policy).__name__,
                        "output_shape": tuple(int(size) for size in action_batch.shape),
                        "execution_horizon": execution_horizon,
                    },
                    action_index_start=observation.action_index_start,
                )
            )
        return BackendResult(latency_s, tuple(chunks))

    def reset(self) -> None:
        with self._reset_lock:
            self._reset_pending = True

    def reset_session(self, session_id: str) -> None:
        """Invalidate shared policy caches before the next inference batch."""

        del session_id
        self.reset()


def _collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    keys = set(items[0])
    if any(set(item) != keys for item in items[1:]):
        raise ValueError("batched observations must have the same fields")
    try:
        import torch
    except ImportError as error:
        raise ImportError("LeRobot batching requires torch") from error
    batch: dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in items]
        if all(isinstance(value, torch.Tensor) for value in values):
            batch[key] = torch.cat(values, dim=0)
        elif all(isinstance(value, list) for value in values):
            batch[key] = [element for value in values for element in value]
        elif all(
            type(value) is type(values[0]) and value == values[0] for value in values
        ):
            batch[key] = values[0]
        else:
            raise TypeError(
                f"preprocessed field {key!r} must contain tensors, lists, "
                "or one shared metadata value"
            )
    return batch


def _to_python(value: Any) -> Any:
    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach().cpu()
    tolist = getattr(value, "tolist", None)
    if tolist is not None:
        value = tolist()
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value
