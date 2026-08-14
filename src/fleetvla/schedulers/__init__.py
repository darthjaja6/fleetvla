"""Built-in schedulers and extension API."""

from .adaptive_slack import AdaptiveSlackConfig, AdaptiveSlackScheduler
from .base import BatchConfig, Scheduler
from .conformance import SchedulerConformanceError, check_scheduler
from .edf import EDFConfig, EDFScheduler
from .fifo import FIFOConfig, FIFOScheduler
from .lookahead import LookaheadConfig, LookaheadScheduler
from .registry import SchedulerRegistry
from .round_robin import RoundRobinConfig, RoundRobinScheduler

registry = SchedulerRegistry()
registry.register("adaptive-slack", AdaptiveSlackScheduler)
registry.register("edf", EDFScheduler)
registry.register("fifo", FIFOScheduler)
registry.register("lookahead", LookaheadScheduler)
registry.register("round-robin", RoundRobinScheduler)

create_scheduler = registry.create

__all__ = [
    "AdaptiveSlackConfig",
    "AdaptiveSlackScheduler",
    "BatchConfig",
    "EDFConfig",
    "EDFScheduler",
    "FIFOConfig",
    "FIFOScheduler",
    "LookaheadConfig",
    "LookaheadScheduler",
    "RoundRobinConfig",
    "RoundRobinScheduler",
    "Scheduler",
    "SchedulerConformanceError",
    "SchedulerRegistry",
    "check_scheduler",
    "create_scheduler",
    "registry",
]
