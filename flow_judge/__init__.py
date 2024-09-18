from importlib.metadata import PackageNotFoundError, version

from .flow_judge import EvalInput, FlowJudge
from .formatting import format_rubric, format_user_prompt, format_vars
from .metrics import CustomMetric, Metric, RubricItem, list_all_metrics
from .models import BaseFlowJudgeModel, ModelFactory
from .parsing import EvalOutput

try:
    __version__ = version("flow-judge")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"

__all__ = [
    "FlowJudge",
    "EvalInput",
    "format_vars",
    "format_rubric",
    "format_user_prompt",
    "RubricItem",
    "Metric",
    "CustomMetric",
    "BaseFlowJudgeModel",
    "ModelFactory",
    "EvalOutput",
]

# Add all metric names to __all__
__all__ += list_all_metrics()
