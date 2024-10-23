from importlib.metadata import PackageNotFoundError, version

from flow_judge.eval_data_types import EvalInput, EvalOutput
from flow_judge.flow_judge import AsyncFlowJudge, FlowJudge
from flow_judge.metrics import CustomMetric, Metric, RubricItem, list_all_metrics
from flow_judge.models.common import BaseFlowJudgeModel
from flow_judge.utils.prompt_formatter import format_rubric, format_user_prompt, format_vars

try:
    __version__ = version("flow-judge")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"

__all__ = [
    "FlowJudge",
    "AsyncFlowJudge",
    "EvalInput",
    "format_vars",
    "format_rubric",
    "format_user_prompt",
    "RubricItem",
    "Metric",
    "CustomMetric",
    "BaseFlowJudgeModel",
    "EvalOutput",
]

# Conditional imports for optional dependencies
try:
    from flow_judge.models.huggingface import Hf

    __all__.append("Hf")
except ImportError:
    Hf = None

try:
    from flow_judge.models.vllm import Vllm

    __all__.append("Vllm")
except ImportError:
    Vllm = None

try:
    from flow_judge.models.llamafile import Llamafile

    __all__.append("Llamafile")
except ImportError:
    Llamafile = None

try:
    from flow_judge.models.baseten import Baseten

    __all__.append("Baseten")
except ImportError:
    Baseten = None


def get_available_models():
    """Return a list of available model classes based on installed extras."""
    models = [BaseFlowJudgeModel]
    if Hf is not None:
        models.append(Hf)
    if Vllm is not None:
        models.append(Vllm)
    if Llamafile is not None:
        models.append(Llamafile)
    if Baseten is not None:
        models.append(Baseten)
    return models


__all__.append("get_available_models")

# Add all metric names to __all__
__all__ += list_all_metrics()
