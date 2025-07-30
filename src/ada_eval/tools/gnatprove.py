import logging
from pathlib import Path

from ada_eval.datasets.types import (
    DatasetType,
    EvaluatedSparkSample,
    EvaluationStatsSpark,
    GeneratedSample,
    GeneratedSparkSample,
)
from ada_eval.utils import run_cmd_with_timeout

from .generic_tool import BaseConfig, EvaluationTool, UnsupportedSampleTypeError

logger = logging.getLogger(__name__)


GNATPROVE_TOOL_NAME = "GNATprove"


class GnatProve(EvaluationTool):
    config_type = BaseConfig
    config: BaseConfig

    def __init__(self, config: BaseConfig):
        self.config = config

    @classmethod
    def from_config_file(cls, config_file: Path):
        config = BaseConfig.model_validate_json(config_file.read_text(encoding="utf-8"))
        return cls(config)

    @classmethod
    def from_config(cls, config: BaseConfig):
        return cls(config)

    @property
    def name(self) -> str:
        return GNATPROVE_TOOL_NAME

    def supported_dataset_types(self) -> tuple[DatasetType]:
        return (DatasetType.SPARK,)

    def evaluate(self, sample: GeneratedSample) -> EvaluatedSparkSample:
        if not isinstance(sample, GeneratedSparkSample):
            raise UnsupportedSampleTypeError(type(sample), GeneratedSparkSample)
        with sample.generated_solution.unpacked() as working_dir:
            logger.debug("Evaluating %s with GNATprove in %s", sample.name, working_dir)
            # Run `gnatprove` with no arguments (except those required to get
            # non-zero exit code on failure)
            #
            # TODO (#2): Restrict which subprogram is analyzed and scrape more
            # detailed results from `obj/gnatprove.out`
            result, time_ms = run_cmd_with_timeout(
                ["gnatprove", "--checks-as-errors=on", "--warnings=error", "-k"],
                working_dir,
                self.config.timeout_s,
            )
            # Return a GeneratedSparkSample
            successfully_proven = False if result is None else (result.returncode == 0)
            return EvaluatedSparkSample(
                **sample.model_dump(),  # Copy all fields from the original sample
                evaluation_stats=EvaluationStatsSpark(
                    successfully_proven=successfully_proven,
                    runtime_ms=time_ms,
                ),
            )
