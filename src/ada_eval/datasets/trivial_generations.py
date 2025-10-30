from collections.abc import Iterable

from ada_eval.datasets import Dataset, ExplainSample, GeneratedSample, Sample
from ada_eval.datasets.types import GENERATED_SAMPLE_TYPES, GenerationStats


def _generate_null_or_canonical(
    datasets: Iterable[Dataset[Sample]], *, canonical: bool
) -> list[Dataset[GeneratedSample]]:
    generated_datasets: list[Dataset[GeneratedSample]] = []
    for dataset in datasets:
        sample_type = GENERATED_SAMPLE_TYPES[dataset.kind]
        samples = [
            sample_type(
                **sample.model_dump(
                    exclude={
                        "generation_stats",
                        "generated_solution",
                        "evaluation_results",
                    }
                ),
                generation_stats=GenerationStats(
                    exit_code=0, stdout="", stderr="", runtime_ms=0
                ),
                generated_solution=(
                    sample.canonical_solution
                    if canonical
                    else ("" if isinstance(sample, ExplainSample) else sample.sources)
                ),
            )
            for sample in dataset.samples
        ]
        generated_datasets.append(
            Dataset(name=dataset.name, sample_type=sample_type, samples=samples)
        )
    return generated_datasets


def generate_null(
    datasets: Iterable[Dataset[Sample]],
) -> list[Dataset[GeneratedSample]]:
    """
    Perform null generations on a collection of datasets.

    For `AdaSample`s, the generated solution will be an unmodified copy of the
    raw `sources`. For `ExplainSample`s, the generated explanation will be the
    empty string.

    The `generation_stats` will be set to
    `GenerationStats(exit_code=0, stdout="", stderr="", runtime_ms=0)`.
    """
    return _generate_null_or_canonical(datasets, canonical=False)


def generate_canonical(
    datasets: Iterable[Dataset[Sample]],
) -> list[Dataset[GeneratedSample]]:
    """
    Perform canonical generations on a collection of datasets.

    The `generated_solution` field of each sample will be set to its
    `canonical_solution`. The `generation_stats` will be set to
    `GenerationStats(exit_code=0, stdout="", stderr="", runtime_ms=0)`.
    """
    return _generate_null_or_canonical(datasets, canonical=True)
