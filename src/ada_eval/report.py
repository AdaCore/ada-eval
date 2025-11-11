from collections.abc import Collection, Iterable, Sequence
from pathlib import Path

from ada_eval.datasets import EvaluatedSample, dataset_has_sample_type, load_datasets
from ada_eval.datasets.types.metrics import MetricSection, metric_section
from ada_eval.utils import type_checked


def print_table(rows: Sequence[tuple[str, str]]) -> None:
    if len(rows) == 0:
        return
    padding = max(len(row[0]) for row in rows) + 2
    for name, value in rows:
        print(f"{name:<{padding}}{value}".rstrip())


def report_evaluation_results(
    dataset_dirs: Iterable[Path],
    datasets_filter: Collection[str] | None,
    dataset_kinds_filter: Collection[str] | None,
    samples_filter: Collection[str] | None,
    metrics_filter: Collection[Sequence[str]] | None,
) -> None:
    # Ensure filter lookup is O(1)
    if datasets_filter is not None:
        datasets_filter = set(datasets_filter)
    if dataset_kinds_filter is not None:
        dataset_kinds_filter = set(dataset_kinds_filter)
    if samples_filter is not None:
        samples_filter = set(samples_filter)
    # Load all datasets
    datasets = [d for directory in dataset_dirs for d in load_datasets(directory)]
    # Accumulate the metrics
    metrics = metric_section(count=0)
    for dataset in datasets:
        if not (
            (datasets_filter is None or dataset.dirname in datasets_filter)
            and (dataset_kinds_filter is None or dataset.kind in dataset_kinds_filter)
        ):
            continue
        if not dataset_has_sample_type(dataset, EvaluatedSample):
            msg = (
                f"dataset '{dataset.dirname}' does not contain "
                f"`EvaluatedSample`s; type is {dataset.sample_type.__name__}."
            )
            raise ValueError(msg)
        for sample in dataset.samples:
            if samples_filter is None or sample.name in samples_filter:
                sample_metrics = sample.metrics()
                if metrics_filter is None or all(
                    sample_metrics.has_metric_at_path(path) for path in metrics_filter
                ):
                    metrics = metrics.add(sample.metrics())
    # Print the report (with top-level sections unindented and separated by blank lines)
    if metrics.count == 0:
        print("No samples matched the specified filters.")
        return
    table = []
    for name, section in metrics.sub_metrics.items():
        table.extend(type_checked(section, MetricSection).table(name, metrics.count))
        table.append(("", ""))
    print_table(table[:-1])  # Remove trailing blank line
