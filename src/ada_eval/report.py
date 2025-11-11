from collections.abc import Collection, Iterable
from pathlib import Path

from ada_eval.datasets import (
    Eval,
    EvaluatedSample,
    dataset_has_sample_type,
    load_datasets,
)
from ada_eval.datasets.types.metrics import MetricSection, metric_section
from ada_eval.utils import type_checked


def print_table(rows: Iterable[tuple[str, str]]) -> None:
    padding = max(len(row[0]) for row in rows) + 2
    for name, value in rows:
        print(f"{name:<{padding}}{value}".rstrip())


def report_evaluation_results(
    dataset_dirs: Iterable[Path],
    datasets_filter: Collection[str] | None,
    dataset_kinds_filter: Collection[str] | None,
    samples_filter: Collection[str] | None,
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
                metrics = metrics.add(sample.metrics())
    # Print the report (with each eval as a separate table)
    eval_sections = {
        e: metrics.sub_metrics[e.value] for e in Eval if e.value in metrics.sub_metrics
    }
    overall_metrics = metric_section(
        {k: v for k, v in metrics.sub_metrics.items() if k not in eval_sections},
        count=metrics.count,
        display="count_no_perc",
    )
    table = overall_metrics.table("total samples", metrics.count)
    for e, section in eval_sections.items():
        table.append(("", ""))
        table.extend(type_checked(section, MetricSection).table(e.value, metrics.count))
    print_table(table)
