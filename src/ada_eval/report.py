import logging
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from ada_eval.datasets import EvaluatedSample, dataset_has_sample_type, load_datasets
from ada_eval.datasets.types.metrics import MetricSection, metric_section
from ada_eval.utils import type_checked

logger = logging.getLogger(__name__)


class ReportCLIArgs(BaseModel):
    """
    CLI arguments for the `report` command.

    When multiple filters are provided, only samples matching all filters
    will be included.

    Args:
        dataset_dirs: Directories containing the evaluated datasets to report on.
        datasets: Dataset dirnames to include. If `None`, include all datasets.
        dataset_kinds: Dataset kinds to include. If `None`, include all kinds.
        samples: Sample names to include. If `None`, include all samples.
        with_metric: Metric paths which must be present for a sample to be
            included. If `None`, include all samples.
        list_samples: If `True`, list the names of all samples matching the
            specified filters instead of printing the report.

    """

    dataset_dirs: Sequence[Path]
    datasets: set[str] | None
    dataset_kinds: set[str] | None
    samples: set[str] | None
    with_metric: Sequence[Sequence[str]] | None
    list_samples: bool = False


def _print_metric_table(rows: Sequence[tuple[str, str]]) -> None:
    """Print a table of metrics to stdout, with padding to separate the columns."""
    padding = (0 if len(rows) == 0 else max(len(row[0]) for row in rows)) + 2
    for name, value in rows:
        print(f"{name:<{padding}}{value}".rstrip())


def report_evaluation_results(args: ReportCLIArgs) -> None:
    """Print a report of evaluation results to stdout, according to `args`."""
    # Load all datasets
    datasets = [d for directory in args.dataset_dirs for d in load_datasets(directory)]
    # Accumulate the metrics
    metrics = metric_section(count=0)
    samples_selected: defaultdict[str, list[str]] = defaultdict(list)  # By dataset
    for dataset in datasets:
        if not (
            (args.datasets is None or dataset.dirname in args.datasets)
            and (args.dataset_kinds is None or dataset.kind in args.dataset_kinds)
        ):
            continue
        if not dataset_has_sample_type(dataset, EvaluatedSample):
            logger.warning(
                "Skipping dataset '%s' as it does not contain evaluated samples.",
                dataset.dirname,
            )
            continue
        for sample in dataset.samples:
            if args.samples is None or sample.name in args.samples:
                sample_metrics = sample.metrics()
                if args.with_metric is None or all(
                    sample_metrics.has_metric_at_path(path) for path in args.with_metric
                ):
                    metrics = metrics.add(sample.metrics())
                    samples_selected[dataset.dirname].append(sample.name)
    # Print something non-empty if there are no matching samples
    if metrics.count == 0:
        print("No samples matched the specified filters.")
    # List selected samples, if requested
    elif args.list_samples:
        for dirname in sorted(samples_selected.keys()):
            print(dirname)
            for sample_name in samples_selected[dirname]:
                print("    " + sample_name)
    # Otherwise, print the report (with top-level sections unindented and
    # separated by blank lines)
    else:
        table = []
        for name, section in metrics.sub_metrics.items():
            table.extend(
                type_checked(section, MetricSection).table(name, metrics.count)
            )
            table.append(("", ""))
        _print_metric_table(table[:-1])  # Remove trailing blank line
