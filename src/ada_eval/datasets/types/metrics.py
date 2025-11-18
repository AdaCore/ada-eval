from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Literal, Self

from pydantic import BaseModel

MetricDisplay = Literal["none", "count", "count_no_perc", "value", "count_and_value"]


class MetricAdditionError(ValueError):
    """Raised when two metrics are incompatible for addition."""


class MetricAdditionTypeError(MetricAdditionError):
    def __init__(self, left: Metric, right: Metric):
        super().__init__(
            f"Cannot add `{type(left).__name__}` and `{type(right).__name__}`"
        )


class MetricAdditionDisplayError(MetricAdditionError):
    def __init__(self, left: MetricValue, right: MetricValue):
        super().__init__(
            f"Cannot add '{left.display}' metric and '{right.display}' metric"
        )


class MetricBase(BaseModel):
    @abstractmethod
    def add(self, other: Self) -> Self:
        """Return the sum of this metric and another metric of the same type."""

    @abstractmethod
    def has_metric_at_path(self, path: Sequence[str]) -> bool:
        """Return whether this metric contains a non-empty metric at a relative path."""


class MetricValue(MetricBase):
    """
    The value of a single metric, aggregated over some number of samples.

    Attributes:
        count: The number of samples to which this metric applies.
        sum: The sum of the metric value over all samples to which this metric applies.
        min: The minimum metric value over all samples to which this metric applies.
        max: The maximum metric value over all samples to which this metric applies.
        display: The format in which to display the value of this metric when
            printing a report.

    """

    count: int
    sum: float | int
    min: float | int
    max: float | int
    display: MetricDisplay

    def add(self, other: MetricValue) -> MetricValue:
        if self.display != other.display:
            raise MetricAdditionDisplayError(self, other)
        return MetricValue(
            count=self.count + other.count,
            sum=self.sum + other.sum,
            min=min(self.min, other.min),
            max=max(self.max, other.max),
            display=self.display,
        )

    def has_metric_at_path(self, path: Sequence[str]) -> bool:
        return len(path) == 0 and self.count != 0

    def value_str(self, count_denominator: int) -> str:
        """Return this metric's value as a string formatted according to `display`."""
        samples = "sample" if self.count == 1 else "samples"
        fraction = (
            self.count / count_denominator if count_denominator != 0 else float("nan")
        )
        if self.display in ("count", "count_no_perc"):
            perc = f" ({fraction:.2%})" if self.display == "count" else ""
            return f"{self.count} {samples}{perc}"
        mean = self.sum / self.count if self.count != 0 else float("nan")
        display_sum = f"{self.sum:.12g}"  # (round off floating point errors)
        if self.display == "value":
            return f"{display_sum} (min {self.min}; max {self.max}; mean {mean:.3g})"
        if self.display == "count_and_value":
            return f"{display_sum} ({self.count} {samples}; {fraction:.2%})"
        return ""


class MetricSection(MetricBase):
    """
    A section containing multiple metrics, aggregated over some number of samples.

    When displaying sample counts as a percentage, the denominator used for the
    sub-metrics is the `count` of the parent section.

    Attributes:
        primary_metric: A primary metric value for the section.
        sub_metrics: A collection of metrics which are subsidiary in some way to
            the primary metric. Takes the form of a mapping from metric names
            to `Metric`s.

    """

    primary_metric: MetricValue
    sub_metrics: Mapping[str, Metric]

    @property
    def count(self) -> int:
        return self.primary_metric.count

    def add(self, other: MetricSection) -> MetricSection:
        combined_primary_metric = self.primary_metric.add(other.primary_metric)
        combined_sub_metrics: dict[str, Metric] = {}
        for name in dict(self.sub_metrics) | dict(other.sub_metrics):
            if name in self.sub_metrics and name in other.sub_metrics:
                sub1 = self.sub_metrics[name]
                sub2 = other.sub_metrics[name]
                if isinstance(sub1, MetricSection) and isinstance(sub2, MetricSection):  # noqa: SIM114  # mypy doesn't support dependent types
                    combined_sub_metrics[name] = sub1.add(sub2)
                elif isinstance(sub1, MetricValue) and isinstance(sub2, MetricValue):
                    combined_sub_metrics[name] = sub1.add(sub2)
                else:
                    raise MetricAdditionTypeError(sub1, sub2)
            elif name in self.sub_metrics:
                combined_sub_metrics[name] = self.sub_metrics[name]
            else:
                combined_sub_metrics[name] = other.sub_metrics[name]
        return MetricSection(
            primary_metric=combined_primary_metric, sub_metrics=combined_sub_metrics
        )

    def has_metric_at_path(self, path: Sequence[str]) -> bool:
        if len(path) == 0:
            return self.count != 0
        first, *rest = path
        if first in self.sub_metrics:
            return self.sub_metrics[first].has_metric_at_path(rest)
        return False

    def table(
        self, top_level_name: str, count_denominator: int, indent: str = ""
    ) -> list[tuple[str, str]]:
        """
        Return a simple table representation of this metric section.

        Returns a list of `(label, value)` tuples.

        `sub_metrics` with a `count` of zero are omitted.

        `sub_metrics` are indented (in both name and value columns) with respect
        to their parent section.

        Args:
            top_level_name: The name to use as the label for this section's
                primary metric.
            count_denominator: The denominator to use when computing this
                section's count percentage (if applicable).
            indent: The initial indentation level prepended to all lines.

        """

        def _row(
            name: str, value: MetricValue, indent: str, denom: int
        ) -> tuple[str, str]:
            return (f"{indent}{name}:", indent + value.value_str(denom))

        lines = [_row(top_level_name, self.primary_metric, indent, count_denominator)]
        indent += " " * 4
        for name, sub_metric in self.sub_metrics.items():
            if sub_metric.count == 0:
                continue
            if isinstance(sub_metric, MetricValue):
                lines.append(_row(name, sub_metric, indent, self.count))
            else:
                lines.extend(sub_metric.table(name, self.count, indent=indent))
        return lines


Metric = MetricValue | MetricSection


def metric_value(
    value: float | None = None,
    display: MetricDisplay | None = None,
    *,
    when: bool = True,
    allow_zero_value: bool = False,
) -> MetricValue:
    """
    Construct a `MetricValue` for a single sample.

    Args:
        value: The value of the metric for this sample, to set to all of `sum`,
            `min`, and `max`. If `None`, defaults to `1`. If `allow_zero_value`
            is `False` (the default), A `value` of `0` implies `when=False`.
        display: The format in which to display the value of this metric when
            printing a report. Defaults to `"count_and_value"` if `value` is
            specified, or `"count"` otherwise.
        when: If `False`, ignore all other parameters and return an empty
            `MetricValue`.
        allow_zero_value: If `False` (default), treat `value=0` as implying
            `when=False`, returning an empty `MetricValue`. If `True`, `value=0`
            receives no special treatment.

    """
    if display is None:
        display = "count" if value is None else "count_and_value"
    if value is None:
        value = 1
    sum_ = value
    min_ = value
    max_ = value
    if not allow_zero_value:
        when = when and (value != 0)
    if not when:
        return empty_metric_value(display=display)
    return MetricValue(count=1, sum=sum_, min=min_, max=max_, display=display)


def empty_metric_value(display: MetricDisplay = "count") -> MetricValue:
    """
    Construct a `MetricValue` representing the absence of any samples.

    Args:
        display: The format in which to display the value of this metric when
            printing a report.

    """
    return MetricValue(
        count=0, sum=0, min=float("inf"), max=float("-inf"), display=display
    )


def metric_section(
    sub_metrics: Mapping[str, Metric] | None = None,
    value: float | None = None,
    display: MetricDisplay | None = None,
    *,
    when: bool = True,
    allow_zero_value: bool = False,
) -> MetricSection:
    """
    Construct a `MetricSection` for a single sample.

    Takes the same parameters as `metric_value()` to construct the primary
    metric, in addition to a `sub_metrics` which is passed directly to the
    `MetricSection`'s `sub_metrics` attribute (defaults to an empty dictionary).

    If the `when` condition is `False`, the `MetricSection` will have an empty
    `sub_metrics` dict, in addition to an empty `primary_metric`.

    """
    primary_metric = metric_value(
        value=value, display=display, when=when, allow_zero_value=allow_zero_value
    )
    if sub_metrics is None or not when:
        sub_metrics = {}
    return MetricSection(primary_metric=primary_metric, sub_metrics=sub_metrics)


def empty_metric_section(display: MetricDisplay = "count") -> MetricSection:
    """
    Construct a `MetricSection` representing the absence of any samples.

    Args:
        display: The format in which to display the value of the primary metric
            when printing a report.

    """
    return metric_section(when=False, display=display)
