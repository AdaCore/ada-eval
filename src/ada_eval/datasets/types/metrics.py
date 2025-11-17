from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Literal, Self

from pydantic import BaseModel

MetricDisplay = Literal["none", "count", "count_no_perc", "value", "count_and_value"]


class MetricAdditionError(ValueError):
    """Raised when two metrics are incompatible for addition."""

    def __init__(self, reason: Literal["type", "display"], left: Metric, right: Metric):
        if reason == "type":
            super().__init__(
                f"Cannot add `{type(left).__name__}` and `{type(right).__name__}`"
            )
        else:
            super().__init__(
                f"Cannot add '{left.display}' metric and '{right.display}' metric"
            )


class MetricBase(BaseModel):
    count: int
    sum: float | int
    min: float | int
    max: float | int
    display: MetricDisplay

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

    def add(self, other: MetricValue) -> MetricValue:
        if self.display != other.display:
            raise MetricAdditionError("display", self, other)
        return MetricValue(
            count=self.count + other.count,
            sum=self.sum + other.sum,
            min=min(self.min, other.min),
            max=max(self.max, other.max),
            display=self.display,
        )

    def has_metric_at_path(self, path: Sequence[str]) -> bool:
        return len(path) == 0 and self.count != 0


class MetricSection(MetricBase):
    """
    A section containing multiple metrics, aggregated over some number of samples.

    Has all the attributes of `MetricValue`, representing a top level metric
    value for the section, in addition to ...

    Attributes:
        sub_metrics: A collection of metrics which are subsidiary in some way to
            the top-level metric. Takes the form of a mapping from metric names
            to `Metric`s.

    """

    sub_metrics: Mapping[str, Metric]

    def add(self, other: MetricSection) -> MetricSection:
        if self.display != other.display:
            raise MetricAdditionError("display", self, other)
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
                    raise MetricAdditionError("type", sub1, sub2)
            elif name in self.sub_metrics:
                combined_sub_metrics[name] = self.sub_metrics[name]
            else:
                combined_sub_metrics[name] = other.sub_metrics[name]
        return MetricSection(
            count=self.count + other.count,
            sum=self.sum + other.sum,
            min=min(self.min, other.min),
            max=max(self.max, other.max),
            display=self.display,
            sub_metrics=combined_sub_metrics,
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
            top_level_name: The name to use as the label for this section.
            count_denominator: The denominator to use when computing this
                section's count percentage (if applicable).
            indent: The initial indentation level prepended to all lines.

        """
        lines = [
            (f"{indent}{top_level_name}:", indent + self.value_str(count_denominator))
        ]
        indent += " " * 4
        for name, sub_metric in self.sub_metrics.items():
            if sub_metric.count == 0:
                continue
            if isinstance(sub_metric, MetricValue):
                lines.append(
                    (f"{indent}{name}:", indent + sub_metric.value_str(self.count))
                )
            else:
                lines.extend(sub_metric.table(name, self.count, indent=indent))
        return lines


Metric = MetricValue | MetricSection


def metric_value(
    count: int = 1,
    value: float | None = None,
    display: MetricDisplay | None = None,
    *,
    when: bool = True,
    allow_zero_value: bool = False,
) -> MetricValue:
    """
    Construct a `MetricValue` for a single sample.

    Args:
        count: The number of samples to which this metric applies.
        value: The value of the metric for this sample, to set to all of `sum`,
            `min`, and `max`. By default, `value=0` implies `when=False` (see
            `allow_zero_value`). If `None`, defaults to `count`, unless `count`
            is 0, in which case those three fields are set as appropriate to
            represent an absence of data.
        display: The format in which to display the value of this metric when
            printing a report. Defaults to `"count_and_value"` if `value` is
            specified, or `"count"` otherwise.
        when: If `False`, ignore all other parameters and return an empty
            `MetricValue` (i.e. `metric_value(count=0)`).
        allow_zero_value: If `False` (default), treat `value=0` as implying
            `when=False`, returning an empty `MetricValue`. If `True`, `value=0`
            receives no special treatment.

    """
    if display is None:
        display = "count" if value is None else "count_and_value"
    if value is None and count == 0:
        value = 0
        sum_: float = 0
        min_ = float("inf")
        max_ = float("-inf")
    else:
        if value is None:
            value = count
        sum_ = value
        min_ = value
        max_ = value
    if not allow_zero_value:
        when = when and value != 0
    if not when:
        return metric_value(count=0, display=display, allow_zero_value=True)
    return MetricValue(count=count, sum=sum_, min=min_, max=max_, display=display)


def metric_section(  # noqa: PLR0913  # Most calls will not specify all arguments
    sub_metrics: Mapping[str, Metric] | None = None,
    count: int = 1,
    value: float | None = None,
    display: MetricDisplay | None = None,
    *,
    when: bool = True,
    allow_zero_value: bool = False,
) -> MetricSection:
    """
    Construct a `MetricSection` for a single sample.

    Takes the same parameters as `metric_value`, in addition to a `sub_metrics`
    which is passed directly to the `MetricSection`'s `sub_metrics` attribute
    (defaults to an empty dictionary).

    """
    if not when:
        return metric_section(count=0, display=display, allow_zero_value=True)
    return MetricSection(
        **metric_value(
            count=count, value=value, display=display, allow_zero_value=allow_zero_value
        ).model_dump(),
        sub_metrics=sub_metrics or {},
    )
