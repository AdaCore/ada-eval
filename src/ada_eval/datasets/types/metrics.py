from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel

MetricDisplay = Literal["none", "count", "value", "both"]


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
        samples = "sample" if self.count == 1 else "samples"
        fraction = self.count / count_denominator
        if self.display == "count":
            return f"{self.count} {samples} ({fraction:.2%})"
        mean = self.sum / self.count if self.count != 0 else float("nan")
        if self.display == "value":
            return f"{self.sum} (min {self.min}, max {self.max}, mean {mean:.2f})"
        if self.display == "both":
            return f"{self.sum} ({self.count} {samples}; {fraction:.2%})"
        return ""


class MetricValue(MetricBase):
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


class MetricSection(MetricBase):
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
                    raise MetricAdditionError("type", self, other)
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

    def table(
        self, top_level_name: str, count_denominator: int, indent: str = ""
    ) -> list[tuple[str, str]]:
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
    if display is None:
        display = "count" if value is None else "both"
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
    if not when:
        return metric_section(count=0, display=display, allow_zero_value=True)
    return MetricSection(
        **metric_value(
            count=count, value=value, display=display, allow_zero_value=allow_zero_value
        ).model_dump(),
        sub_metrics=sub_metrics or {},
    )
