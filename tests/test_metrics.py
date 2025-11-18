import re

import pytest

from ada_eval.datasets.types.metrics import (
    MetricAdditionError,
    MetricSection,
    MetricValue,
    metric_section,
    metric_value,
)


def test_metric_displays():
    value = (
        metric_value(value=-1)
        .add(metric_value(value=3.14))
        .add(metric_value(value=2.72))
        .add(metric_value(value=1.41))
        .add(metric_value(value=1.62))
    )
    assert value.value_str(7) == "7.89 (5 samples; 71.43%)"  # `count_and_value`
    value.display = "none"
    assert value.value_str(7) == ""
    value.display = "count"
    assert value.value_str(7) == "5 samples (71.43%)"
    value.display = "count_no_perc"
    assert value.value_str(7) == "5 samples"
    value.display = "value"
    assert value.value_str(7) == "7.89 (min -1; max 3.14; mean 1.58)"

    # Test singular "sample"
    assert metric_value().value_str(count_denominator=1) == "1 sample (100.00%)"

    # Test div by zero
    assert metric_value().value_str(count_denominator=0) == "1 sample (nan%)"
    assert metric_value(count=0, value=1, display="value").value_str(
        count_denominator=0
    ) == ("1 (min 1; max 1; mean nan)")


def test_metric_addition():
    # Test basic addition
    section = metric_section(count=0)
    assert section == MetricSection(
        primary_metric=MetricValue(
            count=0, sum=0, min=float("inf"), max=float("-inf"), display="count"
        ),
        sub_metrics={},
    )
    section = section.add(
        metric_section(
            sub_metrics={
                "val0": metric_value(value=10),
                "val1": metric_section(value=3.14, display="value"),
            }
        )
    )
    assert section == MetricSection(
        primary_metric=MetricValue(count=1, sum=1, min=1, max=1, display="count"),
        sub_metrics={
            "val0": MetricValue(
                count=1, sum=10, min=10, max=10, display="count_and_value"
            ),
            "val1": MetricSection(
                primary_metric=MetricValue(
                    count=1, sum=3.14, min=3.14, max=3.14, display="value"
                ),
                sub_metrics={},
            ),
        },
    )
    section = section.add(
        metric_section(
            sub_metrics={
                "val1": metric_section(
                    value=1.41, display="value", sub_metrics={"val1_0": metric_value()}
                ),
                "val2": metric_value(value=20),
            }
        )
    )
    assert section == MetricSection(
        primary_metric=MetricValue(count=2, sum=2, min=1, max=1, display="count"),
        sub_metrics={
            "val0": MetricValue(
                count=1, sum=10, min=10, max=10, display="count_and_value"
            ),
            "val1": MetricSection(
                primary_metric=MetricValue(
                    count=2, sum=4.55, min=1.41, max=3.14, display="value"
                ),
                sub_metrics={
                    "val1_0": MetricValue(count=1, sum=1, min=1, max=1, display="count")
                },
            ),
            "val2": MetricValue(
                count=1, sum=20, min=20, max=20, display="count_and_value"
            ),
        },
    )

    # Check that a discrepancy in display types raises an exception
    error_msg = "Cannot add 'count' metric and 'none' metric"
    with pytest.raises(MetricAdditionError, match=re.escape(error_msg)):
        section.add(metric_section(display="none"))
    error_msg = "Cannot add 'count_and_value' metric and 'count' metric"
    with pytest.raises(MetricAdditionError, match=re.escape(error_msg)):
        section.add(metric_section(sub_metrics={"val0": metric_value(display="count")}))

    # Check that a discrepancy in metric types raises an exception
    error_msg = "Cannot add `MetricSection` and `MetricValue`"
    with pytest.raises(MetricAdditionError, match=re.escape(error_msg)):
        section.add(metric_section(sub_metrics={"val1": metric_value()}))
