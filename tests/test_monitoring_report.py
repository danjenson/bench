from copy import deepcopy
from pathlib import Path
import runpy

import pytest


def pair(family, index, *, change=3, interval=(2.5, 3.5)):
    measurement = {"difference_percentage_points": change, "paired_95_interval": list(interval)}
    return {"family": family, "first": str(index), "second": str(index + 1), "monitoring": deepcopy(measurement), "reference": deepcopy(measurement), "doubled": deepcopy(measurement)}


def test_monitoring_agreement_threshold_preserves_reversals_and_each_pair():
    summarize = runpy.run_path(str(Path(__file__).parents[1] / "examples/monitoring_pair_gates.py"))["summarize"]
    pairs = [pair(("cnn", "vit", "glimpse")[index % 3], index) for index in range(10)]
    pairs[-1]["monitoring"] = {"difference_percentage_points": -3, "paired_95_interval": [-3.5, -2.5]}
    report = summarize(pairs, delta=2, minimum_qualifying_pairs=10)
    assert report["agreement"]["status"] == "passed"
    assert report["agreement"]["fraction"] == 0.9
    assert len(report["pairs"]) == 10 and report["pairs"][-1]["agreement_passed"] is False
    pairs[-2]["monitoring"] = pairs[-1]["monitoring"]
    assert summarize(pairs, delta=2, minimum_qualifying_pairs=10)["agreement"]["status"] == "failed"
    pairs[-1]["doubled"] = pairs[-1]["monitoring"]
    report = summarize(pairs, delta=2, minimum_qualifying_pairs=10)
    assert report["budget_reversals"]["status"] == "failed"
    assert report["budget_reversals"]["supported_reversals"] == 1


def test_missing_or_unqualified_pairs_cannot_establish_sensitivity():
    summarize = runpy.run_path(str(Path(__file__).parents[1] / "examples/monitoring_pair_gates.py"))["summarize"]
    pairs = [pair(family, index) for index, family in enumerate(("cnn", "vit", "glimpse"))]
    assert summarize(pairs, delta=2, minimum_qualifying_pairs=3)["agreement"]["status"] == "passed"
    negative = [pair(family, index, change=-3, interval=(-3.5, -2.5)) for index, family in enumerate(("cnn", "vit", "glimpse"))]
    assert summarize(negative, delta=2, minimum_qualifying_pairs=3)["agreement"]["status"] == "passed"
    for missing in ("monitoring", "reference", "doubled"):
        incomplete = deepcopy(pairs)
        incomplete[-1][missing] = None
        report = summarize(incomplete, delta=2, minimum_qualifying_pairs=3)
        gate = "budget_reversals" if missing == "doubled" else "agreement"
        assert report[gate]["status"] == "pending"
        assert len(report["pairs"]) == 3
        if missing == "monitoring":
            assert report["agreement"]["fraction"] is None
    assert summarize(pairs, delta=2, minimum_qualifying_pairs=None)["agreement"]["status"] == "pending"
    assert summarize([], delta=2, minimum_qualifying_pairs=3)["agreement"]["fraction"] is None
    assert summarize([], delta=2, minimum_qualifying_pairs=3)["precision"]["status"] == "pending"
    for changed_reference in ({"difference_percentage_points": 1.99, "paired_95_interval": [1.5, 2.5]}, {"difference_percentage_points": 3, "paired_95_interval": [0, 6]}):
        insufficient = deepcopy(pairs)
        insufficient[-1]["reference"] = changed_reference
        report = summarize(insufficient, delta=2, minimum_qualifying_pairs=3)
        assert report["agreement"]["qualifying_families"] == ["cnn", "vit"]
        assert not report["agreement"]["sufficient_qualifying_set"]
        assert report["agreement"]["status"] == "pending"


def test_precision_is_per_pair_and_zero_is_not_a_supported_direction():
    summarize = runpy.run_path(str(Path(__file__).parents[1] / "examples/monitoring_pair_gates.py"))["summarize"]
    pairs = [pair(family, index) for index, family in enumerate(("cnn", "vit", "glimpse"))]
    pairs[0]["monitoring"] = {"difference_percentage_points": 1, "paired_95_interval": [0, 2]}
    report = summarize(pairs, delta=2, minimum_qualifying_pairs=3)
    assert report["precision"]["status"] == "passed"
    assert report["agreement"]["status"] == "failed"
    pairs[0]["monitoring"]["paired_95_interval"][1] = 2.01
    assert summarize(pairs, delta=2, minimum_qualifying_pairs=3)["precision"]["status"] == "failed"
    invalid = deepcopy(pairs)
    invalid[0]["reference"]["paired_95_interval"] = [float("nan"), 3]
    with pytest.raises(ValueError, match="finite"):
        summarize(invalid, delta=2, minimum_qualifying_pairs=3)
    with pytest.raises(ValueError, match="unique"):
        summarize(pairs + [pairs[0]], delta=2, minimum_qualifying_pairs=3)
    reverse = {**pairs[0], "first": pairs[0]["second"], "second": pairs[0]["first"]}
    with pytest.raises(ValueError, match="unique"):
        summarize(pairs + [reverse], delta=2, minimum_qualifying_pairs=3)
