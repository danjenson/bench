"""Apply SPEC pair gates to verified, predeclared checkpoint comparisons.

Each row names its family/checkpoints and contains ``monitoring``, ``reference``
and ``doubled`` paired_difference reports, or None for measurements still missing.
The caller verifies compatible recipes, readouts and source-image statistics and
the reference's training-budget adequacy before supplying reference comparisons.
This accounting does not certify those prerequisites or predeclaration itself.
"""

import math


def summarize(pairs, *, delta, minimum_qualifying_pairs):
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("The detectable change must be positive and finite")
    if minimum_qualifying_pairs is not None and (type(minimum_qualifying_pairs) is not int or minimum_qualifying_pairs < 3):
        raise ValueError("Predeclare at least three qualifying pairs spanning CNN, ViT and glimpse families")
    rows, identities = [], set()
    for pair in pairs:
        identity = (pair["family"], frozenset((pair["first"], pair["second"])))
        if pair["family"] not in ("cnn", "vit", "glimpse") or pair["first"] == pair["second"] or identity in identities:
            raise ValueError("Each pair must have distinct checkpoints and a unique supported family/pair identity")
        identities.add(identity)
        row = {key: pair[key] for key in ("family", "first", "second")}
        for name in ("monitoring", "reference", "doubled"):
            measurement = pair[name]
            if measurement is None:
                row[name] = None
                continue
            change = measurement["difference_percentage_points"]
            lower, upper = measurement["paired_95_interval"]
            if not all(math.isfinite(value) for value in (change, lower, upper)) or lower > upper:
                raise ValueError("Paired differences and ordered intervals must be finite")
            row[name] = {**measurement, "supported_direction": 1 if lower > 0 else -1 if upper < 0 else 0}
        monitoring, reference, doubled = (row[key] for key in ("monitoring", "reference", "doubled"))
        row["precision_passed"] = (monitoring["paired_95_interval"][1] - monitoring["paired_95_interval"][0]) / 2 <= delta / 2 if monitoring else None
        row["qualifies"] = bool(abs(reference["difference_percentage_points"]) >= delta and reference["supported_direction"]) if reference else None
        row["agreement_passed"] = monitoring["supported_direction"] == reference["supported_direction"] if row["qualifies"] and monitoring else None
        row["supported_reversal"] = doubled["supported_direction"] == -reference["supported_direction"] if row["qualifies"] and doubled else None
        rows.append(row)
    qualifying = [row for row in rows if row["qualifies"]]
    families = sorted({row["family"] for row in qualifying})
    agreeing = sum(row["agreement_passed"] is True for row in qualifying)
    precision_failures = sum(row["precision_passed"] is False for row in rows)
    missing_precision = sum(row["monitoring"] is None for row in rows)
    missing_reference = sum(row["reference"] is None for row in rows)
    missing_agreement = sum(row["agreement_passed"] is None for row in qualifying)
    reversals = sum(row["supported_reversal"] is True for row in qualifying)
    missing_doubled = sum(row["supported_reversal"] is None for row in qualifying)
    enough = minimum_qualifying_pairs is not None and len(qualifying) >= minimum_qualifying_pairs and families == ["cnn", "glimpse", "vit"]
    complete = bool(rows) and not missing_reference and enough
    return {
        "delta_percentage_points": delta,
        "pairs": rows,
        "precision": {"status": "failed" if precision_failures else "pending" if not rows or missing_precision else "passed", "half_width_limit": delta / 2, "failures": precision_failures, "missing": missing_precision},
        "agreement": {"status": "pending" if not complete or missing_agreement else "passed" if 10 * agreeing >= 9 * len(qualifying) else "failed", "minimum_qualifying_pairs": minimum_qualifying_pairs, "qualifying_pairs": len(qualifying), "qualifying_families": families, "agreeing_pairs": agreeing, "fraction": agreeing / len(qualifying) if qualifying and not missing_agreement else None, "missing_reference_pairs": missing_reference, "missing_qualifying_monitoring_pairs": missing_agreement, "sufficient_qualifying_set": enough},
        "budget_reversals": {"status": "failed" if reversals else "pending" if not complete or missing_doubled else "passed", "supported_reversals": reversals, "missing_qualifying_doubled_pairs": missing_doubled, "scope": "Pair-direction check only; the separate 90% checkpoint score-stability gate is also required."},
    }
