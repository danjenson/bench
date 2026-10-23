"""Paired bootstrap and numerical-profile checks, outside the runner API."""

from pathlib import Path

import numpy as np

from .runner import read_result
from .protocols import resolve_names
from .storage import contained_path, digest, read_json


def numerical_parity(first, second, task_id):
    """Check two profiles against the registered task tolerance without merging them."""
    recipe = resolve_names([task_id])[0]
    task_id = recipe["id"]
    outputs = [Path(first), Path(second)]
    results = [read_result(output) for output in outputs]
    profiles = [result["numerics"] for result in results]
    if set(profiles) != {"fp32-v1", "bf16-v1"}:
        raise ValueError("Numerical parity requires one fp32-v1 and one bf16-v1 result")
    models = []
    for result in results:
        model = result["model"]
        if model["numerics"] != result["numerics"] or model["digest"] != digest({key: value for key, value in model.items() if key != "digest"}):
            raise ValueError("Numerical parity requires consistent model identity metadata")
        models.append({key: value for key, value in model.items() if key not in ("digest", "numerics")})
    if models[0] != models[1] or results[0]["software"] != results[1]["software"]:
        raise ValueError("Numerical parity requires matching model state, adapter recipe and software")
    tasks = [result["tasks"][task_id] for result in results]
    for task, result in zip(tasks, results):
        if task["status"] != "complete" or task["task_digest"] != recipe["digest"]:
            raise ValueError("Numerical parity requires completed results for the registered task recipe")
        if task["model_identity"] != result["model"]["digest"] or task["method"] != recipe["method"] or task["readout"] != result["model"]["info"]["readouts"][recipe["method"]]:
            raise ValueError("Numerical parity requires consistent task readout/model metadata")
        for key in ("preparation_digest", "data_digest", "evaluation", "method", "readout", "feature_shape", "sample_count"):
            if task[key] != tasks[0][key]:
                raise ValueError(f"Numerical parity requires matching {key}")
    records = [read_json(contained_path(output, task["statistics"]["file"]))["records"] for output, task in zip(outputs, tasks)]
    indices = [{(row["id"], row.get("seed")): row for row in rows} for rows in records]
    seeds = recipe.get("probe", {}).get("seeds", [None])
    ids = {key[0] for key in indices[0]}
    expected = {(sample, seed) for sample in ids for seed in seeds}
    if not ids or len(ids) != tasks[0]["sample_count"] or any(set(index) != expected or len(index) != len(rows) for index, rows in zip(indices, records)):
        raise ValueError("Numerical parity requires identical source samples and every prescribed probe seed exactly once")
    fixed_fields = {"classification": ("label",), "segmentation": ("eligible_pixels",), "correspondence": ("source_id", "target_id", "eligible", "breakdown")}[recipe["kind"]]
    for sample in ids:
        reference = indices[0][sample, seeds[0]]
        for index in indices:
            for seed in seeds:
                if any(index[sample, seed][key] != reference[key] for key in fixed_fields):
                    raise ValueError("Numerical parity requires matching labels and scoring eligibility")
    scores, profile_seed_scores = [], []
    for task, index in zip(tasks, indices):
        seed_scores = []
        for seed in seeds:
            rows = [index[sample, seed] for sample in sorted(ids)]
            if recipe["kind"] == "classification":
                seed_scores.append(100 * sum(row["label"] == row["prediction"] for row in rows) / len(rows))
            elif recipe["kind"] == "segmentation":
                counts = np.array([[row["intersection"], row["union"]] for row in rows], dtype=float)
                if counts.shape != (len(rows), 2, recipe["classes"]) or not np.isfinite(counts).all() or (counts < 0).any() or (counts[:, 0] > counts[:, 1]).any():
                    raise ValueError("Invalid segmentation intersection/union statistics")
                intersection, union = counts.sum(0)
                if not (union > 0).any():
                    raise ValueError("No eligible segmentation classes for numerical parity")
                seed_scores.append(float((intersection[union > 0] / union[union > 0]).mean() * 100))
            else:
                eligible = sum(row["eligible"] for row in rows)
                if eligible <= 0 or any(not 0 <= row["correct"] <= row["grid_ceiling_correct"] <= row["eligible"] for row in rows):
                    raise ValueError("Invalid correspondence correct/eligible statistics")
                seed_scores.append(100 * sum(row["correct"] for row in rows) / eligible)
        metrics = {recipe["metric"]: float(np.mean(seed_scores))}
        if len(seeds) > 1:
            metrics[recipe["metric"] + "_std"] = float(np.std(seed_scores))
        if recipe["kind"] == "correspondence":
            metrics["grid_limited_pck_ceiling"] = 100 * sum(row["grid_ceiling_correct"] for row in rows) / eligible
        if set(metrics) != set(task["metrics"]) or any(not np.isclose(value, task["metrics"][name]["value"], rtol=0, atol=1e-9) for name, value in metrics.items()):
            raise ValueError("Reported metrics do not reconstruct from committed statistics")
        scores.append(metrics)
        profile_seed_scores.append(seed_scores)
    tolerance = recipe["numerical_tolerance"]["absolute_percentage_points"]

    def compare(values):
        difference = abs(values[1] - values[0])
        return {"scores": dict(zip(profiles, values)), "absolute_difference": difference, "passed": bool(difference <= tolerance or np.isclose(difference, tolerance, rtol=0, atol=1e-12))}

    comparisons = {metric: compare([score[metric] for score in scores]) for metric in scores[0]}
    per_seed = {str(seed): compare([values[index] for values in profile_seed_scores]) for index, seed in enumerate(seeds) if seed is not None}
    return {"task": task_id, "task_digest": recipe["digest"], "evaluation": tasks[0]["evaluation"], "sample_count": len(ids), "probe_seeds": seeds if seeds != [None] else [], "tolerance_percentage_points": tolerance, "metrics": comparisons, "per_seed": per_seed, "passed": all(comparison["passed"] for comparison in [*comparisons.values(), *per_seed.values()])}


def paired_difference(first, second, task_id, *, replicates=2000, seed=0):
    return paired_contrast([first, second], [-1, 1], task_id, replicates=replicates, seed=seed)


def paired_contrast(outputs, coefficients, task_id, *, replicates=2000, seed=0):
    if len(outputs) != len(coefficients) or len(outputs) < 2 or replicates < 1 or not np.isfinite(coefficients).all():
        raise ValueError("Provide matching output paths/contrast coefficients and positive bootstrap replicates")
    outputs = [Path(output) for output in outputs]
    results = [read_result(output) for output in outputs]
    if any(result["numerics"] != results[0]["numerics"] for result in results[1:]):
        raise ValueError("Paired comparisons require the same numerical profile")
    tasks = [result["tasks"][task_id] for result in results]
    a = tasks[0]
    for task in tasks:
        if task["status"] != "complete":
            raise ValueError("Paired comparisons require completed task results")
        for key in ("task_digest", "preparation_digest", "evaluation"):
            if a[key] != task[key]:
                raise ValueError(f"Paired comparison requires matching {key}")
    if "pck_at_0.1" in a["metrics"]:
        raise ValueError("SPair images recur across pairs; use category counts, not pair-independent bootstrap intervals")
    records = [read_json(contained_path(path, task["statistics"]["file"]))["records"] for path, task in zip(outputs, tasks)]
    indices = [{(row["id"], row["seed"]): row for row in rows} for rows in records]
    if any(set(index) != set(indices[0]) for index in indices[1:]):
        raise ValueError("Paired comparisons require the same source images and probe seeds")
    ids = sorted({key[0] for key in indices[0]})
    seeds = sorted({key[1] for key in indices[0]})
    if not ids or len(indices[0]) != len(ids) * len(seeds) or any(len(index) != len(rows) for index, rows in zip(indices, records)):
        raise ValueError("Paired statistics must contain each source-image/probe-seed observation exactly once")
    classification = "label" in records[0][0]
    if classification:
        arrays = [np.array([[index[sample, seed]["prediction"] == index[sample, seed]["label"] for seed in seeds] for sample in ids], dtype=float) for index in indices]
        labels = [indices[0][sample, seeds[0]]["label"] for sample in ids]
        for sample in ids:
            for probe_seed in seeds:
                if any(indices[0][sample, probe_seed]["label"] != index[sample, probe_seed]["label"] for index in indices[1:]):
                    raise ValueError("Paired labels differ")

        def metric(array, selection):
            return float(array[selection].mean() * 100)
    else:
        arrays = [np.array([[[index[sample, seed]["intersection"], index[sample, seed]["union"]] for seed in seeds] for sample in ids], dtype=float) for index in indices]

        def metric(array, selection):
            totals = array[selection].sum(0)
            ratios = np.divide(totals[:, 0], totals[:, 1], out=np.full_like(totals[:, 0], np.nan), where=totals[:, 1] > 0)
            return float(np.nanmean(ratios, axis=1).mean() * 100)
    original = np.arange(len(ids))
    observed = sum(coefficient * metric(array, original) for coefficient, array in zip(coefficients, arrays))
    groups = [np.flatnonzero(np.asarray(labels) == label) for label in sorted(set(labels))] if classification and task_id.startswith("imagenet") else [original]
    generator = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(replicates):
        selection = np.concatenate([generator.choice(group, size=len(group), replace=True) for group in groups])
        bootstrap.append(sum(coefficient * metric(array, selection) for coefficient, array in zip(coefficients, arrays)))
    lower, upper = np.percentile(bootstrap, [2.5, 97.5])
    return {"difference_percentage_points": observed, "paired_95_interval": [float(lower), float(upper)], "replicates": replicates, "source_images": len(ids), "probe_seeds": seeds, "resampling_unit": "source-image-with-all-probe-seeds", "stratified": len(groups) > 1}
