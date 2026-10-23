"""Check the pinned Probe3D kernel with explicitly aligned v1 conventions.

The downloaded upstream files remain outside bench's dependency graph. Only
their two verified functions execute. See docs/references.md for attribution
and why this is algorithm parity, not parity with published Probe3D scores.
"""

import argparse
import ast
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from bench.storage import file_digest, write_json
from bench.tasks.correspondence import match_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching-source", type=Path, required=True)
    parser.add_argument("--helper-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = {args.matching_source: "c63bb41f35a0dc2dee7ba01bbdaf6ced1bacf4d7293186c8fcd1e9583f18161a", args.helper_source: "1a1a6ebdeb4018e3772ff29db4558d0dab5345f4920d4dc80ff328fed7703b0c"}
    for path, checksum in expected.items():
        if file_digest(path) != checksum:
            raise ValueError(f"Unexpected Probe3D source digest: {path}")
    if not torch.cuda.is_available():
        raise ValueError("The pinned upstream kernel requires CUDA")
    replacements = [
        ("feats = nn_F.normalize(feats, p=2, dim=1)", "feats = feats.float()"),
        ('mode="bilinear", align_corners=True', 'mode="bilinear", align_corners=False, padding_mode="border"'),
        ("kp_i_F = kp_i_F[0, :, 0].t()", "kp_i_F = nn_F.normalize(kp_i_F[0, :, 0].t(), p=2, dim=1)"),
        ('einsum(kp_i_F, feats_j, "k f, f h w -> k h w")', 'einsum(kp_i_F, nn_F.normalize(feats_j, p=2, dim=0), "k f, f h w -> k h w")'),
        ("argmax_2d(heatmaps, max_value=True).float().cpu() / feats.shape[-1]", "(argmax_2d(heatmaps, max_value=True).float().cpu() + 0.5) / feats.shape[-1]"),
    ]
    aligned = args.matching_source.read_text()
    for original, replacement in replacements:
        if aligned.count(original) != 1:
            raise ValueError("The pinned reference alignment no longer applies exactly once")
        aligned = aligned.replace(original, replacement)

    def einsum(first, second, pattern):
        assert pattern == "k f, f h w -> k h w"
        return torch.einsum("kf,fhw->khw", first, second)

    scope = {"torch": torch, "np": np, "nn_F": F, "einsum": einsum}
    for text, name in [(args.helper_source.read_text(), "argmax_2d"), (aligned, "compute_errors")]:
        tree = ast.parse(text)
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
        if len(nodes) != 1:
            raise ValueError(f"Missing pinned reference function: {name}")
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(args.matching_source), "exec"), scope)
    generator = torch.Generator().manual_seed(41)
    checks = []
    for side in (4, 7, 16, 28):
        for trial in range(12):
            values = torch.randn(2, 24, side, side, generator=generator)
            values *= torch.rand(2, 1, side, side, generator=generator) * 9 + 0.1
            points = torch.rand(2, 40, 2, generator=generator) * 448
            points[:, :4] = torch.tensor([[0., 0.], [447., 447.], [0., 447.], [447., 0.]])
            if trial == 0:
                values.zero_()  # The first raster location must win ties.
            keypoints = [torch.cat([part, torch.ones(40, 1)], dim=1) for part in points]
            instance = (torch.zeros(3, 448, 448), np.ones((448, 448)), keypoints[0], torch.zeros(3, 448, 448), np.ones((448, 448)), keypoints[1], 0.8, None)
            errors, _, _, _, heatmaps = scope["compute_errors"](lambda images: values.to(images.device), instance, return_heatmaps=True)
            expected_correct = int((errors <= 0.1).sum())
            features = [{"values": part, "grid_to_original": torch.eye(3), "extent_in_original": torch.tensor([0., 0., 1., 1.])} for part in values]
            annotation = {"src_kps": points[0].tolist(), "trg_kps": points[1].tolist(), "src_imsize": [448, 448, 3], "trg_imsize": [448, 448, 3], "trg_bndbox": [44.8, 44.8, 403.2, 403.2]}
            actual = match_pair(*features, annotation)
            if actual["correct"] != expected_correct or actual["eligible"] != len(errors):
                raise AssertionError(f"Reference disagreement at grid {side}, trial {trial}")
            predicted = (scope["argmax_2d"](heatmaps).float().cpu() + 0.5) / side * 448
            exact_targets = {**annotation, "trg_kps": predicted.tolist()}
            matched_locations = match_pair(*features, exact_targets, threshold=1e-6)["correct"]
            if matched_locations != len(predicted):
                raise AssertionError(f"Reference predicted-location disagreement at grid {side}, trial {trial}")
            checks.append({"grid": side, "trial": trial, "matched_locations": matched_locations, **actual})
    write_json(args.output, {"status": "complete", "reference": "mbanani/probe3d@a1f14640076e38b8bc07b66d0fe2d01d15691e9d", "source_sha256": {str(path): checksum for path, checksum in expected.items()}, "alignment_edits": replacements, "cases": checks, "scope": "Kernel parity on identical square canvases and native feature grids; explicit v1 normalization, interpolation, border and center alignment; dataset preprocessing and category aggregation differ from upstream"})
    print(f"Aligned matching parity: {len(checks)} cases, {sum(row['eligible'] for row in checks)} correspondences")


if __name__ == "__main__":
    main()
