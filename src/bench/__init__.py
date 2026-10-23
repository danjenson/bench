"""Frozen vision benchmarks. Importing this module performs no I/O."""


def __getattr__(name):
    if name == "run":
        from .runner import run

        return run
    if name in {"VisionModel", "ModelInfo", "ImageBatch", "SpatialFeatures", "Extraction", "Execution", "RunResult"}:
        from . import api

        return getattr(api, name)
    raise AttributeError(name)
