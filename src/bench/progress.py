"""Transient terminal progress shared by benchmark stages."""

import warnings

from rich.console import Console
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm


def progress_bar(iterable=None, *, total, desc, unit, enabled=None):
    console = Console(stderr=True, force_terminal=enabled)
    disabled = enabled is False or total == 0 or (enabled is None and (not console.is_terminal or console.is_dumb_terminal))
    # tqdm.rich announces its experimental status on every construction.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TqdmExperimentalWarning)
        return tqdm(
            iterable, total=total, desc=desc, unit=unit, disable=disabled, leave=False,
            mininterval=0.25,
            options={"console": console, "refresh_per_second": 4,
                     "redirect_stdout": False, "redirect_stderr": False},
        )
