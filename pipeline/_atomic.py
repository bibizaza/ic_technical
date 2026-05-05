"""Helpers for atomic file writes.

Use ``atomic_write_text`` whenever a process kill mid-write would leave a
shared file (master_prices.csv, history.json, draft_state.json, breadth_cache.json)
in a torn state. Writes go to a sibling tempfile + os.replace; the target
file is never partially written.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Union

PathLike = Union[str, os.PathLike]


def atomic_write_text(path: PathLike, content: str, *, encoding: str = "utf-8") -> None:
    """Write ``content`` to ``path`` atomically.

    - Writes to a tempfile in the same directory, then ``os.replace`` — which
      is atomic on POSIX and NTFS.
    - Preserves the original file's permission bits when it exists; defaults
      to 0o644 otherwise (mkstemp's 0o600 default would leak through).
    - Removes the tempfile if anything fails, so partial state never lingers.
    """
    target = Path(path)

    try:
        original_mode = os.stat(target).st_mode & 0o777
    except OSError:
        original_mode = 0o644

    fd, tmp_path = tempfile.mkstemp(
        prefix=target.name + ".",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        os.chmod(tmp_path, original_mode)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
