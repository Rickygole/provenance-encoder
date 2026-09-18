"""Refuse to run against cloud-synced storage.

MIMIC-III is PhysioNet-credentialed. The Credentialed Health Data Use
Agreement's clause 4 requires reasonable and prudent care over electronic
security, and PhysioNet's guidance of 24 September 2025 strongly recommends
locally deployed models over any cloud service. A path that resolves inside a
sync client's watched folder -- Google Drive, Dropbox, OneDrive, iCloud Drive,
Box, or a Colab Drive mount -- is a path a third party receives a copy of the
moment something is written there, whether or not anyone chooses to "share"
it. This check exists to be called before that write happens, not after.

Deliberately small: a case-insensitive name match on the resolved absolute
path, not a network probe or filesystem introspection. It is a floor, not a
proof of privacy -- it catches the well-known sync-client directory names
this project has actually run into, not every possible remote mount.

Usage, in any script that writes MIMIC-derived files or checkpoints trained
on them::

    from accord.cloud_paths import assert_local_only
    OUT = assert_local_only(os.environ.get("ACCORD_MY_OUT", DEFAULT_LOCAL_OUT))
"""

from __future__ import annotations

import os
import pathlib

# The one environment variable every clinical script reads the credentialed
# directory from; see assert_local_only for how this check honours it.
LOCAL_ONLY_ENV = "MIMIC_LOCAL_ONLY_DIR"

# Case-insensitive substrings that name a cloud-sync client's local mirror.
# "content/drive" and "mydrive" catch the Colab `drive.mount('/content/drive')`
# convention specifically; "cloudstorage" catches macOS's unified sync
# location for Google Drive, Dropbox, OneDrive, Box and iCloud Drive alike
# (~/Library/CloudStorage/<Provider>-<account>/); the rest catch the older,
# non-unified desktop clients that still exist on other machines.
CLOUD_MARKERS = (
    "content/drive",
    "mydrive",
    "google drive",
    "googledrive",
    "cloudstorage",
    "dropbox",
    "onedrive",
    "icloud drive",
    "icloud~",
    "box sync",
    "box/",
)


class CloudStoragePathError(RuntimeError):
    """Raised when a path meant for credentialed-only local storage resolves
    inside what looks like a cloud-synced directory."""


def _cloud_marker(resolved: pathlib.Path):
    """The first CLOUD_MARKERS entry found in `resolved`, or None."""
    haystack = str(resolved).lower()
    for marker in CLOUD_MARKERS:
        if marker in haystack:
            return marker
    return None


def assert_local_only(path) -> pathlib.Path:
    """Raise CloudStoragePathError if `path` resolves inside cloud-synced
    storage. Returns the resolved path otherwise, so a caller can chain it:

        OUT = assert_local_only(OUT)

    The credentialed directory is relocatable through MIMIC_LOCAL_ONLY_DIR
    (default ~/mimic_local_only). Relocating it must not become a way around
    this check, so when that variable is set the directory it names is held
    to the same rule as `path`: a run whose credentialed root sits in a sync
    folder is refused even if the particular path passed in happens not to.
    With the variable unset the behaviour is exactly the path check alone.
    """
    resolved = pathlib.Path(path).expanduser().resolve()
    local_root = os.environ.get(LOCAL_ONLY_ENV)
    checks = [(resolved, str(resolved))]
    if local_root:
        root_resolved = pathlib.Path(local_root).expanduser().resolve()
        checks.append((root_resolved, f"{LOCAL_ONLY_ENV}={root_resolved}"))
    for candidate, shown in checks:
        marker = _cloud_marker(candidate)
        if marker is not None:
            raise CloudStoragePathError(
                f"refusing to run: {shown} resolves inside what looks like "
                f"a cloud-synced directory (matched {marker!r}). MIMIC-derived "
                f"files and checkpoints trained on them must stay on "
                f"credentialed-only local storage -- never Google Drive, "
                f"Dropbox, OneDrive, iCloud, or a Colab Drive mount. Point this "
                f"at a local path instead, e.g. under ${LOCAL_ONLY_ENV} "
                f"(default ~/mimic_local_only/)."
            )
    return resolved
