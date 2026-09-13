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

import pathlib

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


def assert_local_only(path) -> pathlib.Path:
    """Raise CloudStoragePathError if `path` resolves inside cloud-synced
    storage. Returns the resolved path otherwise, so a caller can chain it:

        OUT = assert_local_only(OUT)
    """
    resolved = pathlib.Path(path).expanduser().resolve()
    haystack = str(resolved).lower()
    for marker in CLOUD_MARKERS:
        if marker in haystack:
            raise CloudStoragePathError(
                f"refusing to run: {resolved} resolves inside what looks like "
                f"a cloud-synced directory (matched {marker!r}). MIMIC-derived "
                f"files and checkpoints trained on them must stay on "
                f"credentialed-only local storage -- never Google Drive, "
                f"Dropbox, OneDrive, iCloud, or a Colab Drive mount. Point this "
                f"at a local path instead, e.g. under ~/mimic_local_only/."
            )
    return resolved
