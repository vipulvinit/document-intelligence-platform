"""Object storage for figure images and uploaded files.

`LocalStorage` keeps objects in a folder (free, no setup); the API serves them at /objects/<key>.
Anything with the same three methods (for example a MinIO or S3 client) can replace it later.
Keys look like "articles/PMC1234567/fig-1.jpg".
"""

import mimetypes
import re
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from app.core.config import get_settings

_SAFE_PART = re.compile(r"[^A-Za-z0-9._-]+")


def safe_key(*parts: str) -> str:
    """Build a key from untrusted parts: no "..", no absolute paths, no odd characters."""
    cleaned = [_SAFE_PART.sub("_", p).strip("._") or "x" for p in parts]
    return "/".join(cleaned)


class ObjectStorage(Protocol):
    def put(self, key: str, data: bytes) -> str: ...
    def get(self, key: str) -> bytes | None: ...
    def url(self, key: str) -> str: ...


class LocalStorage:
    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Object key escapes the storage folder: {key!r}")
        return path

    def put(self, key: str, data: bytes) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key

    def get(self, key: str) -> bytes | None:
        try:
            path = self._path(key)
        except ValueError:
            return None
        return path.read_bytes() if path.is_file() else None

    def url(self, key: str) -> str:
        return f"/objects/{key}"


def content_type(key: str) -> str:
    return mimetypes.guess_type(key)[0] or "application/octet-stream"


@lru_cache
def get_storage() -> ObjectStorage:
    s = get_settings()
    if s.storage_backend != "local":
        raise ValueError(f"Unknown STORAGE_BACKEND {s.storage_backend!r} (supported: local)")
    return LocalStorage(s.storage_dir)
