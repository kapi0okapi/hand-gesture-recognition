"""Read single members of a large remote ZIP archive using HTTP range requests.

KITTI raw drives are distributed as 0.5-30 GB archives (camera, LiDAR, GPS/IMU). This module
fetches only the central directory and the requested files, e.g. the few MB of GPS/IMU data.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import requests


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP, backed by Range requests."""

    def __init__(self, url: str, session: requests.Session | None = None):
        self.url = url
        self.session = session or requests.Session()
        head = self.session.head(url, allow_redirects=True, timeout=30)
        head.raise_for_status()
        self.size = int(head.headers["Content-Length"])
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, min(self.size, base + offset))
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        end = self.size - 1 if n is None or n < 0 else min(self.size, self.pos + n) - 1
        response = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=60)
        response.raise_for_status()
        data = response.content
        if response.status_code == 200:  # server ignored the range and sent the whole file
            data = data[self.pos : end + 1]
        self.pos += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def extract_members(url: str, out_dir: Path, contains: str) -> list[Path]:
    """Extracts every member whose name contains ``contains`` (e.g. "/oxts/") into ``out_dir``."""
    raw = io.BufferedReader(HttpRangeFile(url), buffer_size=1 << 20)
    written = []
    with zipfile.ZipFile(raw) as archive:
        for info in archive.infolist():
            if contains in info.filename and not info.is_dir():
                target = out_dir / info.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(info))
                written.append(target)
    return written
