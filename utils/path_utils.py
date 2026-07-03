import os
import re


_WINDOWS_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")


def normalize_platform_path(path: str) -> str:
    if not path or os.name == "nt":
        return path

    match = _WINDOWS_DRIVE_RE.match(path)
    if not match:
        return path

    drive = match.group(1).lower()
    rest = match.group(2).replace("\\", "/").lstrip("/")
    return f"/mnt/{drive}/{rest}"
