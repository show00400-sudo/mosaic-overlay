"""GitHub Releases 기반 업데이트 확인/적용."""
import json
import os
import subprocess
import sys
import tempfile
import urllib.request

from version import __version__, GITHUB_REPO, ASSET_NAME


def _parse(v):
    return tuple(int(x) for x in v.lstrip("vV").split(".") if x.isdigit())


def check_latest(timeout=5):
    """새 버전이 있으면 (tag, download_url) 반환, 없으면 None."""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
    req = urllib.request.Request(url, headers={"User-Agent": "mosaic-overlay"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
    except Exception:
        return None

    tag = data.get("tag_name", "")
    if not tag or _parse(tag) <= _parse(__version__):
        return None

    for asset in data.get("assets", []):
        if asset.get("name") == ASSET_NAME:
            return tag, asset["browser_download_url"]
    return None


def apply_update(download_url):
    """새 exe를 받아 종료 후 교체하고 재실행. exe(frozen) 실행 중일 때만 동작."""
    if not getattr(sys, "frozen", False):
        return False

    current = sys.executable
    new_path = current + ".new"
    urllib.request.urlretrieve(download_url, new_path)

    bat = os.path.join(tempfile.gettempdir(), "mosaic_update.bat")
    with open(bat, "w", encoding="utf-8") as f:
        f.write(
            "@echo off\n"
            "ping 127.0.0.1 -n 3 > nul\n"
            f'move /y "{new_path}" "{current}"\n'
            f'start "" "{current}"\n'
            'del "%~f0"\n'
        )
    subprocess.Popen(["cmd", "/c", bat], creationflags=0x08000000)
    return True
