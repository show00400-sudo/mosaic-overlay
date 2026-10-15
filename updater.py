"""GitHub Releases 에서 새 버전 '확인만' 한다. (자동 다운로드/교체는 보안 프로그램 오탐 위험이 커서 제거)"""
import json
import urllib.request

from version import __version__, GITHUB_REPO


def _parse(v):
    return tuple(int(x) for x in v.lstrip("vV").split(".") if x.isdigit())


def check_latest(timeout=5):
    """새 버전이 있으면 (tag, 릴리스 페이지 URL) 반환, 없으면 None."""
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
    return tag, data.get("html_url", f"https://github.com/{GITHUB_REPO}/releases")
