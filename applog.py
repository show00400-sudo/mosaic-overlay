"""진단 로그: exe 옆 mosaic_log.txt 에 기록 (보안 프로그램 충돌 원인 파악용)."""
import faulthandler
import logging
import os
import platform
import sys

from version import __version__

_fault_file = None


def app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.abspath(".")


def setup():
    global _fault_file
    path = os.path.join(app_dir(), "mosaic_log.txt")
    logging.basicConfig(
        filename=path,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        encoding="utf-8",
    )

    def hook(exc_type, exc, tb):
        logging.critical("처리되지 않은 예외", exc_info=(exc_type, exc, tb))

    sys.excepthook = hook

    try:  # 네이티브 크래시(세그폴트 등) 스택
        _fault_file = open(os.path.join(app_dir(), "mosaic_crash.txt"), "w")
        faulthandler.enable(file=_fault_file)
    except Exception:
        logging.exception("faulthandler 활성화 실패")

    logging.info("매의 눈 카피바라 v%s", __version__)
    logging.info("OS: %s | Python %s | frozen=%s", platform.platform(), sys.version.split()[0], getattr(sys, "frozen", False))
    logging.info("exe/dir: %s", app_dir())
    return path
