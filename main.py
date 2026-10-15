import sys
import os
import cv2
import csv
import time
import ctypes
import threading
import logging
import numpy as np
import mss

from version import __version__
import updater
import applog


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel
from PyQt5.QtGui import QPainter, QImage, QColor, QPen, QDesktopServices
from PyQt5.QtCore import Qt, QTimer, QRect, QPoint, QUrl, pyqtSignal


MODEL_PATH = resource_path("face_detection_yunet_2023mar.onnx")
SAVE_DIR = "training_data"


class YuNetMosaicOverlay(QWidget):
    update_found = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()

        if not os.path.exists(MODEL_PATH):
            logging.critical("모델 파일 없음: %s", MODEL_PATH)
            print(f"[ERROR] 모델 파일 없음: {MODEL_PATH}")
            print("face_detection_yunet_2023mar.onnx 파일을 코드와 같은 폴더에 넣어주세요.")
            sys.exit(1)

        os.makedirs(SAVE_DIR, exist_ok=True)

        self.sct = mss.MSS()
        self.monitor = self.sct.monitors[1]
        logging.info("모니터: %s", self.monitor)

        self.black_frames = 0
        self.dxcam = None
        self.capture_backend = "mss"

        self.screen_w = self.monitor["width"]
        self.screen_h = self.monitor["height"]

        self.detector = cv2.FaceDetectorYN_create(
            MODEL_PATH,
            "",
            (self.screen_w, self.screen_h),
            score_threshold=0.55,
            nms_threshold=0.3,
            top_k=5000
        )

        self.mosaic_on = True
        self.train_mode = False

        self.faces = []
        self.mosaic_images = []

        self.last_faces = []
        self.last_seen_time = []
        self.keep_seconds = 2.00

        self.dragging = False
        self.drag_start = QPoint()
        self.drag_end = QPoint()

        self.latest_frame = None
        self.update_url = None

        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )

        self.trackers = []
        self.tracker_rects = []
        self.detect_every = 2
        self.frame_count = 2.0

        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)

        self.setGeometry(
            self.monitor["left"],
            self.monitor["top"],
            self.screen_w,
            self.screen_h
        )

        # -------------------------
        # Buttons
        # -------------------------
        self.btn_mosaic = QPushButton("MOSAIC ON", self)
        self.btn_mosaic.setGeometry(20, 20, 140, 45)
        self.btn_mosaic.clicked.connect(self.toggle_mosaic)

        self.btn_train = QPushButton("TRAIN OFF", self)
        self.btn_train.setGeometry(170, 20, 140, 45)
        self.btn_train.clicked.connect(self.toggle_train_mode)

        self.btn_exit = QPushButton("EXIT", self)
        self.btn_exit.setGeometry(320, 20, 100, 45)
        self.btn_exit.clicked.connect(self.close)

        self.btn_update = QPushButton("UPDATE", self)
        self.btn_update.setGeometry(980, 20, 160, 45)
        self.btn_update.clicked.connect(self.do_update)
        self.btn_update.hide()

        btn_style = """
            QPushButton {
                background-color: rgba(30, 30, 30, 230);
                color: white;
                font-size: 15px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: rgba(80, 80, 80, 240);
            }
        """

        self.btn_mosaic.setStyleSheet(btn_style)
        self.btn_train.setStyleSheet(btn_style)
        self.btn_exit.setStyleSheet(btn_style)
        self.btn_update.setStyleSheet(btn_style.replace("30, 30, 30", "20, 120, 60"))

        self.label = QLabel("TRAIN OFF | 놓친 얼굴은 TRAIN ON 후 드래그 저장", self)
        self.label.setGeometry(440, 25, 520, 35)
        self.label.setStyleSheet("""
            QLabel {
                background-color: rgba(30, 30, 30, 180);
                color: white;
                font-size: 14px;
                padding-left: 10px;
                border-radius: 6px;
            }
        """)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(25)

        # 시작 시 백그라운드로 GitHub 새 버전 확인
        self.update_found.connect(self.on_update_found)
        threading.Thread(target=self._check_update, daemon=True).start()

    # -------------------------
    # Update
    # -------------------------
    def _check_update(self):
        result = updater.check_latest()
        if result:
            self.update_found.emit(*result)

    def on_update_found(self, tag, url):
        self.update_url = url
        self.btn_update.setText(f"NEW {tag}")
        self.btn_update.show()
        self.label.setText(f"새 버전 {tag} 있음 (현재 v{__version__}) - 버튼을 누르면 다운로드 페이지가 열립니다")

    def do_update(self):
        if self.update_url:
            QDesktopServices.openUrl(QUrl(self.update_url))

    def showEvent(self, event):
        super().showEvent(event)

        try:
            hwnd = int(self.winId())
            WDA_EXCLUDEFROMCAPTURE = 0x00000011
            ok = ctypes.windll.user32.SetWindowDisplayAffinity(
                hwnd,
                WDA_EXCLUDEFROMCAPTURE
            )
            logging.info("SetWindowDisplayAffinity 결과: %s", ok)
            if not ok:
                logging.warning("캡처 제외 설정 실패 - 모자이크가 캡처에 다시 잡혀 깜빡일 수 있음")
        except Exception:
            logging.exception("SetWindowDisplayAffinity 예외")

    def toggle_mosaic(self):
        self.mosaic_on = not self.mosaic_on

        if self.mosaic_on:
            self.btn_mosaic.setText("MOSAIC ON")
        else:
            self.btn_mosaic.setText("MOSAIC OFF")
            self.faces.clear()
            self.mosaic_images.clear()
            self.last_faces.clear()
            self.last_seen_time.clear()

        self.update()

    def create_tracker(self):
        if hasattr(cv2, "legacy"):
            return cv2.legacy.TrackerCSRT_create()
        return cv2.TrackerCSRT_create()

    def toggle_train_mode(self):
        self.train_mode = not self.train_mode

        if self.train_mode:
            self.btn_train.setText("TRAIN ON")
            self.label.setText("TRAIN ON | 놓친 얼굴을 마우스로 드래그하면 저장됩니다")
        else:
            self.btn_train.setText("TRAIN OFF")
            self.label.setText("TRAIN OFF | 놓친 얼굴은 TRAIN ON 후 드래그 저장")

        self.update()

    def make_mosaic(self, img, scale=0.045):
        h, w = img.shape[:2]

        if h < 5 or w < 5:
            return img

        small_w = max(1, int(w * scale))
        small_h = max(1, int(h * scale))

        small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        return mosaic

    def mousePressEvent(self, event):
        if not self.train_mode:
            return

        # 버튼 영역 클릭은 드래그 저장으로 처리하지 않음
        if event.y() < 80:
            return

        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_start = event.pos()
            self.drag_end = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if self.train_mode and self.dragging:
            self.drag_end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if not self.train_mode:
            return

        if event.button() == Qt.LeftButton and self.dragging:
            self.dragging = False
            self.drag_end = event.pos()

            rect = self.get_drag_rect()

            if rect.width() > 15 and rect.height() > 15:
                self.save_training_sample(rect)

            self.update()

    def get_drag_rect(self):
        x1 = min(self.drag_start.x(), self.drag_end.x())
        y1 = min(self.drag_start.y(), self.drag_end.y())
        x2 = max(self.drag_start.x(), self.drag_end.x())
        y2 = max(self.drag_start.y(), self.drag_end.y())

        return QRect(x1, y1, x2 - x1, y2 - y1)

    def save_training_sample(self, rect):
        if self.latest_frame is None:
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ms = int((time.time() % 1) * 1000)

        img_name = f"sample_{timestamp}_{ms}.jpg"
        csv_name = "labels.csv"

        img_path = os.path.join(SAVE_DIR, img_name)
        csv_path = os.path.join(SAVE_DIR, csv_name)

        frame = self.latest_frame.copy()
        cv2.imwrite(img_path, frame)

        x = rect.x()
        y = rect.y()
        w = rect.width()
        h = rect.height()

        file_exists = os.path.exists(csv_path)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(["image", "class", "x", "y", "w", "h"])

            writer.writerow([img_name, "face", x, y, w, h])

        self.label.setText(f"SAVED | face box: x={x}, y={y}, w={w}, h={h}")
        print(f"[SAVE] {img_path}, box=({x}, {y}, {w}, {h})")

    def iou(self, a, b):
        ax1, ay1 = a.x(), a.y()
        ax2, ay2 = a.x() + a.width(), a.y() + a.height()

        bx1, by1 = b.x(), b.y()
        bx2, by2 = b.x() + b.width(), b.y() + b.height()

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)

        inter = iw * ih
        union = a.width() * a.height() + b.width() * b.height() - inter

        if union <= 0:
            return 0

        return inter / union

    def merge_with_tracking(self, detected_rects):
        now = time.time()

        updated_faces = []
        updated_times = []

        for rect in detected_rects:
            updated_faces.append(rect)
            updated_times.append(now)

        for old_rect, old_time in zip(self.last_faces, self.last_seen_time):
            if now - old_time <= self.keep_seconds:
                matched = False

                for new_rect in detected_rects:
                    if self.iou(old_rect, new_rect) > 0.25:
                        matched = True
                        break

                if not matched:
                    updated_faces.append(old_rect)
                    updated_times.append(old_time)

        self.last_faces = updated_faces
        self.last_seen_time = updated_times

        return updated_faces

    def grab_frame(self):
        try:
            if self.dxcam is not None:
                frame = self.dxcam.grab()
                if frame is None:
                    return None
            else:
                screenshot = np.array(self.sct.grab(self.monitor))
                frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        except Exception:
            logging.exception("화면 캡처 실패 (backend=%s)", self.capture_backend)
            self.capture_failed()
            return None

        # 보안 프로그램이 캡처를 막으면 전체가 검은 화면으로 들어온다
        if frame.max() == 0:
            self.black_frames += 1
            if self.black_frames == 40:
                logging.warning("캡처 프레임이 계속 검은색 (backend=%s) - 보안 프로그램의 캡처 차단 의심", self.capture_backend)
                self.capture_failed()
        else:
            self.black_frames = 0
        return frame

    def capture_failed(self):
        """mss 가 막히면 dxcam(Desktop Duplication) 으로 한 번 전환 시도."""
        self.black_frames = 0
        if self.dxcam is not None or getattr(self, "dxcam_tried", False):
            self.label.setText("화면 캡처가 차단된 것 같습니다. mosaic_log.txt 를 확인하세요")
            return
        self.dxcam_tried = True
        try:
            import dxcam
            self.dxcam = dxcam.create(output_color="BGR")
            self.dxcam.start(target_fps=40)
            self.capture_backend = "dxcam"
            logging.info("캡처 방식을 dxcam 으로 전환")
        except Exception:
            self.dxcam = None
            logging.exception("dxcam 전환 실패")
            self.label.setText("화면 캡처가 차단된 것 같습니다. mosaic_log.txt 를 확인하세요")

    def update_frame(self):
        if not self.mosaic_on:
            return

        frame = self.grab_frame()
        if frame is None:
            return

        self.latest_frame = frame.copy()
        self.frame_count += 1

        detected_rects = []

        # 1) 매 프레임 tracker로 먼저 위치 추적
        tracker_rects = []
        alive_trackers = []

        for tracker in self.trackers:
            ok, box = tracker.update(frame)

            if ok:
                x, y, w, h = [int(v) for v in box]

                if w > 10 and h > 10:
                    rect = QRect(x, y, w, h)
                    tracker_rects.append(rect)
                    alive_trackers.append(tracker)

        self.trackers = alive_trackers

        # 2) 몇 프레임마다 YuNet으로 재검출해서 tracker 보정
        if self.frame_count % self.detect_every == 0 or len(self.trackers) == 0:
            self.detector.setInputSize((frame.shape[1], frame.shape[0]))
            _, faces = self.detector.detect(frame)

            if faces is not None:
                for face in faces:
                    x, y, w, h = face[:4].astype(int)
                    score = float(face[-1])

                    if score < 0.50:
                        continue

                    pad_x = int(w * 0.50)
                    pad_y = int(h * 0.60)

                    x1 = max(x - pad_x, 0)
                    y1 = max(y - pad_y, 0)
                    x2 = min(x + w + pad_x, frame.shape[1])
                    y2 = min(y + h + pad_y, frame.shape[0])

                    detected_rects.append(QRect(x1, y1, x2 - x1, y2 - y1))

            # 3) 검출된 얼굴로 tracker 새로 생성
            self.trackers = []

            for rect in detected_rects:
                tracker = self.create_tracker()
                tracker.init(
                    frame,
                    (
                        rect.x(),
                        rect.y(),
                        rect.width(),
                        rect.height()
                    )
                )
                self.trackers.append(tracker)

            active_rects = detected_rects

        else:
            # 검출하지 않는 프레임에서는 tracker 위치 사용
            active_rects = tracker_rects

        # 4) 그래도 혹시 놓친 경우 기존 위치 유지
        tracked_rects = self.merge_with_tracking(active_rects)

        new_faces = []
        new_images = []

        for rect in tracked_rects:
            x1 = max(rect.x(), 0)
            y1 = max(rect.y(), 0)
            x2 = min(x1 + rect.width(), frame.shape[1])
            y2 = min(y1 + rect.height(), frame.shape[0])

            face_img = frame[y1:y2, x1:x2]

            if face_img.size == 0:
                continue

            mosaic_img = self.make_mosaic(face_img)

            rgb = cv2.cvtColor(mosaic_img, cv2.COLOR_BGR2RGB)

            qimg = QImage(
                rgb.data,
                rgb.shape[1],
                rgb.shape[0],
                rgb.strides[0],
                QImage.Format_RGB888
            ).copy()

            new_faces.append(QRect(x1, y1, x2 - x1, y2 - y1))
            new_images.append(qimg)

        self.faces = new_faces
        self.mosaic_images = new_images

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)

        if self.mosaic_on:
            for rect, img in zip(self.faces, self.mosaic_images):
                painter.drawImage(rect, img)

        if self.train_mode:
            pen = QPen(QColor(255, 80, 80, 230))
            pen.setWidth(3)
            painter.setPen(pen)

            for rect in self.faces:
                painter.drawRect(rect)

            if self.dragging:
                painter.drawRect(self.get_drag_rect())


if __name__ == "__main__":
    applog.setup()
    try:
        app = QApplication(sys.argv)

        overlay = YuNetMosaicOverlay()
        overlay.showFullScreen()
        logging.info("오버레이 표시 완료")

        code = app.exec_()
        logging.info("정상 종료 (code=%s)", code)
        sys.exit(code)
    except SystemExit:
        raise
    except Exception:
        logging.exception("시작 중 예외")
        raise
