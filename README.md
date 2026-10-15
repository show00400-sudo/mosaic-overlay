# Mosaic Overlay

화면의 얼굴을 YuNet으로 검출해 실시간 모자이크 처리하는 오버레이 (Windows).

## 실행 (소스)
```
pip install -r requirements.txt
python main.py
```
`face_detection_yunet_2023mar.onnx` 를 `main.py` 와 같은 폴더에 두세요
([opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)).

## 업데이트(배포) 방법
1. `version.py` 의 `__version__` 올리기 (예: `1.0.1`)
2. 커밋 후 태그 푸시
   ```
   git commit -am "v1.0.1"
   git tag v1.0.1
   git push origin main --tags
   ```
3. GitHub Actions가 exe를 빌드해 Releases에 `MosaicOverlay.exe` 업로드
4. 사용자가 프로그램을 켜면 새 버전을 감지해 **UPDATE** 버튼이 표시되고, 클릭하면 자동 교체 후 재시작

> 최초 1회: `version.py` 의 `GITHUB_REPO` 를 본인 저장소로 변경.
