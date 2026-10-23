# 매의 눈 카피바라 (Mosaic Overlay)

화면의 얼굴을 YuNet으로 검출해 실시간 모자이크 처리하는 오버레이 (Windows).

## 실행 (소스)
```
pip install -r requirements.txt
python main.py
```
`face_detection_yunet_2023mar.onnx` 를 `main.py` 와 같은 폴더에 두세요
([opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)).

## 배포 방법
1. `version.py` 의 `__version__` 올리기 (예: `1.1.1`)
2. 커밋 후 태그 푸시
   ```
   git commit -am "v1.1.1"
   git tag v1.1.1
   git push origin main --tags
   ```
3. GitHub Actions가 빌드해 Releases에 `CapybaraEye.zip` 업로드
4. 사용자가 프로그램을 켜면 새 버전을 확인해 **NEW** 버튼이 표시되고, 클릭하면 릴리스 페이지가 열립니다.
   (자동 다운로드/교체는 보안 프로그램 오탐 때문에 사용하지 않음)

## 보안 프로그램 충돌 시
- 압축을 풀고 `CapybaraEye.exe` 실행 (폴더째 보관)
- 실행 폴더에 생기는 `mosaic_log.txt`, `mosaic_crash.txt` 를 개발자에게 전달
- 백신이 차단하면 폴더를 예외 등록하거나 오탐 신고
- 화면 캡처가 검게 나오면 자동으로 dxcam 방식으로 전환을 시도하며 로그에 기록됨

## 속도 설정
상단 **SPEED** 버튼으로 전환 (FAST → NORMAL → ACCURATE)
| 설정 | 검출 화면 너비 | 추적기 | 특징 |
|---|---|---|---|
| FAST | 640px | KCF | 가장 빠름, 작은 얼굴을 놓칠 수 있음 |
| NORMAL (기본) | 960px | CSRT | 균형 |
| ACCURATE | 1440px | CSRT | 정확하지만 느림 |

`mosaic_log.txt` 에 120프레임마다 평균 처리 시간(ms)이 기록됩니다.
