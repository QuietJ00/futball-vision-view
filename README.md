# futball-vision

아마추어 축구 경기 영상에서 선수 한 명을 찍으면, 속도 분석이 들어간 중계 스타일 하이라이트를 만들어 주는 로컬 웹앱이에요. 화면은 QuarterMate 다크 테마를 따릅니다.

## 기능
- 유튜브 링크(구간 지정) 또는 영상 파일 입력. 90초 이하 한 컷만 받아요.
- 선수 검출(YOLO26m)과, 클릭 한두 번으로 시작하는 선수 추적. 다른 선수로 넘어갈 수 없는 이동은 막고, 헷갈리는 구간은 표시해요.
- 카메라 움직임 추적과 경기장 보정
  - 저장된 구장에서 찍은 영상은 자동으로 맞춰요.
  - 처음 보는 구장이나 촬영 위치는 기준점 4개를 직접 찍어요.
- 경기장 위 실제 거리 기준의 속도(km/h)·거리 분석
- 렌더링: 스코어버그, 선수 마커와 실시간 속도, 이동 경로, 하단 스탯 패널, GOAL / ASSIST / SPRINT 카드, 오프닝·엔딩 전환 화면
- 경기장 보정이 안 될 때 쓰는 "속도 없이 만들기"
- 경기장 인식 모델 학습용 라벨링 화면(`/label`)

## 설치
```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
ffmpeg는 `imageio-ffmpeg`에 들어 있는 것을 써요. YOLO 가중치는 처음 실행할 때 자동으로 받아요.

## 실행
`하이라이트메이커.command`를 더블클릭하거나, 아래 명령을 실행하세요.
```bash
.venv/bin/python -m autohl.app
```
그다음 http://127.0.0.1:8765 를 열면 돼요.

## 구조
| 파일 | 역할 |
|---|---|
| `autohl/video.py` | 다운로드, 디코딩(1080p30) |
| `autohl/camera.py` | 프레임 간 카메라 움직임(키프레임 호모그래피) |
| `autohl/detect.py` | 선수 검출 |
| `autohl/track.py` | 클릭 기반 단일 선수 추적(DP, 속도·외형 비용) |
| `autohl/pitch.py` | 구장 모델, 자동·수동 보정, 라인 추적 |
| `autohl/camfit.py` | 카메라 모델 기반 보정(실험용) |
| `autohl/speed.py` | 경기장 좌표 기반 속도·거리 |
| `autohl/render.py`, `autohl/ui.py` | 하이라이트 그래픽 |
| `autohl/offside.py` | 오프사이드 분석·그래픽(아직 앱에 연결 안 됨) |
| `autohl/pipeline.py`, `autohl/app.py` | 작업 관리, Flask 웹앱 |
| `autohl/venues/noksi` | 고려대 녹지운동장 프리셋(100×60 m, 실측 박스 크기) |
