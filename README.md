# futball-vision

커밋과 입력을 고정해 구장 인식 엔진 두 개를 같은 조건에서 채점하는 평가 CLI(field_benchmark)와, 그 엔진을 사용하는 경기 영상 분석 앱입니다. 엔진·채점기·점검 스크립트는 LLM 에이전트가 작성했습니다. 저는 무엇을 고정할지, 어떤 결과를 실패로 집계할지, 어느 커밋을 받아들일지 정했습니다.

[![Unit tests](https://github.com/QuietJ00/futball-vision-view/actions/workflows/tests.yml/badge.svg)](https://github.com/QuietJ00/futball-vision-view/actions/workflows/tests.yml)

## 먼저 볼 것: 엔진 평가 CLI `field_benchmark`

구장 인식 엔진은 Antigravity가 작성한 PivotAnchor와 Claude가 작성한 TurfLens입니다. 두 에이전트가 서로 다른 영상과 기준으로 개선을 보고해 결과를 비교할 수 없었습니다. [experiments/field_benchmark](experiments/field_benchmark/README.md)는 두 엔진에서 커밋을 골라 동일한 입력과 채점기로 비교하는 명령줄 도구입니다.

### 공정성 장치

| 장치 | 구현 | 막으려는 것 |
|---|---|---|
| 엔진 커밋 고정 | [fixtures.json](experiments/field_benchmark/fixtures.json)의 `engines`에 전체 SHA를 적고, `solve`가 그 커밋을 격리된 폴더에 추출해 실행 | 커밋하지 않은 수정본으로 낸 점수 |
| 입력 고정 | 클립 7개의 구간과 구장 치수는 `fixtures.json`, 원본 영상과 센서 파일의 SHA-256은 [sources.json](experiments/field_benchmark/sources.json) | 엔진마다 다른 영상이나 치수로 채점 |
| manifest 복사본 비교 | 후보 커밋을 볼 때는 세 JSON을 결과 폴더에 복사해 SHA만 바꾸고 `--manifest`로 전달 | 비교하려고 기준 파일을 고치는 일 |
| 커버리지 동시 기록 | 오차와 함께 "좌표를 낸 프레임 / 전체 프레임", "기준을 통과한 표식 프레임 / 전체 표식 프레임"을 기록 | 어려운 프레임을 거부해서 평균 오차만 낮추는 것 |
| 정답 분리 | 수동 표식([reference.json](experiments/field_benchmark/reference.json))과 센서 값은 채점에만 쓰고 엔진 입력에 넣지 않음 | 정답에 맞춘 보정 |

### 실행

```bash
python -B experiments/field_benchmark/bench.py fetch      # 공개 영상과 센서 자료 내려받기, SHA-256 대조
python -B experiments/field_benchmark/bench.py prepare    # 디코딩, 선수 검출 등 공용 입력 캐시
python -B experiments/field_benchmark/bench.py solve --engine pivotanchor
python -B experiments/field_benchmark/bench.py solve --engine turflens
python -B experiments/field_benchmark/bench.py score      # report.json, score.log, speed_trace.csv
```

`fetch`부터 `solve`까지는 영상 내려받기와 YOLO 가중치가 필요합니다. 채점기 자체의 단위 테스트는 영상 없이 실행할 수 있습니다.

```bash
pip install -r requirements-ci.txt
pip install torch==2.15.0
PYTHONPATH="$PWD" python -B -m unittest discover -s experiments/field_benchmark   # 채점기 5개
PYTHONPATH="$PWD" python -B -m unittest discover -s tests                         # 엔진 단위 테스트 75개
```

### 최근 결과 (2026-11-10)

클립마다 180프레임입니다. "좌표"는 엔진이 좌표를 낸 프레임 수, "선 오차"는 수동 표식 대비 구장선 오차의 중앙값 / p95(px), "표식"은 20 px 기준을 통과한 표식 프레임 수입니다. 원본 결과는 [results/2026-11-10](experiments/field_benchmark/results/README.md)에서 확인할 수 있습니다.

**고정 커밋** (PivotAnchor `6a8ab06`, TurfLens `efd1387`)

| 클립 | PivotAnchor 좌표 | 선 오차 | 표식 | 판정 | TurfLens 좌표 | 선 오차 | 표식 | 판정 |
|---|---:|---:|---:|---|---:|---:|---:|---|
| `anzhi_c1` | 0/180 | | 0/4 | 실패 | 0/180 | | 0/4 | 실패 |
| `stromsgodset_c1` | 0/180 | | 0/4 | 실패 | 0/180 | | 0/4 | 실패 |
| `dfl_2e57b9` | 180/180 | 2.8 / 12.8 | 4/4 | 통과 | 180/180 | 2.2 / 6.2 | 4/4 | 통과 |
| `dfl_08fd33` | 0/180 | | 0/4 | 실패 | 0/180 | | 0/4 | 실패 |
| `dfl_0bfacc` | 0/180 | | 0/4 | 실패 | 180/180 | 2.5 / 18.6 | 3/4 | 실패(정확도) |
| `dfl_573e61` | 180/180 | 3.1 / 12.5 | 4/4 | 통과 | 180/180 | 2.5 / 26.0 | 2/4 | 실패(정확도) |
| `dfl_121364` | 0/180 | | 0/4 | 실패 | 180/180 | 4.4 / 36.8 | 2/4 | 실패(정확도) |

**후보 커밋** (PivotAnchor `5f63329`, TurfLens `bf09bdb`)

| 클립 | PivotAnchor 좌표 | 선 오차 | 표식 | 판정 | TurfLens 좌표 | 선 오차 | 표식 | 판정 |
|---|---:|---:|---:|---|---:|---:|---:|---|
| `anzhi_c1` | 180/180 | 3.0 / 10.0 | 4/4 | 통과 | 177/180 | 2.4 / 5.5 | 4/4 | 통과 |
| `stromsgodset_c1` | 180/180 | 2.6 / 10.2 | 4/4 | 실패(속도) | 177/180 | 2.5 / 7.5 | 4/4 | 실패(속도) |
| `dfl_2e57b9` | 180/180 | 2.4 / 5.9 | 4/4 | 통과 | 180/180 | 2.2 / 6.2 | 4/4 | 통과 |
| `dfl_08fd33` | 180/180 | 3.2 / 8.4 | 4/4 | 통과 | 177/180 | 3.6 / 9.2 | 4/4 | 통과 |
| `dfl_0bfacc` | 0/180 | | 0/4 | 실패 | 0/180 | | 0/4 | 실패 |
| `dfl_573e61` | 0/180 | | 0/4 | 실패 | 0/180 | | 0/4 | 실패 |
| `dfl_121364` | 0/180 | | 0/4 | 실패 | 177/180 | 1.6 / 5.1 | 3/4 | 실패(정확도) |

결과 해석

- 통과한 클립은 고정 커밋에서 PivotAnchor 2개, TurfLens 1개였고, 후보 커밋에서는 두 엔진 모두 3개입니다. 일곱 클립을 모두 통과한 엔진은 아직 없습니다.
- `dfl_573e61`은 후보 커밋에서 두 엔진 모두 0/180이 됐습니다. TurfLens는 이전에 180프레임 모두 좌표를 냈지만 표식 4개 중 2개만 기준을 통과했고, 후보 커밋에서는 그 클립의 출력을 거부했습니다. 평균 오차만 보면 놓칠 수 있는 회귀를 커버리지 기준에서 실패로 기록합니다.
- 속도는 Alfheim 두 클립의 센서 값과만 비교합니다. 범위와 한계는 [벤치마크 README](experiments/field_benchmark/README.md)의 채점 항목에 정리했습니다.

### 에이전트에게 주는 실행 Spec

[AGENT_RUN_PROMPT.md](experiments/field_benchmark/AGENT_RUN_PROMPT.md)는 Claude와 Antigravity 어느 쪽에 붙여 넣어도 같은 절차로 벤치마크를 실행하도록 정리한 지시문입니다. 실행 범위, 변경하지 말아야 할 항목(체크섬, 치수, 합격 기준, 작업 트리), 실행 명령, 보고 형식을 명시합니다. "실패를 그대로 적고 숫자는 이번 실행 결과에서 인용한다"는 규칙도 포함합니다.

벤치마크 이전의 점검 라운드(합격 점검 17개와 48개, 오차 주입 12개 차단)는 [docs/evaluation-history.md](docs/evaluation-history.md)에 원본 로그와 함께 정리했습니다.

## 기능
- 유튜브 링크로 구간을 지정하거나 영상 파일을 입력합니다. 90초 이하 한 컷만 받습니다.
- YOLO26m으로 선수를 검출하고, 클릭 한두 번으로 선수 추적을 시작합니다. 추적 대상이 다른 선수로 바뀌는 비현실적인 이동을 막고, 대상을 구분하기 어려운 구간은 표시합니다.
- 카메라 움직임 추적과 경기장 보정
  - 클립을 짧은 구간으로 나눠 구간마다 실제 카메라 한 대의 위치, 높이, 초점거리, 방향을 추정합니다. 페인트선 네 개와 구장 사각형의 가능한 대응을 모두 조사하고, 구간별 추정 위치가 일치할 때만 채택합니다.
  - 페인트 외에 두 가지 조건을 확인합니다. 검출된 사람의 키가 약 1.74 m로 계산되고 발이 경기장 안에 있어야 합니다. 팬 영상에서는 카메라 움직임만으로 구한 초점거리도 사용합니다.
  - 초기 보정 후에는 서클 곡선까지 식별해 클립 전체의 카메라와 렌즈 왜곡을 함께 다시 보정합니다. 보정 결과에서 선이 잔디 위를 프레임당 1 px 넘게 움직이면 해당 결과를 사용하지 않습니다.
  - 기준 구간끼리 카메라 움직임으로 이어 봤을 때 40 px 넘게 어긋나면 좌표를 내지 않고 수동 보정이나 기준 사진을 사용하는 경로로 전환합니다.
  - 스탠드 높이(약 6 m) 영상은 자동으로 보정합니다. 카메라 높이가 3.5 m 미만으로 추정되면 화면에서 페인트선의 원근이 약해 위치를 신뢰하기 어려우므로 결과를 거부합니다. 이런 영상은 기준점을 직접 지정해야 합니다.
  - 저장된 구장의 기준 사진과 잔디 위 특징점을 비교하고, 실제 페인트선과 맞는 프레임을 기준으로 삼습니다.
  - 팬·줌 중에도 카메라 추적이 이어지면 가까운 기준 프레임의 좌표를 사용합니다. 추적이 끊기면 끊어진 지점의 양쪽을 별도로 보정합니다.
  - 확인한 기준에서 멀거나 두 기준의 정합이 어긋난 구간은 화면에 구장선을 표시하지 않습니다. 선수 추적 중 그런 프레임이 있으면 하이라이트에서 속도와 거리를 생략합니다.
  - 처음 보는 구장은 실제 길이와 폭을 입력하고 라인이 잘 보이는 프레임에서 기준점 4개 이상을 직접 지정합니다. 프리셋이 없는 구장의 박스는 표준 치수를 사용하므로 실측과 다를 수 있습니다.
  - 팬 도중 다른 구장선이 나타나면 그 프레임에 기준점을 추가할 수 있습니다. 각 프레임에서 두 방향의 페인트선을 확인하고 기준이 서로 충돌하는 구간을 제외합니다.
- 경기장 위 실제 거리 기준의 속도(km/h)·거리 분석
- 렌더링: 스코어버그, 선수 마커와 실시간 속도, 이동 경로, 하단 스탯 패널, GOAL / ASSIST / SPRINT 카드, 오프닝·엔딩 전환 화면
- 경기장 보정이 안 될 때 쓰는 "속도 없이 만들기"
- 경기장 인식 모델 학습용 라벨링 화면(`/label`)

## 경기장 인식 모델
이 저장소에는 경기장 인식 엔진 두 개를 제공합니다.

| 이름 | 브랜치 | 방식 |
|---|---|---|
| TurfLens | `turflens/pitch-model` | 잔디 움직임(Turf)으로 카메라 경로를 따라가고, 줌과 렌즈 왜곡(Lens)을 보정합니다. 한 대의 카메라에서 줌에 따라 달라지는 초점거리를 구간별로 추정합니다. |
| PivotAnchor | `pivotanchor/pitch-model` | 고정된 삼각대 축(Pivot)을 가정하고, 구장 기준 사진(Anchor)으로 초기 위치를 추정합니다. |

실제 경기 영상 7개 클립과 같은 채점기를 사용해 두 엔진을 비교할 수 있습니다. 실행 방법은 [experiments/field_benchmark](experiments/field_benchmark/README.md)에 있습니다.

## 설치
```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
`imageio-ffmpeg`에 포함된 ffmpeg를 사용합니다. YOLO 가중치는 처음 실행할 때 자동으로 내려받습니다.

## 실행
`하이라이트메이커.command`를 더블클릭하거나 아래 명령을 실행합니다.
```bash
.venv/bin/python -m autohl.app
```
그다음 브라우저에서 http://127.0.0.1:8765 주소를 엽니다.

## 구조
| 파일 | 역할 |
|---|---|
| `autohl/video.py` | 다운로드, 디코딩(1080p30) |
| `autohl/camera.py` | 프레임 간 카메라 움직임(키프레임 호모그래피) |
| `autohl/detect.py` | 선수 검출 |
| `autohl/track.py` | 클릭 기반 단일 선수 추적(DP, 속도·외형 비용) |
| `autohl/pitch.py` | 구장 모델, 자동·수동 보정, 라인 추적 |
| `autohl/field_solver.py` | 페인트·사람·카메라 움직임으로 카메라를 푸는 자동 보정 |
| `autohl/field_geometry.py` | 자동·수동 기준으로 프레임별 구장 좌표 생성, 정합 검사 |
| `autohl/camfit.py` | 카메라 모델 기반 보정(실험용) |
| `autohl/speed.py` | 경기장 좌표 기반 속도·거리 |
| `autohl/render.py`, `autohl/ui.py` | 하이라이트 그래픽 |
| `autohl/offside.py` | 오프사이드 분석·그래픽(아직 앱에 연결 안 됨) |
| `autohl/pipeline.py`, `autohl/app.py` | 작업 관리, Flask 웹앱 |
| `autohl/venues/noksi` | 고려대 녹지운동장 프리셋(100×60 m, 실측 박스 크기) |

녹지운동장 프리셋의 `ref_04`는 14:48 골 클립의 134번 프레임과 수동으로 확인한 보정 결과를 사용합니다. 같은 클립에서 자동 정합을 재현하는 데 쓸 수 있습니다. 프리셋과 촬영 각도가 크게 다르거나 페인트선이 거의 보이지 않는 영상은 직접 보정이 필요할 수 있습니다.

구장 좌표는 `GET /api/job/<id>/field/<frame>`에서 확인할 수 있습니다. `field_to_image`와 `image_to_field`는 3×3 행렬입니다. 구장 좌표에서는 왼쪽 골라인을 기준으로 X를, 중앙에서 가까운 터치라인 방향으로 Y를 미터 단위로 표시합니다. 예를 들어 `?x=1000&y=800`으로 원본 1920×1080 영상의 발 위치를 넘기면 구장 위 위치를 반환합니다. `paint_support`는 선 주변에서 관측된 픽셀의 비율이고 `local_m_per_px`는 해당 위치의 좌표 민감도입니다. 두 값은 실측 오차와 별도로 확인해야 합니다.
