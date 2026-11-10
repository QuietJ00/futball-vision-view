# 고정 위치 카메라 구장·속도 벤치마크

`fixed_mount_v2`는 실제 경기 7개 클립을 사용한다. 각 엔진의 커밋과 경기장 이름, 공개 구장 치수, 자르는 구간을 [fixtures.json](fixtures.json)에 고정한다. 원본과 센서의 SHA-256은 [sources.json](sources.json)에 기록한다.

| 클립 | 구장 | 길이×폭 | 촬영 조건 |
|---|---|---|---|
| `anzhi_c1`, `stromsgodset_c1` | Alfheim Stadion | 105×68 m | 데이터 논문에 명시된 관중석 상단 고정 단일 카메라. ZXY 속도 평가 |
| `dfl_2e57b9`, `dfl_573e61` | RheinEnergieSTADION | 105×68 m | 상단 스카우팅 카메라의 연속 팬. `573e61`은 명암 대비가 큰 장면 |
| `dfl_08fd33` | BORUSSIA-PARK | 105×68 m | 상단 스카우팅 카메라의 연속 팬·줌 |
| `dfl_0bfacc` | MHP Arena (촬영 당시 Mercedes-Benz Arena) | 105×68 m | 상단 스카우팅 카메라의 연속 팬 |
| `dfl_121364` | Stadion An der Alten Försterei | 105×68 m | 상단 스카우팅 카메라가 하프라인에서 페널티박스 방향으로 팬 |

촬영 위치가 유지되는 단일 카메라 샷을 사용한다. 같은 축의 팬·틸트는 포함하고, 줌이 보이는 구간은 고정 화면과 구분한다. 이동 촬영, 컷·리플레이, 스티칭, 표식 없는 구장은 본평가에서 제외한다. 이전의 `charrua_pan`, `zenit_pan`은 고정 마운트를 확인하기 어려운 하단 촬영이고, `unmarked_overhead`는 구장선 없는 대조군이다. 세 구간은 `dfl_0bfacc`, `dfl_573e61`, `dfl_121364`로 교체했다. 이전 구성을 재현할 때는 GitHub의 기존 `benchmark/field-benchmark` 브랜치의 원본 파일과 별도 결과 폴더를 사용한다.

DFL 영상은 간판·좌석으로 구장을 식별하고, 6초 연속 샷과 서로 다른 깊이의 관중석·잔디 특징점의 프레임 간 움직임을 확인했다. `fixed_viewpoint_consistent`는 영상으로 확인한 고정 시점 적합성을 뜻한다. 실제 설치 좌표와 렌즈 줌 로그는 공개 자료에 없다. Alfheim은 출처에서 고정 위치를 명시한다. 엔진 보정 결과와 무관하게 평가 대상을 선정했다.

치수는 클립별로 두 엔진에 직접 입력한다. 이 구성에서 일곱 클립의 공개 치수가 우연히 모두 105×68 m다. 별도 측정값이 없는 세부 라인 규격은 엔진의 표준 구장선을 사용한다. 입력 치수와 채점용 수동 표식은 분리한다.

- Alfheim · Pettersen 외, *Soccer Video and Player Position Dataset*. 익명 센서 자료는 비상업 연구용이다. [데이터 및 사용 조건](https://datasets.simula.no/alfheim/) · [촬영·치수 논문](https://home.simula.no/~paalh/publications/files/mmsys2014-dataset.pdf)
- DFL · Roboflow 예제에서 연결한 스카우팅 원본을 연구 평가에 사용한다. [영상 출처](https://github.com/roboflow/sports/blob/main/examples/soccer/setup.sh)
- DFL 구장 치수 · [RheinEnergieSTADION](https://www.koelnersportstaetten.de/wp-content/uploads/Factsheet-2022.pdf) · [BORUSSIA-PARK](https://www.transfermarkt.de/borussia-monchengladbach/stadion/verein/18) · [MHP Arena](https://www.mhparena-stuttgart.de/arena/daten-fakten) · [An der Alten Försterei](https://www.stadion-an-der-alten-foersterei.de/fakten-zahlen/)

## 실행

프로젝트 루트에서 Python 3.12 이상, 기존 프로젝트 의존성과 `work/models/yolo26m.pt`가 있는 환경을 사용한다. 영상·프레임 캐시는 `work/external_benchmark`, 결과는 `outputs/external_benchmark`에 기록한다. 각 영상의 디코딩과 YOLO 검출, 초기 ORB 카메라 추적을 공유한다. `solve`는 각 Git 커밋을 격리된 디렉터리에 추출한다.

```bash
python -B experiments/field_benchmark/bench.py fetch
python -B experiments/field_benchmark/bench.py prepare
python -B experiments/field_benchmark/bench.py solve --engine pivotanchor
python -B experiments/field_benchmark/bench.py solve --engine turflens
python -B experiments/field_benchmark/bench.py score
PYTHONPATH="$PWD" python -B -m unittest discover -s experiments/field_benchmark
```

각 단계에 `--clip ID`를 반복해 일부만 실행할 수 있다.

### 후보 커밋 비교와 비교 영상

새 엔진 커밋을 비교할 때는 원본 manifest를 고치지 않는다. `fixtures.json`, `sources.json`, `reference.json`을 결과 폴더의 `manifest/`에 복사하고 `engines`의 SHA만 바꾼 뒤, 네 단계에 같은 `--manifest`를 넘긴다. 러너는 Git 커밋만 읽으므로 비교할 커밋은 이 저장소(또는 `origin`)에 있어야 한다.

```bash
OUT=outputs/external_benchmark/run-mine
mkdir -p "$OUT/manifest" && cp experiments/field_benchmark/{fixtures,sources,reference}.json "$OUT/manifest/"
# $OUT/manifest/fixtures.json 의 engines 를 비교할 SHA로 수정
python -B experiments/field_benchmark/bench.py solve --manifest "$OUT/manifest/fixtures.json" --engine pivotanchor --output "$OUT"
python -B experiments/field_benchmark/bench.py solve --manifest "$OUT/manifest/fixtures.json" --engine turflens --output "$OUT"
python -B experiments/field_benchmark/bench.py score --manifest "$OUT/manifest/fixtures.json" --output "$OUT"
python -B experiments/field_benchmark/render_compare.py "$OUT" "$OUT/benchmark_compare.mp4"
```

`render_compare.py`는 채점이 끝난 결과 폴더로 두 엔진을 나란히 놓은 1080p 영상을 만든다. 각 엔진이 투영한 구장선, 수동 표식(표식 프레임에서 잠시 정지), Alfheim 선수 A·B의 센서 대비 속도 그래프, 클립별 지표와 판정, 마지막 종합표가 들어간다. 한글 글꼴(macOS Apple SD Gothic Neo, Noto Sans CJK, 맑은 고딕 중 하나)이 필요하다. 프레임 캐시가 다른 위치면 `--cache`를 지정한다.
 `score --strict`는 실패한 기준이 있으면 결과 파일을 저장하고 상태 코드 1을 반환한다. 기준은 manifest의 `policy`에서 변경할 수 있다. 배포용 합격선은 별도의 다구장 검증으로 설정해야 한다.

## 채점

- **좌표 커버리지**는 전체 프레임 중 모델이 좌표를 지원하는 비율이다. **구장선 정확도**는 `reference.json`의 수동 표식에서 동일한 구장선까지의 픽셀 거리를 잰다. 180° 대칭은 클립당 한 번 정렬한다. 조건을 통과한 프레임 수를 전체 표식 프레임 수로 나누는 지표도 기록해 거부로 점수를 올릴 수 없게 한다.
- **고정점 오차**는 대응하는 구장 랜드마크를 역투영해 미터 단위로 비교한다. 현재 미터 단위 정답 표식은 Alfheim 두 클립에 있고, DFL은 구장선의 픽셀 오차를 채점한다.
- **잔디 드리프트**는 선수 박스를 제외하고 SIFT로 1초 간격의 잔디 특징점을 매칭한다. 첫 프레임의 점을 구장 좌표로 역투영한 후 다음 프레임의 위치를 예측해 이미지 오차를 잰다. 두 엔진이 공통 지원한 쌍도 따로 집계한다.
- **속도**는 동일한 영상 선수 추적에서 발 좌표를 각 엔진으로 미터로 바꾸고 공용 속도 계산기를 적용한다. Alfheim 영상 파일의 시작 시각과 센서의 20Hz m/s 값을 정렬한 뒤 km/h로 변환한다. 0.15초를 넘는 센서 공백은 보간하지 않는다. 선수 위치는 영상의 직접 표식 두 점과 대조한다. 선별한 영상 표식과 센서 위치에 맞춘 시차는 Anzhi +3.6초, Strømsgodset +3.4초다. 이 시차는 엔진 출력이나 속도 오차로 맞추지 않았다. 일치 여부는 별도 프레임 30·60에서 검사한다. **수동 구장 보정 대조군**은 카메라가 고정된 Alfheim의 첫 프레임에 표기한 구장 교차점으로 만든다. 자동 엔진 점수와 분리한다.
- 속도는 MAE, RMSE, 편향, 절대오차 p95, 동일한 프레임 지원 범위에서의 11프레임 지속 최고속도와 발생 시각 오차를 기록한다. 센서가 있는 프레임과 모델이 속도를 내보낸 프레임의 분모를 모두 저장한다. 속도 샘플 간 상관이 있으므로 프레임 수를 독립 경기 수로 해석할 수 없다.
- `report.json`에 커밋, 표식별·프레임쌍별 요약과 게이트를 기록한다. `speed_trace.csv`에는 동일 시각의 센서와 예측을 적고, 자동 보정이 거부하면 예측란을 비워 둔다. 경과시간은 동시 실행의 영향을 받는다.

현재 파일럿의 속도 검증 범위는 Alfheim의 두 경기와 3초씩의 짧은 구간이다. 장시간 추적, 아마추어 풀매치, 공중 볼 속도와 오프사이드 판정 정확도는 이 평가 범위에 포함되지 않는다.
