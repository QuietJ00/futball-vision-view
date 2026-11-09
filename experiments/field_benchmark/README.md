# 외부 경기 구장·속도 벤치마크

기존 녹지운동장·중계 평가 캐시와 별개인 실제 경기 영상 7개를 사용한다. 각 엔진의 커밋과 구장 치수, 자르는 구간을 [fixtures.json](fixtures.json)에 고정한다. 원본과 센서의 SHA-256은 [sources.json](sources.json)에 기록한다.

| 그룹 | 영상 | 평가 |
|---|---|---|
| 상단 고정 | Alfheim · Tromsø vs Anzhi, Tromsø vs Strømsgodset | 구장선, 실측 105×68m 기준 고정점, 카메라 흔들림, ZXY 센서 속도 |
| 상단 팬·줌 | DFL Scouting Feed · 2e57b9, 08fd33 | 구장선, 잔디 재투영. 구장 치수는 105×68m 템플릿 가정 |
| 관중석 하단 스트레스 | U-17 뉴질랜드–캐나다, 제니트–첼시 | 빠른 팬, 시야 일부, 줌, 선수·관중 가림 |
| 거부 대조군 | 구장선 없는 아마추어 경기 | 잘못된 구장 좌표 반환 횟수 |

Alfheim 데이터는 익명 센서 자료를 포함하며 비상업 연구용이다. DFL 클립은 Roboflow 예제에서 연결한 원본을 연구 평가에 사용한다. Wikimedia 영상은 자른 구간을 분석에 사용했다. 자료 귀속은 아래 출처를 참고한다.

- Alfheim · Pettersen 외, *Soccer Video and Player Position Dataset*. [데이터 및 사용 조건](https://datasets.simula.no/alfheim/) · [논문](https://home.simula.no/~paalh/publications/files/mmsys2014-dataset.pdf)
- DFL · [Roboflow Soccer 예제의 다운로드 목록](https://github.com/roboflow/sports/blob/main/examples/soccer/setup.sh)
- 뉴질랜드–캐나다 · NaBUru38, CC BY-SA 4.0. [원본](https://commons.wikimedia.org/wiki/File:2018_FIFA_U-17_Women%27s_World_Cup_-_New_Zealand_vs_Canada_-_27.webm)
- 제니트–첼시 · Voltmetro, CC BY-SA 4.0. [원본](https://commons.wikimedia.org/wiki/File:2021-12-08_-_FC_Zenit_Saint_Petersburg_v_Chelsea_F.C._-_Ozdoev_goal.webm)
- 아마추어 경기 · Maximilian Schönherr, CC BY-SA 4.0. [원본](https://commons.wikimedia.org/wiki/File:Amateurfu%C3%9Fball_Torschuss_von_oben.webm)

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
- **고정점 오차**는 실측 치수의 구장 랜드마크를 역투영해 미터 단위로 비교한다. DFL과 Charrúa의 템플릿 치수는 측량값이 없어 미터 지표를 적용하지 않는다.
- **잔디 드리프트**는 선수 박스를 제외하고 SIFT로 1초 간격의 잔디 특징점을 매칭한다. 첫 프레임의 점을 구장 좌표로 역투영한 후 다음 프레임의 위치를 예측해 이미지 오차를 잰다. 두 엔진이 공통 지원한 쌍도 따로 집계한다.
- **속도**는 동일한 영상 선수 추적에서 발 좌표를 각 엔진으로 미터로 바꾸고 공용 속도 계산기를 적용한다. Alfheim 영상 파일의 시작 시각과 센서의 20Hz m/s 값을 정렬한 뒤 km/h로 변환한다. 0.15초를 넘는 센서 공백은 보간하지 않는다. 선수 위치는 영상의 직접 표식 두 점과 대조한다. 선별한 영상 표식과 센서 위치에 맞춘 시차는 Anzhi +3.6초, Strømsgodset +3.4초다. 이 시차는 엔진 출력이나 속도 오차로 맞추지 않았다. 일치 여부는 별도 프레임 30·60에서 검사한다. **수동 구장 보정 대조군**은 카메라가 고정된 Alfheim의 첫 프레임에 표기한 구장 교차점으로 만든다. 자동 엔진 점수와 분리한다.
- 속도는 MAE, RMSE, 편향, 절대오차 p95, 동일한 프레임 지원 범위에서의 11프레임 지속 최고속도와 발생 시각 오차를 기록한다. 센서가 있는 프레임과 모델이 속도를 내보낸 프레임의 분모를 모두 저장한다. 속도 샘플 간 상관이 있으므로 프레임 수를 독립 경기 수로 해석할 수 없다.
- `report.json`에 커밋, 표식별·프레임쌍별 요약과 게이트를 기록한다. `speed_trace.csv`에는 동일 시각의 센서와 예측을 적고, 자동 보정이 거부하면 예측란을 비워 둔다. 경과시간은 동시 실행의 영향을 받는다.

현재 파일럿의 속도 검증 범위는 Alfheim의 두 경기와 3초씩의 짧은 구간이다. 장시간 추적, 아마추어 풀매치, 공중 볼 속도와 오프사이드 판정 정확도는 이 평가 범위에 포함되지 않는다.
