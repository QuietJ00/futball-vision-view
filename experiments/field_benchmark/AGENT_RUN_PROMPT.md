# Claude·Antigravity 공용 벤치마크 실행 프롬프트

아래 구분선 다음부터 복사해서 두 에이전트 중 어느 쪽에든 붙여넣으세요.

---

Futbol Vision 프로젝트에서 **고정 위치 단일 카메라 벤치마크 `fixed_mount_v2`**를 실행해서 PivotAnchor와 TurfLens를 비교해 줘. 프로젝트 경로는 이 저장소를 체크아웃한 위치야. 실제 경기 일곱 클립을 모두 실행하고 이번 산출물로 한국어 보고서를 작성해 줘.

## 실행 범위

- `experiments/field_benchmark/README.md`와 `fixtures.json`을 먼저 확인해. 기본 실행은 `fixtures.json`에 고정된 PivotAnchor(Antigravity)와 TurfLens(Claude)의 커밋을 사용해. 현재 체크아웃 브랜치와 별개로 `bench.py`가 두 커밋을 격리해서 불러와. Git 객체가 빠져 있으면 `origin`의 `pivotanchor/pitch-model`, `turflens/pitch-model` 브랜치에서 가져온 뒤 manifest의 정확한 SHA가 있는지 확인해.
- `fixtures.json`에서 각 클립의 **구장 이름, 길이·폭(m), 치수 출처, 촬영 근거**를 먼저 표로 정리해. 확인된 `pitch.length`와 `pitch.width`를 영상 분석 전에 두 엔진에 동일하게 제공해. 새 영상 추가 요청을 받으면 구장과 치수 출처부터 확인하고, 세부 라인 치수가 따로 확인되면 해당 구장 템플릿에 명시해. 임의 기본 치수를 넣어 통과시키지 마.
- 평가 대상은 카메라 설치 위치가 유지되는 연속 샷이야. 같은 축의 팬·틸트는 포함하고, 줌이 있는 구간은 고정 화면과 구분해 보고해. 카메라 이동, 컷·리플레이, 스티칭, 구장선 미표시는 본평가에서 제외해. 교체 전 `charrua_pan`, `zenit_pan`, `unmarked_overhead`는 새 결과에 섞지 마. `anzhi_c1`, `stromsgodset_c1`, `dfl_08fd33`처럼 고정 시점 조건을 충족한 클립이 자동 보정에 실패하면 실패로 집계해.
- 원본 체크섬, 구장 치수, 기준점과 센서 시차, 합격 기준을 유지해. 두 엔진에 동일한 공용 입력 캐시를 쓰고, 구장 기준 사진이나 수동 보정값은 자동 실행에 주입하지 마. 소스 코드나 Git 작업 트리도 수정하지 마.
- 내가 후보 커밋 비교를 명시하면 기본 실행을 먼저 보존하고, `fixtures.json`, `sources.json`, `reference.json`을 별도 디렉터리에 복사해 **후보 엔진 SHA 하나만** 교체해. `--manifest`를 네 단계에 동일하게 전달하고 새 결과 폴더를 사용해. 러너는 커밋만 읽으므로 미커밋 변경은 후보 SHA의 결과로 보고하지 마.

## 실행 명령

Python 3.12 이상과 프로젝트 의존성, 로컬 `work/models/yolo26m.pt`가 필요해. 의존성이 설치된 Python을 `PYTHON_BIN`에 지정해 (없으면 `python3`). 한 셸 세션에서 실행해 줘.

```bash
set -e
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
BENCH="$ROOT/experiments/field_benchmark/bench.py"
FIXTURES="$ROOT/experiments/field_benchmark"
PY="${PYTHON_BIN:-$(command -v python3)}"
test -x "$PY"
test -f "$ROOT/work/models/yolo26m.pt"
"$PY" -B -c 'import sys, cv2, numpy, scipy, torch, ultralytics, imageio_ffmpeg; assert sys.version_info >= (3, 12)'
"$PY" -B -c 'import json, sys; assert json.load(open(sys.argv[1]))["suite"] == "fixed_mount_v2"' "$FIXTURES/fixtures.json"

mkdir -p "$ROOT/outputs/external_benchmark"
OUT="$(mktemp -d "$ROOT/outputs/external_benchmark/run-XXXXXXXX")"
mkdir -p "$OUT/manifest"
cp "$FIXTURES/fixtures.json" "$FIXTURES/sources.json" "$FIXTURES/reference.json" "$OUT/manifest/"
MANIFEST="$OUT/manifest/fixtures.json"
echo "Benchmark output: $OUT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

"$PY" -u -B "$BENCH" fetch --manifest "$MANIFEST" > "$OUT/fetch.log" 2>&1
"$PY" -u -B "$BENCH" prepare --manifest "$MANIFEST" > "$OUT/prepare.log" 2>&1
"$PY" -u -B "$BENCH" solve --manifest "$MANIFEST" --engine pivotanchor --output "$OUT" > "$OUT/pivotanchor.log" 2>&1
"$PY" -u -B "$BENCH" solve --manifest "$MANIFEST" --engine turflens --output "$OUT" > "$OUT/turflens.log" 2>&1

SCORE_STATUS=0
"$PY" -u -B "$BENCH" score --manifest "$MANIFEST" --output "$OUT" --strict > "$OUT/score.log" 2>&1 || SCORE_STATUS=$?
TEST_STATUS=0
PYTHONPATH="$ROOT" "$PY" -B -m unittest discover -s "$ROOT/experiments/field_benchmark" > "$OUT/unit.log" 2>&1 || TEST_STATUS=$?
printf 'score exit=%s\nunit exit=%s\noutput=%s\n' "$SCORE_STATUS" "$TEST_STATUS" "$OUT"
```

`fetch`와 `prepare`는 일치하는 원본과 공용 캐시를 재사용해. 새 결과 폴더를 쓰면 두 모델의 예측을 새로 실행하고 이전 리포트도 보존할 수 있어. `score --strict`의 종료 코드 1은 합격 기준 미달에도 발생해. `report.json`과 `score.log`를 열어 모델의 기준 미달과 스크립트 오류를 구분해. 준비 단계가 실패하면 해당 로그에서 원인을 확인하고, 해결 가능한 환경 문제는 복구한 뒤 재실행해. 일곱 클립을 모두 처리했는지, 양쪽 결과의 커밋과 입력 키가 맞는지 확인해.

## 보고 형식

1. **재현 정보** — 엔진별 정확한 커밋 SHA, 구장과 입력 치수·출처, 고정 화면과 팬·줌 클립 수, 새 결과 폴더의 절대 경로, 단위 테스트 결과.
2. **구장 인식** — 일곱 클립별 지원 프레임/전체 프레임, 수동 구장선 표식 오차 중앙값·p95, 통과 표식 프레임/전체 표식 프레임, 1초 잔디 재투영 오차 p95와 표본 수. 두 엔진이 공통 지원한 잔디 프레임 쌍의 결과도 따로 보여 줘. 근거 부족으로 보류한 구간과 실행 오류를 구분해.
3. **선수 속도** — Alfheim ZXY 센서와 대조한 프레임 수/센서 유효 프레임 수, MAE·RMSE·절대오차 p95, 같은 지원 범위의 11프레임 지속 최고속도와 시각 오차. 자동 보정으로 속도를 낸 구간이 없으면 `0/N, 오차 산출 불가`라고 적어. 수동 구장 보정 대조군은 별도 표로 표시하고, 선수 대응 검사에서 빠진 구간과 이유도 적어.
4. **합격 여부** — 실패한 항목을 그대로 적고 숫자는 이번 실행 결과에서 인용해. DFL의 고정 시점 판정은 영상에 근거하고, 센서 속도 정답은 Alfheim 두 경기의 짧은 구간에 한정돼 있음을 명시해. `report.json`, `score.log`, `speed_trace.csv`, `unit.log`의 실제 절대 경로를 첨부해.
