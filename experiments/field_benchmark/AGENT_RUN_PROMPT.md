# Claude·Antigravity 공용 벤치마크 실행 프롬프트

아래 구분선 다음부터 복사해서 두 에이전트 중 어느 쪽에든 붙여넣으세요.

---

Futbol Vision 프로젝트의 **외부 실제 경기 7개 구장·속도 벤치마크**를 직접 실행하고, PivotAnchor와 TurfLens를 비교해 줘. 프로젝트 경로는 이 저장소를 체크아웃한 위치야. 실행이 끝날 때까지 확인하고, 실제 산출물을 근거로 한국어로 보고해 줘.

## 실행 범위

- `experiments/field_benchmark/README.md`와 `fixtures.json`을 먼저 확인해. 기본 실행은 `fixtures.json`에 고정된 PivotAnchor(Antigravity)와 TurfLens(Claude)의 커밋을 사용해. 현재 체크아웃 브랜치와 별개로 `bench.py`가 두 커밋을 격리해서 불러와. Git 객체가 빠져 있으면 `origin`의 `pivotanchor/pitch-model`, `turflens/pitch-model` 브랜치에서 가져온 뒤 manifest의 정확한 SHA가 있는지 확인해.
- `fixtures.json`의 클립·치수·합격 기준, `sources.json`의 원본 체크섬, `reference.json`의 표식·센서 시차를 그대로 사용해. 두 엔진에 동일한 공용 입력 캐시를 쓰고, 별도 구장 기준 사진이나 수동 보정값은 자동 실행에 주입하지 마. 소스 코드나 Git 작업 트리도 수정하지 마.
- 내가 후보 커밋 비교를 명시하면 기본 실행을 먼저 보존하고, `fixtures.json`, `sources.json`, `reference.json`을 별도 디렉터리에 복사해 **후보 엔진 SHA 하나만** 교체해. `--manifest`를 네 단계에 동일하게 전달하고 새 결과 폴더를 사용해. 러너는 커밋만 읽으므로 미커밋 변경은 후보 SHA의 결과로 보고하지 마.

## 실행 명령

Python 3.12 이상과 프로젝트 의존성, 로컬 `work/models/yolo26m.pt`가 필요해. 의존성이 설치된 Python을 `PYTHON_BIN`에 지정해 (없으면 `python3`). 한 셸 세션에서 실행해 줘.

```bash
set -e
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
BENCH="$ROOT/experiments/field_benchmark/bench.py"
MANIFEST="$ROOT/experiments/field_benchmark/fixtures.json"
PY="${PYTHON_BIN:-$(command -v python3)}"
test -x "$PY"
test -f "$ROOT/work/models/yolo26m.pt"
"$PY" -B -c 'import sys, cv2, numpy, scipy, torch, ultralytics, imageio_ffmpeg; assert sys.version_info >= (3, 12)'

mkdir -p "$ROOT/outputs/external_benchmark"
OUT="$(mktemp -d "$ROOT/outputs/external_benchmark/run-XXXXXXXX")"
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

1. **재현 정보** — 엔진별 정확한 커밋 SHA, 처리한 클립 수, 새 결과 폴더의 절대 경로, 단위 테스트 결과.
2. **구장 인식** — 일곱 클립별 지원 프레임/전체 프레임, 수동 구장선 표식 오차 중앙값·p95, 통과 표식 프레임/전체 표식 프레임, 1초 잔디 재투영 오차 p95와 표본 수. 두 엔진이 공통 지원한 잔디 프레임 쌍의 결과도 따로 보여 줘. 구장선 없는 대조군의 오탐, 근거 부족으로 보류한 구간, 실행 오류를 구분해. 관중석 하단 두 클립은 진단 그룹이라 오류 없이 끝났어도 자동 인식 성공으로 세지 마.
3. **선수 속도** — Alfheim ZXY 센서와 대조한 프레임 수/센서 유효 프레임 수, MAE·RMSE·절대오차 p95, 같은 지원 범위의 11프레임 지속 최고속도와 시각 오차. 자동 보정으로 속도를 낸 구간이 없으면 `0/N, 오차 산출 불가`라고 적어. 수동 구장 보정 대조군은 별도 표로 표시하고, 선수 대응 검사에서 빠진 구간과 이유도 적어.
4. **합격 여부** — 실패한 항목을 그대로 적고, 숫자는 이번 실행 결과에서 인용해. 미확인 구장 치수로 평가한 영상과 센서가 한 구장의 짧은 구간에 한정된 점을 명시해. `report.json`, `score.log`, `speed_trace.csv`, `unit.log`의 실제 절대 경로를 첨부해.
