# 평가 이력: 에이전트가 만든 엔진을 어떻게 검증했나

경기장 인식 엔진 두 개는 서로 다른 LLM 에이전트가 작성했다. PivotAnchor는 Antigravity, TurfLens는 Claude가 맡았다. 한 에이전트가 "개선했다"고 보고하면 다른 쪽이 만든 점검 스크립트로 다시 실행해 확인했다. 이 문서는 그 과정의 숫자와 원본 로그 위치를 정리한다.

합격 기준은 내가 정했고, 기준 완화나 수동 보정값 주입을 금지했다. 실패는 그대로 집계하도록 했으며, 어떤 결과를 받아들일지 판단했다. 점검 스크립트와 엔진 코드는 에이전트가 작성했다.

## 라운드별 결과

| 라운드 | 날짜 | 점검 | 결과 | 원본 로그 |
|---|---|---|---|---|
| 2 | 2026-11-05 | 합격 점검 v1, 19개 | 11개 통과 | 로그 없음. 수정 요청서에 숫자만 남아 있다(아래 참고). |
| 3 | 2026-11-06 | 합격 점검 v1, 17개 | 17개 통과 | [round3-acceptance.log](evaluation-history/logs/round3-acceptance.log) |
| 4 | 2026-11-07 | 합격 점검 v2, 48개 | 41개 통과. 변형 버전은 43개 통과 | [round4-acceptance-v2.log](evaluation-history/logs/round4-acceptance-v2.log), [변형](evaluation-history/logs/round4-acceptance-v2-variant.log), [단위 테스트 46개](evaluation-history/logs/round4-unit.log) |
| 5 | 2026-11-07 | 중계 영상 점검, 29개 | PivotAnchor 17개 통과 | [round5-broadcast-pivotanchor.log](evaluation-history/logs/round5-broadcast-pivotanchor.log) |
| 5 | 2026-11-07 | 같은 점검의 TurfLens 판, 42개 | 33개 통과 | [round5-broadcast-turflens.log](evaluation-history/logs/round5-broadcast-turflens.log) |

라운드마다 점검 스크립트가 달라 점검 개수도 19 → 17 → 48 → 29로 바뀌었다. 라운드별 통과율은 직접 비교할 수 없다.

## 오차 주입 12개

4라운드 점검 v2는 여섯 클립 각각에 "화면에서 12 px" 오차와 "구장에서 2 m" 오차를 일부러 넣고, 엔진이 틀린 좌표를 내보내지 않고 거부하는지 확인한다. 12개 모두 차단했다([로그](evaluation-history/logs/round4-acceptance-v2.log)의 `오차 주입 차단` 줄).

같은 라운드에서 통과하지 못한 7개는 다음과 같다.

- SIFT 2초 오차 p95가 10 px를 넘은 클립 4개
- 13:09 클립의 손 표식 오차
- 틀린 시작 카메라를 거부하지 못한 경우 2개(09:20의 `y3`, 8:56의 `x2`)

주입한 오차는 차단했지만 잘못된 초기 카메라 설정 2개는 통과했다. 이 결과를 반영해 이후 `field_benchmark`에서는 커버리지와 정확도를 함께 채점했다.

## 이 저장소에 없는 것

- 점검 스크립트(`antigravity_acceptance_check.py`, `antigravity_acceptance_v2.py`, `antigravity_broadcast_check.py`)와 에이전트에게 보낸 수정 요청서는 개발 장비의 `outputs/` 폴더에만 있다. 로컬 절대 경로와 비공개 경기 영상에 의존하므로 그대로 넣지 않았다.
- 2라운드의 "19개 중 11개"는 원본 로그가 남아 있지 않다. 같은 날 작성한 수정 요청서 첫 문단의 숫자를 옮겼다.
- 점검에 쓴 아마추어 경기 영상은 저장소에 없다. 위 로그만으로는 결과를 다시 만들 수 없다.

누구나 다시 실행할 수 있는 평가는 공개 영상과 고정 커밋을 쓰는 [experiments/field_benchmark](../experiments/field_benchmark/README.md)다.
