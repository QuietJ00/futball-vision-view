# 채점 결과 스냅샷

`bench.py score`가 만든 `report.json`과 `score.log`를 실행 당시 manifest 복사본과 함께 보관한다. 영상, 프레임 캐시, 예측 `.npz`는 크기와 데이터 사용 조건 때문에 저장소에 넣지 않는다.

| 폴더 | 엔진 커밋 | 설명 |
|---|---|---|
| `2026-11-10/pinned/` | PivotAnchor `6a8ab06`, TurfLens `efd1387` | 저장소의 `fixtures.json`에 고정된 커밋으로 실행한 기본 결과. manifest가 `fixtures.json`과 바이트 단위로 같다. |
| `2026-11-10/candidates/` | PivotAnchor `5f63329`, TurfLens `bf09bdb` | manifest 복사본에서 엔진 SHA만 바꾼 후보 비교. 각 엔진의 예측을 따로 실행한 뒤 한 번에 다시 채점했다. |

두 결과 모두 2026-11-10에 개발 장비에서 실행했다. 이 스냅샷을 추가할 때 다시 실행하지는 않았다.
