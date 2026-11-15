## Summary
- 실험용 카메라 보정 세 가지를 기능 플래그 뒤에 추가했다. 앱의 기본 동작은 바뀌지 않는다.
- 합성 61개 장면과 실제 영상으로 A~E를 비교한 벤치마크와 보고서를 넣었다.
- 팬 중 호모그래피 보간이 상쇄되던 버그를 `pitch.blend`로 고쳤다.
- 선이 잔디 위를 미끄러지던 문제와 먼 터치라인이 트랙 위에 그려지던 문제를 고쳤다.

## 결과
보고서: `experiments/hybrid_pitch/results/report.md`
- 합성: AUTO는 A 대비 위치 오차 중앙값이 오라클 97%, 노이즈 86% 감소
- 실제: 공개 AI 라인 모델 0개 검출. 솔버 S와 S의 라인 이름을 받는 D는 스탠드 클립 3개를 입력 없이 풀고, E는 가장 불안정. 낮은 카메라 7구간은 거부
- 실제 미터·km/h 정확도는 정답 데이터가 없어 미검증

## 권고
운영 기본값은 유지하고 `AUTOHL_CALIBRATOR=camera`는 스탠드 높이 영상의 초안 용도로만 사용한다.

## Test plan
- [x] `python -m unittest discover -s tests` (31개 통과)
- [x] `python -m experiments.hybrid_pitch.benchmark --seed 42` (2,440회)
- [x] `python -m experiments.hybrid_pitch.real_eval`
- [ ] 실제 선수 측정 데이터로 위치·속도 검증
