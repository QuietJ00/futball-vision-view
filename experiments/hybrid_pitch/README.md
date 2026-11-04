# Hybrid pitch calibration experiment

Compares calibration approaches on synthetic clips with exact truth and on the real clips kept locally.
Nothing here changes the app's defaults. Results and the recommendation are in `results/report.md`.

| ID | Method | Code |
|---|---|---|
| A | Clicked landmarks at one frame, then the camera chain (production path for a new venue) | `pitch.from_points` |
| B | Existing shared camera model, started from A | `camfit.fit` |
| C | Free homography per key frame from named lines | `hybrid_calib.fit_homographies` |
| D0 | Shared position and lens from named lines, no camera-chain ties | `hybrid_calib.fit_shared(temporal=False)` |
| D | D0 with camera-chain ties | `hybrid_calib.fit_shared` |
| E | D with tilt and roll also shared: rotation about a vertical axis, optional lens offset | `hybrid_calib.fit_tripod` |
| AUTO | Strictest of D, E, moving camera or C that the clip still fits | `hybrid_calib.calibrate` |
| S (real clips) | Camera search from unnamed paint, people and motion | `field_solver.solve` |

`TRUTH` is the true camera run through the same speed code, so its error is the tracker's share.

## Three kinds of test, reported apart

- **Oracle**: true line names and correspondences. This validates geometry and says nothing about any detector.
- **Noisy semantic**: lines lost, misnamed or displaced, to imitate an imperfect detector.
- **Real model inference**: the released PnLCalib line model on real frames. Only `real_eval.py` does this.

## Running

```bash
python -m unittest discover -s tests
```

```bash
python -m experiments.hybrid_pitch.benchmark --dataset synthetic --models A B C D0 D E AUTO --seed 42
```

```bash
AUTOHL_LINE_WEIGHTS=/path/to/SV_lines python -m experiments.hybrid_pitch.real_eval
```

```bash
python -m experiments.hybrid_pitch.report
```

`real_eval` needs the local job folders under `jobs/`, the clips under `work/`, and optionally the line weights.
Pass clip names to re-run only those; other results in `real.json` are kept.

To try the camera solver in the app, set `AUTOHL_CALIBRATOR=camera`. It is off by default.

## Line model licence

`autohl/semantic_pitch.py` is an independent implementation of HRNetV2-W48, so no GPL code from PnLCalib is included.
The weights `SV_lines` come from [PnLCalib releases](https://github.com/mguti97/PnLCalib/releases) under CC BY 4.0
(Gutiérrez-Pérez and Agudo, "PnLCalib: Sports Field Registration via Points and Lines Optimization") and are not stored
in this repository. They are read with `weights_only=True`.
