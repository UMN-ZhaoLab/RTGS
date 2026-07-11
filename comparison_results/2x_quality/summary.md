# 2× FPS Quality Stack — fr1_desk (100 frames)

Fair comparison: both disable keyframe sleep throttle.

| Method | FPS | Wall (s) | PSNR (dB) | ATE (cm) |
|--------|-----|----------|-----------|----------|
| MonoGS Baseline | 0.898 | 111.4 | 20.04 | 3.17 |
| **RTGS** | **1.798** | **55.6** | **19.96** | 3.41 |

## Delta (RTGS − Baseline)
- FPS: **×2.00** (+100.3%)
- PSNR: **-0.07 dB**
- ATE: +0.24 cm
- Wall: -50.1%

## Recipe (`fr1_desk_2x_target.yaml`)
1. Adaptive tracking downsample (0.60–1.0) + coarse-to-fine refine
2. Tracking early-stop (min 10 iters, patience 2, thr 3e-4) + soft iter cut ÷1.6
3. Mapping ÷2.1 with **protected init BA**; random views = 1
4. Late gradient prune (~14%, start 72%)
5. Skip per-frame full-res finalize (full-res only on keyframe create)
6. idle_mapping every 5 frames; kf_interval 6
7. Disable keyframe 3 FPS sleep throttle

## Sweep history
| Config | FPS | PSNR | ATE | Notes |
|--------|-----|------|-----|-------|
| 2x_quality (map÷1.5) | 1.34 | 20.45 | 3.06 | quality-first |
| 2x_mid (map÷1.6) | 1.43 | 19.91 | 2.70 | solid |
| 2x_v2 (map÷1.8) | 1.54 | 20.30 | 2.62 | good |
| 2x_v3 (map÷2.0) | 1.63 | 20.24 | 2.59 | near 2× |
| 2x_final (+skip fullres) | 1.66 | 20.23 | 2.86 | near 2× |
| **2x_target** | **1.80** | **19.96** | 3.41 | **hit ~2×** |
| 2x_push (too aggressive) | 1.48 | 18.02 | — | PSNR fail |
