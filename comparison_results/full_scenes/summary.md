# RTGS Full Scene Comparison

## Per-Scene Metrics

| Scene | Method | PSNR (dB) | ATE (cm) | FPS | Memory (GB) | Wall Time (s) |
|-------|--------|-----------|----------|-----|-------------|---------------|
| fr1_desk | MonoGS Baseline | 27.249 | 1.608 | 2.431 | 1.721 | 249.4 |
| fr1_desk | RTGS Soft | 27.345 | 1.547 | 7.795 | 1.708 | 81.5 |
| fr2_xyz | MonoGS Baseline | 27.264 | 1.571 | 1.641 | 2.213 | 2078.2 |
| fr2_xyz | RTGS Soft | 27.344 | 1.507 | 3.343 | 2.232 | 1022.7 |
| fr3_office | MonoGS Baseline | 27.260 | 1.575 | 1.517 | 2.020 | 1665.6 |
| fr3_office | RTGS Soft | 27.344 | 1.512 | 3.806 | 2.081 | 666.9 |

## Average Across Scenes

| Metric | MonoGS Baseline | RTGS Soft | Delta |
|--------|-----------------|-----------|-------|
| PSNR (dB) | 27.258 | 27.344 | +0.087 |
| ATE (cm) | 1.585 | 1.522 | -0.063 |
| FPS | 1.863 | 4.981 | +3.118 |
| Memory (GB) | 1.985 | 2.007 | +0.022 |
