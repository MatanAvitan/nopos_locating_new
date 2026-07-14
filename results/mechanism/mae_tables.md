# MAE / relative-MAE for all experiments

MAE in absolute positions; relMAE = median |ŷ−i|/max(i,1) per bin. Position 0 excluded from overall MAE. Bins: early 1–15, middle 16–127, late 128–L−1.

### attn2_1h_L1024 — probes & decoders (L=1024)

| experiment | MAE | MAE early | MAE middle | MAE late | relMAE early (median) | relMAE middle (median) | relMAE late (median) |
|---|---|---|---|---|---|---|---|
| probe_linear_proxy | 92.3 | 83.4 | 79.4 | 94.1 | 979.5% | 95.3% | 13.9% |
| probe_linear_h1bar | 201.9 | 259.1 | 210.7 | 199.9 | 2762.9% | 286.1% | 30.4% |
| probe_linear_post1 | 258.3 | 418.3 | 453.3 | 231.3 | 4718.0% | 615.4% | 36.0% |
| probe_linear_h1 | 187.6 | 206.6 | 166.6 | 189.9 | 2465.3% | 220.9% | 29.1% |
| probe_linear_o2 | 105.5 | 52.9 | 77.4 | 109.9 | 608.6% | 95.9% | 17.1% |
| probe_linear_post2 | 96.6 | 73.2 | 86.3 | 98.3 | 839.4% | 112.4% | 15.6% |
| probe_linear_ln4 | 34.9 | 19.7 | 22.7 | 36.7 | 227.8% | 29.3% | 5.7% |
| probe_linear_mlp_hidden | 23.5 | 13.7 | 10.5 | 25.3 | 144.7% | 12.8% | 3.8% |
| probe_linear_m2 | 30.5 | 15.8 | 13.9 | 32.8 | 165.1% | 17.4% | 5.0% |
| probe_linear_h2 | 28.6 | 53.7 | 33.8 | 27.6 | 585.1% | 40.0% | 4.1% |
| probe_mlp_h1 | 170.7 | 45.1 | 115.5 | 179.7 | 518.1% | 146.9% | 27.6% |
| probe_mlp_o2 | 50.3 | 7.3 | 14.9 | 55.5 | 81.9% | 18.4% | 8.2% |
| probe_mlp_post2 | 33.5 | 15.9 | 16.4 | 35.9 | 175.4% | 20.6% | 5.5% |
| decode_linear_alpha | 124.9 | 306.2 | 112.2 | 123.5 | 3676.7% | 125.4% | 20.6% |
| decode_analytic_alpha | 89.3 | 17.0 | 20.5 | 99.1 | 226.1% | 23.7% | 14.1% |
| decode_mlp_alpha | 71.6 | 11.2 | 21.5 | 78.9 | 118.5% | 24.5% | 12.0% |
| decode_linear_Y | 123.0 | 325.5 | 111.9 | 121.1 | 3978.5% | 121.1% | 20.8% |
| decode_mlp_Y | 55.1 | 7.1 | 15.7 | 60.8 | 72.9% | 18.4% | 9.2% |
| decode_term_h1 | 80.4 | 51.1 | 35.1 | 86.6 | 642.1% | 51.2% | 14.1% |
| decode_term_o2 | 87.7 | 17.3 | 26.7 | 96.5 | 223.9% | 39.0% | 15.3% |
| decode_terms_joint | 80.5 | 50.3 | 35.3 | 86.6 | 632.2% | 51.8% | 14.1% |
| decode_terms_all | 18.6 | 15.6 | 9.9 | 19.8 | 168.0% | 12.0% | 2.9% |
| model_head | 18.4 | 16.2 | 9.9 | 19.6 | 176.2% | 11.8% | 2.9% |

### attn2_1h_L1024 — causal interventions (L=1024)

| experiment | MAE | MAE early | MAE middle | MAE late | relMAE early (median) | relMAE middle (median) | relMAE late (median) |
|---|---|---|---|---|---|---|---|
| baseline | 18.4 | 16.2 | 9.9 | 19.6 | 176.2% | 11.8% | 2.9% |
| C1_ideal_mixture | 81.9 | 27.2 | 26.6 | 89.7 | 327.1% | 33.7% | 13.3% |
| C5_residual_only | 193.1 | 28.6 | 110.9 | 206.1 | 347.4% | 159.9% | 46.5% |
| C4_bos_to_nonbos | 784.8 | 754.1 | 1049.8 | 752.2 | 9375.7% | 1496.1% | 132.5% |
| C4_nonbos_to_bos | 496.1 | 33.7 | 75.0 | 556.5 | 420.2% | 103.0% | 96.5% |
| C2_retain_mech_span | 52.7 | 30.7 | 29.5 | 56.0 | 383.2% | 39.0% | 8.3% |
| C2_ablate_mech_span | 692.7 | 830.8 | 1047.6 | 646.0 | 10406.2% | 1494.9% | 112.2% |
| C2_retain_top2_svd | 381.9 | 53.9 | 215.1 | 408.2 | 663.7% | 310.3% | 72.4% |
| C2_ablate_top2_svd | 259.6 | 423.4 | 453.6 | 232.6 | 5279.3% | 640.2% | 36.3% |
| C3_ablate_dw_sum | 370.5 | 280.5 | 584.5 | 345.2 | 3490.5% | 840.8% | 57.9% |
| C3_ablate_dw_span | 370.5 | 280.5 | 584.5 | 345.2 | 3490.5% | 840.8% | 57.9% |
| C3_ablate_top1_svd | 255.5 | 449.2 | 466.1 | 226.0 | 5602.6% | 657.2% | 34.6% |
| C6_alpha_refmean | 75.6 | 17.9 | 20.1 | 83.5 | 198.5% | 25.4% | 12.7% |
| C6_alpha_permuted | 311.0 | 173.2 | 299.4 | 314.8 | 2200.0% | 437.6% | 53.0% |
| C6_alpha_clamped | 266.1 | 157.2 | 220.6 | 273.6 | 1975.1% | 317.3% | 46.9% |
| C6_uniform_nonbos_attn | 72.3 | 15.2 | 15.3 | 80.4 | 161.3% | 19.6% | 14.3% |
| S5_remove_mlp2 | 824.2 | 158.5 | 224.0 | 910.3 | 1994.0% | 315.0% | 161.0% |
| S5_linearize_mlp2 | 35.0 | 14.3 | 12.8 | 38.1 | 158.8% | 15.9% | 5.9% |
| S5_mlp2_readdir_only | 29.6 | 16.0 | 9.8 | 32.3 | 173.5% | 11.8% | 4.7% |
| S5_patch_bos_coordinate | 77.2 | 16.3 | 20.4 | 85.3 | 180.5% | 26.6% | 13.8% |
| C7_remove | 794.7 | 109.5 | 905.5 | 792.4 | 1148.5% | 1348.3% | 138.6% |
| C7_fixed_random | 833.9 | 1176.6 | 1255.8 | 775.5 | 15012.9% | 1765.8% | 134.8% |
| C7_seq_varying | 833.2 | 1136.4 | 1255.2 | 775.4 | 14431.6% | 1767.5% | 134.6% |

### full12h_L1024 — probes & decoders (L=1024)

| experiment | MAE | MAE early | MAE middle | MAE late | relMAE early (median) | relMAE middle (median) | relMAE late (median) |
|---|---|---|---|---|---|---|---|
| probe_linear_proxy | 70.1 | 76.4 | 62.7 | 71.0 | 922.0% | 74.1% | 11.0% |
| probe_linear_h1bar | 189.9 | 149.6 | 191.5 | 190.4 | 1679.6% | 253.1% | 29.4% |
| probe_linear_post1 | 189.2 | 139.3 | 190.5 | 189.9 | 1628.5% | 252.8% | 29.3% |
| probe_linear_h1 | 184.4 | 92.9 | 159.1 | 189.1 | 935.1% | 190.0% | 29.1% |
| probe_linear_o2 | 57.6 | 111.2 | 63.9 | 55.9 | 1225.0% | 88.5% | 9.7% |
| probe_linear_post2 | 54.6 | 74.9 | 52.7 | 54.4 | 759.7% | 68.5% | 9.6% |
| probe_linear_ln4 | 16.8 | 22.2 | 13.1 | 17.2 | 225.6% | 15.6% | 2.9% |
| probe_linear_mlp_hidden | 1.8 | 2.2 | 1.2 | 1.9 | 21.5% | 1.4% | 0.2% |
| probe_linear_m2 | 8.1 | 13.3 | 10.8 | 7.7 | 126.8% | 12.2% | 1.0% |
| probe_linear_h2 | 3.4 | 11.8 | 4.5 | 3.1 | 89.2% | 4.9% | 0.4% |
| probe_mlp_h1 | 167.9 | 38.4 | 104.0 | 178.1 | 361.3% | 99.6% | 26.8% |
| probe_mlp_o2 | 11.2 | 8.2 | 5.2 | 12.0 | 77.2% | 5.6% | 1.8% |
| probe_mlp_post2 | 10.3 | 10.7 | 5.7 | 10.9 | 114.1% | 6.4% | 1.7% |
| decode_linear_alpha | 69.4 | 161.9 | 91.2 | 65.1 | 1865.7% | 120.0% | 12.0% |
| decode_analytic_alpha | 13.9 | 20.3 | 15.3 | 13.6 | 206.5% | 21.3% | 2.0% |
| decode_mlp_alpha | 10.6 | 9.1 | 4.6 | 11.4 | 79.5% | 5.5% | 1.8% |
| decode_linear_Y | 66.9 | 122.3 | 81.5 | 64.2 | 1315.5% | 112.5% | 11.3% |
| decode_mlp_Y | 7.2 | 4.6 | 2.4 | 7.9 | 39.2% | 2.9% | 1.2% |
| decode_term_h1 | 32.6 | 21.3 | 8.1 | 35.9 | 96.9% | 10.6% | 5.9% |
| decode_term_o2 | 17.1 | 72.0 | 20.3 | 15.8 | 694.2% | 21.3% | 2.3% |
| decode_terms_joint | 14.4 | 60.1 | 14.8 | 13.6 | 568.1% | 13.9% | 1.9% |
| decode_terms_all | 2.1 | 3.9 | 1.8 | 2.1 | 45.1% | 2.0% | 0.3% |
| model_head | 2.6 | 2.0 | 1.5 | 2.8 | 20.1% | 1.7% | 0.4% |

### full12h_L1024 — causal interventions (L=1024)

| experiment | MAE | MAE early | MAE middle | MAE late | relMAE early (median) | relMAE middle (median) | relMAE late (median) |
|---|---|---|---|---|---|---|---|
| baseline | 2.6 | 2.0 | 1.5 | 2.8 | 20.1% | 1.7% | 0.4% |
| C1_ideal_mixture | 190.5 | 68.9 | 119.4 | 201.4 | 818.1% | 167.5% | 42.3% |
| C5_residual_only | 287.8 | 13.1 | 29.4 | 324.7 | 154.3% | 40.3% | 54.2% |
| C4_bos_to_nonbos | 601.6 | 298.3 | 375.8 | 634.9 | 3850.8% | 516.4% | 129.3% |
| C4_nonbos_to_bos | 531.4 | 25.8 | 91.1 | 594.9 | 322.4% | 127.6% | 103.4% |
| C2_retain_mech_span | 2.3 | 1.9 | 1.5 | 2.4 | 18.8% | 1.5% | 0.3% |
| C2_ablate_mech_span | 303.3 | 653.5 | 658.3 | 253.1 | 8698.8% | 919.1% | 29.6% |
| C2_retain_top2_svd | 32.5 | 18.4 | 17.0 | 34.7 | 180.9% | 24.0% | 5.8% |
| C2_ablate_top2_svd | 322.3 | 708.3 | 695.4 | 269.2 | 9448.0% | 970.7% | 36.0% |
| C3_ablate_dw_sum | 336.8 | 787.3 | 754.4 | 277.1 | 10363.2% | 1054.6% | 35.4% |
| C3_ablate_dw_span | 302.6 | 651.6 | 656.4 | 252.6 | 8694.7% | 916.3% | 29.5% |
| C3_ablate_top1_svd | 110.6 | 135.2 | 141.0 | 106.4 | 1720.6% | 197.0% | 17.0% |
| C6_alpha_refmean | 7.0 | 3.0 | 4.0 | 7.5 | 30.1% | 4.7% | 1.0% |
| C6_alpha_permuted | 187.5 | 32.2 | 85.8 | 202.8 | 412.9% | 133.6% | 33.1% |
| C6_alpha_clamped | 191.2 | 31.9 | 73.6 | 208.5 | 394.4% | 107.6% | 34.0% |
| C6_uniform_nonbos_attn | 182.1 | 59.3 | 126.3 | 191.1 | 724.5% | 177.2% | 39.2% |
| S5_remove_mlp2 | 75.8 | 307.4 | 168.3 | 60.4 | 3134.9% | 234.4% | 11.7% |
| S5_linearize_mlp2 | 9.3 | 7.5 | 5.1 | 9.9 | 83.0% | 6.5% | 1.7% |
| S5_mlp2_readdir_only | 101.2 | 5.8 | 72.8 | 106.4 | 62.3% | 101.7% | 19.2% |
| S5_patch_bos_coordinate | 382.4 | 137.5 | 197.9 | 409.6 | 1622.4% | 280.8% | 87.1% |
| C7_remove | 639.2 | 1001.7 | 990.5 | 589.2 | 12599.3% | 1388.1% | 103.7% |
| C7_fixed_random | 655.3 | 1037.9 | 1021.0 | 603.2 | 13372.1% | 1426.9% | 106.0% |
| C7_seq_varying | 649.4 | 1035.4 | 1006.3 | 598.4 | 12975.0% | 1410.5% | 105.1% |

### lm6_L128 — NoPE LM block-wise position probes (L=128)

| experiment | MAE | MAE early | MAE middle | MAE late | relMAE early (median) | relMAE middle (median) | relMAE late (median) |
|---|---|---|---|---|---|---|---|
| linear probe @ L0 residual | 18.7 | 17.0 | 18.9 | 39.0 | 146.1% | 24.7% | 30.4% |
| linear probe @ L1 residual | 4.3 | 2.8 | 4.5 | 7.3 | 29.8% | 5.4% | 4.9% |
| linear probe @ L2 residual | 4.4 | 2.7 | 4.6 | 7.6 | 29.8% | 5.8% | 5.0% |
| linear probe @ L3 residual | 4.9 | 3.2 | 5.2 | 8.1 | 35.8% | 6.5% | 5.3% |
| linear probe @ L4 residual | 5.3 | 3.7 | 5.5 | 8.5 | 41.3% | 6.9% | 5.4% |
| linear probe @ L5 residual | 20.8 | 17.9 | 21.2 | 40.4 | 207.8% | 26.5% | 31.5% |

### lm6_L128_nobos — NoPE LM block-wise position probes (L=128)

| experiment | MAE | MAE early | MAE middle | MAE late | relMAE early (median) | relMAE middle (median) | relMAE late (median) |
|---|---|---|---|---|---|---|---|
| linear probe @ L0 residual | 20.1 | 17.2 | 20.5 | 42.2 | 179.7% | 26.6% | 32.5% |
| linear probe @ L1 residual | 4.4 | 3.1 | 4.5 | 7.4 | 32.4% | 5.5% | 4.8% |
| linear probe @ L2 residual | 4.5 | 3.0 | 4.7 | 7.6 | 31.4% | 5.8% | 5.0% |
| linear probe @ L3 residual | 5.0 | 3.5 | 5.2 | 8.4 | 38.0% | 6.5% | 5.5% |
| linear probe @ L4 residual | 5.4 | 4.1 | 5.5 | 8.6 | 43.9% | 7.0% | 5.8% |
| linear probe @ L5 residual | 22.3 | 21.3 | 22.4 | 42.7 | 248.1% | 28.0% | 33.4% |
