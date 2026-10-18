# Spec: validating the BOS-share position mechanism in two-layer NoPE Transformers

This file records the experiment spec that the `beta_*.py` scripts implement.
Results go to `results/mechanism/<run_id>/beta/` and `results/mechanism/beta_sim/`.

## Models

| key | run_id | checkpoint |
|---|---|---|
| `1h` | `attn2_full_L1024` | ATTN2-FULL, L = 1024 (the paper's 1H model) |
| `1h_8192` | `attn2_full_8192_L8192` | ATTN2-FULL, L = 8192 |
| `1h_seed1..4` | `attn2_full_seed*_L128` | ATTN2-FULL, L = 128, four seeds |
| `12h` | `full12h_L1024` | FULL-12H, L = 1024 |
| `lm` | `lm6_L128` | `out-lm-6layer-fulltrain-ddp`, 6 layers, L = 128 |

"Layer 1" and "Layer 2" are blocks 0 and 1.

## Mechanism under test

1. Layer 1 spreads BOS into every token. The BOS share of key j after Layer 1 is large early and small late.
2. A Layer-2 BOS head scores each key by its share: score ≈ B · share. BOS weight then decays more slowly than 1/i.
3. The Layer-2 output collects BOS by two routes: the BOS key, and the shares carried by other keys.
4. The output is a mix of BOS's value and a mean content value. Its direction encodes position and survives LayerNorm.

## Definitions (Layer 2 unless marked)

- x_j: output of the LayerNorm that feeds the block's attention. x_BOS = x_0 (checked identical across sequences).
- β_j = ⟨x_j, x_BOS⟩ / ‖x_BOS‖². β^c_j: the same after removing the LN bias vector from both.
- ξ_j = x_j − β_j x_BOS.
- B^h_i = q_i^h · W_K^h x_BOS / √d_h. E^h_ij = q_i^h · W_K^h ξ_j / √d_h. b^h_i = q_i^h · b_K^h / √d_h.
- B^h: median over reference queries i ≥ 16 of the per-query OLS slope of s_ij on β_j.
- α̂^h_ij = exp(B^h β_j) / Σ_{k≤i} exp(B^h β_k).
- w^h_BOS = B^h_OV x_BOS. w^h_nonBOS = mean of B^h_OV x_j over sequences and j ≥ 1. Δw^h = w_nonBOS − w_BOS.
- t_i = Σ_j α_ij β_j. BOS-key route α_i0, carried route Σ_{j≥1} α_ij β_j.
- m_j: mean over sequences of B_OV ξ_j. m̄: mean of m_j over j ≥ 1.
- z_i = ⟨Σ_j α_ij B_OV x_j, ŵ_BOS⟩. Aggregate: sum over BOS heads, projected on the sum of their w_BOS.
- r_h = mean BOS weight / mean per-key non-BOS weight (i ≥ 1). BOS heads: r_h > 10.
- γ: OLS slope of log ᾱ_{i,BOS} on log i over a window.

## Protocol

Splits reference / calibration / evaluation = 640 / 320 / 1600 sequences (128 / 64 / 340 at L = 8192), disjoint.
Bins: early 1–15, middle 16–127, late 128 to L−1. relMAE = median |ŷ−i|/i over (sequence, position) pairs in the bin.
Frozen decoder: the trained position head (1H, 12H), or a ridge probe on the clean Layer-2 residual output (LM).
Refit probe: ridge on the same features, refit on intervened calibration activations.
Smoothed derivative: degree-6 polynomial in log i over i ≥ 4.

## Experiments

- A0: per-sequence panel of all steps, plus the fraction of evaluation sequences that pass each step.
- A1: z_i bundles for 40 sequences along ŵ_BOS and Δŵ, Layer 1 and Layer 2, plus 2D trajectories on (ŵ_BOS, m̄⊥).
- A2: isotonic z → i decoder fitted on calibration, evaluated per bin.
- A3: cos(Δw^h, −w^h_BOS).
- A4: sd_seq(z_i) / (i |f′(i)|) for Layer 1 and Layer 2.
- B1: share profiles β, β^c at both layers; power-law fit on j ≥ 16.
- C1: exact score identity; explained variance of scores by β_j B^h_i; CV of B^h_i.
- C2: per-query OLS slopes and R².
- C3: α̂ from B^h with the reference-mean shares and with each sequence's own shares, against measured α.
- C4: γ on windows [16,64], [64,256], [256,1024], [1024,8192].
- C5: B^h against ln r_h.
- C7: −i α_ii against d log ᾱ_BOS / d log i.
- C8: regression of B^h_i on the query's own share.
- D1: routes per sequence. D2: content term along ŵ and Δŵ. D3: split of z into four parts.
- F1 keys, F2 values, F3 queries, F4 all shares removed, F5a permuted shares, F5b random x_BOS (5 draws); BOS-heads-only variants.
- G1: idealized simulation, five settings. G2: B1 and A4 on the untrained model. G3: hand-set Layer 2 in the 1H model.

## Scripts

| script | experiments |
|---|---|
| `beta_extract.py` | reference/calibration/evaluation passes |
| `beta_analyze.py` | A0–A4, B1, C1–C5, C7, C8, D1–D3 and figures |
| `beta_interventions.py` | F1–F5 |
| `beta_sim.py` | G1 |
| `beta_construct.py` | G3 |

## Phase 2: sinks, pull, transition point

Phase 1 found that the trained shares at L >= 1024 are a step, not a smooth decay.
Phase 2 keeps the softmax-over-shares model and changes the claim about the shares.

Definitions (outputs under `results/mechanism/<run_id>/beta_v2/`):

- Centered share (primary): β_j = ⟨x_j − μ, x_BOS − μ⟩ / ‖x_BOS − μ‖², μ = mean non-BOS input of that layer (reference split).
- Sinks: keys j ≥ 1 with centered share > 0.5 in more than 90% of reference sequences. a = 1 + number of sinks.
- β̄: mean centered share over non-sink keys with j ≥ 16.
- B: median per-query OLS slope of the scores on the centered share (queries i ≥ 16).
- Pull of key j: e^{Bβ_j}. Schedule: α_BOS(i) = e^B / (a e^B + n_i e^{Bβ̄}), n_i = number of ordinary keys in 0..i.
- Transition point: i* = a e^{B(1 − β̄)}.

| script | items |
|---|---|
| `beta_v2.py` | N1 (profiles, C1–C3 centered), N2 (schedule fits), N3 (resolution), N4 (cleanliness), N6 (stages, norms, threshold neurons) |
| `beta_v2_causal.py` | N7 (a) pull shift, (b) sink removal, (c) share noise, (d) random-direction controls; Layer-1 threshold-neuron zeroing |
| `beta_extras.py` | sanity check 2, t̂, C2b, C3b, C6, C9, C10, B4, D4–D7 |
| `beta_f6.py` | F6 |
