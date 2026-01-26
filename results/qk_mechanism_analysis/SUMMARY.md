# Investigation: How Does Position-0 Attention Emerge?

## The Puzzle
In the BOS@80 experiment, Head 7 learns to attend ~100% to position 0, even though the token at position 0 is constantly changing across training examples.

## Hypotheses Tested

### Hypothesis 1: Bias Terms Dominate ❌ REJECTED
**Prediction:** If b_Q · b_K is very large, attention score is high regardless of tokens.

**Findings:**
- Constant term b_Q · b_K = -0.0003 (tiny!)
- Token-dependent term q·k = 0.0009 (mean)
- **Conclusion:** Bias terms do NOT explain position-0 attention

### Hypothesis 2: Low Key Variance ✅ SUPPORTED
**Prediction:** W_K projects all tokens to similar vectors, reducing variance.

**Findings:**
- Head 7 key variance: 0.000393
- Head 9 key variance: 0.000647 (1.65x higher)
- **Conclusion:** Head 7 has LOWER key variance - W_K learned to make all tokens look similar!

### Hypothesis 3: Gradient Visibility ⚠️ PARTIALLY SUPPORTED
**Prediction:** Position 0 gets more gradient signal due to causal visibility (128 vs 48 queries).

**Findings:**
- Block 1: r = -0.08 (NOT significant)
- Block 2: r = 0.18 (weak but significant)
- Actual ratio (pos 0 / pos 80): 1.62x
- Predicted ratio: 2.67x
- **Conclusion:** Gradient visibility plays a role but doesn't fully explain the phenomenon

### Hypothesis 4: Initialization Bias ✅ CONFIRMED (but not the cause)
**Prediction:** Position 0 already gets elevated attention at initialization.

**Findings:**
- At init: attention to pos 0 ≈ 0.044 (vs uniform 0.0078)
- This is due to harmonic sum from causal masking: E[att to pos 0] = H_L / L ≈ 0.04
- After training: Head 7 → 1.000 (100%!)
- **Conclusion:** Small init bias exists, but training amplifies it 25x!

### Hypothesis 5: Training Dynamics ✅ CONFIRMED
**Prediction:** Position-0 attention emerges gradually during training.

**Findings:**
- After 2,000 iterations: Head 7 → 0.025 (no emergence yet)
- After 20,000 iterations: Head 7 → 1.000 (full emergence)
- **Conclusion:** Position-0 attention is LEARNED during training, requires substantial optimization

## Key Mechanism

The model learns to attend to position 0 by:

1. **W_K becomes "position-agnostic":** The key projection W_K learns to project all token embeddings to similar vectors, reducing key variance.

2. **W_Q learns a "position-0 query":** The query projection W_Q learns to produce queries that dot-product highly with the low-variance key space.

3. **Training amplification:** The small initialization bias (position 0 is always visible) provides a seed, which gradient descent amplifies over ~20k iterations.

## Remaining Questions

1. **Why Head 7 specifically?** Why does this head specialize in position-0 attention while others don't?

2. **Optimization dynamics:** What is the loss landscape that drives this specialization?

3. **Functional role:** What does position-0 attention provide to the model's position prediction?

## Files
- `analysis_scripts/analyze_qk_mechanism_detail.py` - Bias and variance analysis
- `analysis_scripts/visualize_attention_gradients.py` - Gradient flow analysis
- `analysis_scripts/train_with_attention_logging.py` - Training dynamics
- `results/qk_mechanism_analysis/` - Output plots
- `results/gradient_analysis/` - Gradient visualizations
- `results/training_dynamics/` - Training curve plots
