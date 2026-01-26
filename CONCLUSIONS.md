# Conclusions: Position Without Positional Embeddings

## The BOS@80 Experiment
We trained a 2-layer NoPE transformer where the BOS token (constant ID) was placed at position 80 instead of position 0. Position 0 contained variable random tokens.

### Key Findings

1.  **Dual Reference Heads:**
    *   **Head 9 (BOS Head):** Attends to position 80 (the actual BOS token). This confirms the model can find and use the constant BOS token as a reference.
    *   **Head 7 (Position 0 Head):** Attends almost exclusively (>96%) to position 0, *despite the token at position 0 varying across sequences*.

2.  **Mechanism of Position 0 Attention:**
    *   **Why Position 0?** Position 0 is the only position visible to *all* future queries in a causal mask. This universal visibility provides a consistent gradient signal during training, encouraging the model to use it as an anchor.
    *   **How?** The strong attention is driven by a massive dot product ($Q_t \cdot K_0^T$).
    *   **Directional Alignment vs. Norm:** Our analysis shows this is *not* simply due to a large norm of the key vector at position 0. Instead, it is due to **directional alignment**. The learned $W_Q$ and $W_K$ matrices for Head 7 are aligned such that query vectors at positions $t > 80$ (and likely earlier) have a high cosine similarity with the key vector projection of *any* token at position 0 relative to other positions.
    *   **Rank-1 Dominance:** SVD analysis of $W_Q^T W_K$ reveals a dominant top singular value, suggesting the head acts as a rank-1 scorer that projects input tokens onto a specific direction that reliably triggers attention when present at the "universally visible" position 0.

3.  **Implications for "Availability vs. Extraction":**
    *   This confirms that "Availability" of positional information is robust. Even without a constant token at the start, the geometric property of "being first" (and thus universally visible) is extracted by the model.
    *   The "Extraction" mechanism (Circuit A - R0) is robust to token content at the anchor position. It uses the "Position 0" anchor to cancel out sequence-specific offsets, enabling length extrapolation.

### Summary
The model solves the "NoPE" problem by anchoring to position 0 not just because it's usually a BOS token, but because it is the causal "root" of the sequence. When the BOS is moved, the model splits its strategy: it still anchors to the root (Pos 0) *and* finds the BOS (Pos 80), likely combining these signals (or using them in different subspace channels) to recover absolute position.
