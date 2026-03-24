# AGENTS.md — Editing Context for `nopos_icml_2026`

## Paper

**Title:** A Directional Mechanism in NoPE Transformers (ICML 2026 submission)

**Topic:** How position encoding works in a 2-layer NoPE (No Positional Embeddings) transformer. Layer 1 creates a shared BOS-direction component via causal prefix averaging, and Layer 2 uses an OV matrix to separate BOS/nonBOS directions and interpolate between them based on position-dependent attention weights, creating a linearly decodable position signal.

**File structure:** The paper is monolithic — the entire paper lives in `main.tex` (~1384 lines). No `\input`/`\include`. Custom macros (`\wBOS`, `\dBOS`, `\oB`, `\bos`, `\nbos`, `\battnB`, `\aBij`, `\modelminimal`, `\modelfull`, `\LN`, `\Attn`, `\MLP`, `\softmax`, `\lidx`, etc.) are defined in `icml2025.sty` (lines 845-897), NOT in `main.tex`. Bibliography is in `custom.bib`.

---

## Goal

1. Replace non-standard terminology ("write map", "gauge", "rotation", "poles") with standard mechanistic interpretability language ("OV matrix", "mechanism", "interpolation", "directions").
2. Fix the introduction of the matrix B — show it emerging from the full attention formula via underbrace, not as a standalone definition.
3. Clarify the 1D vs 2D story (computation lives in a 2D subspace, position signal is a 1D path within it).
4. Address all `\ye{}` reviewer comments from the supervisor (Yanai).
5. Reframe Section 4 as describing an algebraic mechanism that *can* exist (no "learned" language), with Section 5 showing that trained models *actually converge* to it.
6. Sharpen the 1-layer insufficiency argument with two explicit obstacles (noisy SNR + MLP can't invert 1/sqrt(i)).
7. Remove Hebrew text on lines 163-164 (already captured in English on lines 166-168).
8. Ongoing sentence-level polish based on user feedback.
9. Distinguish between "BOS attention weight" (post-softmax, alpha) and "attention score" (pre-softmax, s) — the post-softmax weight is called the "mixing coefficient" when it plays its mechanistic mixing role.
10. All terms must be formally introduced before use — "mixing coefficient" is introduced in Step 3 with `\emph{}`.
11. Section 4 must not discuss empirical results — it describes an algebraic mechanism only.
12. Move mechanism-specific definitions (OV matrix, BOS/nonBOS directions, interventions, readout head) from Section 2 (Notation) into a new subsection "Preliminaries" (4.1) in Section 4, since readers skip notation sections but this content is essential for understanding the mechanism.
13. Describe the OV matrix concisely — it's "the linear map from each input x_j to its contribution to the attention output" — plus its mechanistic role. Don't use interpretive gloss about "reading" and "writing".
14. Formally define "BOS anchor" with `\emph{}` at first mechanistic use in Step 1.

---

## Global Terminology Mapping (ALL APPLIED)

| Old | New |
|-----|-----|
| geometric gauge / gauge | removed entirely; use "mechanism" / "trajectory" / "coordinate" |
| write map | OV matrix |
| write space / write-space | OV subspace |
| write directions / write vectors | OV directions / direction vectors |
| rotation / rotates / rotated (as noun) | interpolation / interpolates / interpolated |
| two-pole / pole | two-direction / direction (or endpoint) |
| BOS-aligned directional anchor / directional anchor | BOS-direction component / component in the BOS direction |
| "learned" (in Section 4 only) | removed or replaced with neutral algebraic framing |
| attention mass (for alpha, post-softmax) | "attention weight" before Step 3 introduction; "mixing coefficient" after |
| mixture coefficient | mixing coefficient (normalized) |

---

## Key Constraints

- **Section 4 must NOT use "learned"** — it describes a mechanism that *can* exist algebraically, not what a model learns. Section 5 is where "learned" belongs.
- **Section 4 must NOT discuss empirical results** — forward pointers to Section 5 should state what the mechanism *requires* and note that Section 5 verifies the condition, not "empirical verification is deferred to..."
- **"Write map" must become "OV matrix"** citing Elhage et al. 2021.
- **B definition must use underbrace** in the full attention formula, not standalone.
- **Terminology discipline:**
  - "attention score" = pre-softmax logit s_{i,0}^{(2)}
  - "attention weight" = post-softmax alpha_{i,j}^{(ell)}
  - "mixing coefficient" = alpha^{(2)}_{i,BOS} in its mechanistic mixing role, formally introduced in Step 3
- **Every term must be formally introduced/defined before first use.**
- The **1-layer argument** has TWO obstacles:
  1. Theoretical: noisy signal (Theta(d/sqrt(i)) + O(sqrt(d)), SNR degrades at later positions)
  2. Practical/architectural: after W1 aligns with BOS direction to extract scalar, only W2 . act(.) remains, which cannot implement s -> 1/s^2
- The 2-layer mechanism sidesteps both: produces approximately affine signal, linear head suffices.
- OV matrix description should be concise and factual: "the linear map from each input x_j to its contribution to the attention output" — no interpretive gloss about "reading" and "writing".
- Don't use "end-to-end" for B = W_O W_V since it's just the composition of two adjacent projections, not a full pipeline.
- "BOS anchor" is formally introduced with `\emph{}` in Step 1.

---

## User's Editing Style

- Prefers concise, direct statements.
- Dislikes redundant parenthetical clarifications of already-defined terms.
- Wants mathematical precision — if Step 1 guarantees a BOS component, say "has" not "may have".
- Prefers explicit mathematical notation (e.g., `$\{Bx_j^{(2)}\}_{j=1}^{i}$`) over vague references ("nonBOS aggregate").
- Prefers index notation ($j > 0$) over macro names (\nbos) when referring to positions in equations.
- Rejects "word washing" — avoid words that imply semantics when discussing mathematical objects (e.g., "content" was rejected because it implies token semantics), avoid hedging ("may") when something is guaranteed by prior steps, avoid redundant clarifications.
- Quality bar: "remember it's a top tier paper."

---

## Current Section Structure

### Section 2: Notation
```
[Architecture paragraph: pre-norm 2-layer transformer]
[Causal self-attention paragraph: QKV, scores, weights, aggregation, output (eq:attn_output)]
[Forward pointer to Section 4.1 for mechanism-specific quantities]
[Conventions paragraph: indices, expectations, BOS/nonBOS, superscript notation]
```

### Section 4: The Mechanism: Position Encoding in a 2-Layer Transformer
```
[Intro paragraphs: algebraic framing, 1-layer obstacles, five-step overview] (~lines 447-484)
4.1 Preliminaries (~lines 486-557)
  - OV matrix and affine form (underbrace equation, B definition, mechanistic role)
  - BOS and nonBOS directions (Layer-1 anchor directions, Layer-2 OV directions)
  - Linear readout head (w_head, b_head)
4.2 The Five-Step Circuit (~lines 558-700)
  - Step 1: Causal prefix averaging writes a BOS anchor into every position
    - Paragraph 1: Near-orthogonality precondition (Xavier init → uniform attention)
    - Paragraph 2: BOS component survives prefix average → BOS anchor (with \emph{})
  - Step 2: The OV matrix separates BOS and nonBOS directions
  - Step 3: The QK circuit creates a position-dependent BOS mixing coefficient
    - Formally introduces "mixing coefficient" with \emph{}
  - Step 4: The Layer-2 attention output interpolates between the two directions
    - "Linear readout" sub-paragraph
  - Step 5: A linear head decodes position from the directional signal
4.3 Position signal survives the LayerNorm (~lines 700-715)
```

---

## All Completed Edits

### Terminology replacements (bulk)
1. **Title**: "A Geometric Gauge in NoPE Transformers" -> "A Directional Mechanism in NoPE Transformers"
2. **Figure 1 caption**: gauge -> directional/mechanism
3. **Introduction** (~8 edits): all gauge/write-map terminology replaced, Hebrew text removed, "geometric-gauge" -> "directional"/"mechanism"
4. **TikZ figure**: step box text updated, Structural/Learned/Emergent labels removed, caption rewritten
5. **Section 2 Notation**: B now introduced via underbrace, "Write map and affine form" -> "OV matrix and affine form", "Layer-2 write-space directions" -> "Layer-2 OV directions", "Write-subspace interventions" -> "OV-subspace interventions"
6. **Section 4 opening paragraph**: Full rewrite with algebraic framing, two 1-layer obstacles, five-step overview
7. **Section 4 Steps 1-5**: All terminology fixed, all \ye{} comments resolved, subsection renamed "The Five-Step Circuit", 1D vs 2D clarification added in Step 4
8. **Section 4.2**: gauge -> mechanism, write-space rotation -> directional interpolation
9. **Section 5 Empirical** (~25 replacements): all gauge/write-map/rotation/pole replaced
10. **Section 6 Validation**: write map -> OV matrix (4 replacements)
11. **Conclusions**: gauge -> mechanism, two-pole write-space -> two-direction OV-subspace
12. **Appendix A**: Full rewrite with two explicit obstacles (noisy SNR + MLP limitation)
13. **Appendix C**: ~12 replacements (gauge -> mechanism, write -> OV, rotation -> interpolation)
14. **Appendix E**: 3 replacements (gauge -> directional trajectory)
15. **Bibliography**: Added Elhage et al. 2021 to custom.bib

### Sentence-level fixes
16. **Final sweep**: grep confirmed zero remaining occurrences of gauge (body text), write map, write space, write direction, two-pole, \ye{}, @@, rotation (as noun)
17. **Step 2 fixes**: "content" removed (implies semantics), index $j$ used instead of \nbos macro, "may" -> "has" (Step 1 guarantees BOS component), redundant "directional purity" clause removed, "nonBOS aggregate" -> explicit notation
18. **Section 4 empirical language removed**: "empirical verification is deferred to" -> "Section 5 verify that trained models satisfy this condition"
19. **Terminology unification (attention mass -> mixing coefficient/attention weight)**: All "attention mass" (6 occurrences) replaced. "mixing coefficient" formally introduced in Step 3 with `\emph{}`. Before Step 3: "attention weight". After Step 3: "mixing coefficient". "mixture coefficient" normalized to "mixing coefficient". Stray "mass" in TikZ box and Section 5 fixed.
20. **OV matrix description rewrite**: Trimmed interpretive gloss ("which directions are read/written back"), replaced with concise factual description ("the linear map from each input to its contribution to the attention output") + mechanistic role sentence pointing to OV-subspace interventions via `\Cref`.
21. **Major restructuring (Section 2 -> Section 4)**: Moved mechanism-specific definitions (OV matrix + affine form, BOS/nonBOS directions, OV-subspace interventions, linear readout head) from Section 2 (Notation) into new `\subsection{Preliminaries}` (4.1) in Section 4. Section 2 now contains only standard transformer equations + conventions + a forward pointer to Section 4.1.
22. **Cleveref fix**: Removed `\label{subsec:bos_nonbos_directions}` (was on an unnumbered `\paragraph`), updated all 4 references to point to `\label{subsec:preliminaries}` (numbered Section 4.1).
23. **Section 4 opening sentence**: "using only the causal mask and direction in the residual stream" -> "using only the causal mask and the direction of the attention output in the residual stream"
24. **Added `\label{eq:attn_output}`** to the $o_i$ equation in Section 2, referenced with `\Cref` in Preliminaries "OV matrix and affine form" paragraph.
25. **OV matrix description**: Separated definition from mechanistic claim. Sentence 1: what B is. Sentence 2: "In our mechanism, two distinct directions in the image of B are necessary for position encoding; OV-subspace interventions provide a causal intervention for this claim."
26. **Formally defined "BOS anchor"** with `\emph{}` in Step 1 body text.
27. **Last "gauge" in body text** fixed: "arrows in the gauge" -> "arrows in the trajectory" (Appendix, line 1280).
28. **1-layer argument rewrite**: Continuous SNR framing instead of binary can/cannot; "half MLP" footnote with scaled-activation explanation; scoped $W_1$ claim to "under our construction".
29. **Five-step overview rewrite**: Plain language, no undefined notation, resolved 10 `\ye{}` comments.
30. **Simplified Preliminaries opening**: "We define below the quantities used by the five-step circuit."
31. **OV matrix paragraph**: Reference $v_j$ definition instead of restating it.
32. **Simplified bias remark**: Dropped alpha-sum justification.
33. **Unified BOS/nonBOS terminology**: `\bos`/`\nbos` macros everywhere, removed dashes, dropped redundant superscript sentence, removed "(Step~1 diagnostics)" from paragraph title.
34. **Clarified $B$ as abbreviation**: "We abbreviate the Layer-2 OV matrix as".
35. **Step 1 expansion**: Added near-orthogonality precondition paragraph (Xavier init → uniform attention); revised BOS anchor derivation (BOS component survives, not dominates); added "BOS anchor" with `\emph{}` to five-step overview.
36. **Removed OV-subspace interventions from Preliminaries**: Duplicated in Section 6; `par:ov_interventions` label had zero references.
37. **Fixed readout head naming**: "The position predictor is" → "We define the \emph{linear readout head} as"; eliminated duplicate term.
38. **All `\ye{}` comments resolved**: grep confirms zero remaining occurrences.
39. **Tightened Steps 2-5** (`cc82aeb`):
    - **Step 2**: Deleted wordy skip-connection sentence. Added two named OV conditions: (i) *directional separation* and (ii) *directional coherence*, with forward pointer to Section 5.
    - **Step 3**: Removed opening recap of Step 2 conditions. Replaced "Suppose" hedging with direct condition statement.
    - **Step 4**: Replaced restated concentration condition with reference to "directional-coherence condition (ii) from Step 2". Added forward pointer for near-antipodal OV directions. Shortened concentration reference in Linear readout.
    - **Step 5**: Deleted redundant parameter reminder sentence. Made antipodal sentence explicit.
    - **Section 4.3**: Cut re-summary to one sentence about encoding position in direction, not scale.
40. **Section 5 terminology alignment** (`94a8823`): Updated paragraph title "Residual cancellation and concentration check" → "Residual cancellation and directional coherence". Changed "concentration assumption used at the beginning of Step~4" → "directional-coherence condition~(ii) from Step~2". Changed "consistent with the Step~4 approximation" → "consistent with the directional-coherence condition."
41. **Fixed `\nbos` macro** in `icml2025.sty` line 887: "non-BOS" → "nonBOS" (no dash) (`49880a3`).

---

## Git Commits

| Hash | Description |
|------|-------------|
| `b6b52d8` | Main bulk edit (194 insertions, 151 deletions) |
| `969e085` | Step 2 sentence clarifications |
| `ed9abd8` | Section 4: replace empirical deferral with forward pointer; explicit nonBOS aggregate notation |
| `e067f75` | Distinguish BOS attention weight (post-softmax) from score (pre-softmax); formally introduce 'mixing coefficient' in Step 3 and unify terminology throughout |
| `0464b51` | Move mechanism-specific definitions from Section 2 to new Section 4.1 'Key quantities'; rewrite OV matrix description; fix cleveref labels |
| `ead2f90` | Rename 'Key quantities' subsection to 'Preliminaries' and update all references |
| `35c5802` | Clarify 'direction in the residual stream' in Section 4 opening |
| `463ecf5` | Rewrite OV matrix description: separate definition from mechanistic claim, add cref to attn output and OV-subspace interventions |
| `a739b68` | Rewrite 1-layer argument: continuous SNR framing, half MLP footnote |
| `f81c4a9` | Rewrite five-step overview in plain language |
| `cc92c46` | Simplify Preliminaries opening |
| `e12dfb9` | OV matrix paragraph: reference v_j definition instead of restating |
| `d5273b6` | Simplify bias remark: drop alpha-sum justification |
| `b7d7c5d` | Unify BOS/nonBOS terminology: macros, remove dashes, drop redundant sentence |
| `a86b37e` | Clarify B as abbreviation for Layer-2 OV matrix |
| `e8a5eee` | Expand Step 1: near-orthogonality precondition, BOS anchor derivation, introduce BOS anchor in overview |
| `b9036b3` | Remove OV-subspace interventions from Preliminaries; fix readout head naming |
| `49880a3` | Fix `\nbos` macro: "non-BOS" → "nonBOS" (no dash) |
| `cc82aeb` | Tighten Steps 2-5: named conditions, remove redundancy, cut Section 4.3 |
| `94a8823` | Align Section 5 terminology with Section 4: concentration/Step 4 → directional-coherence condition (ii) from Step 2 |

---

## Discoveries / Technical Notes

- Custom macros are in `icml2025.sty` (lines 845-897), NOT in main.tex.
- The `\label{eq:def_B_ell}` was kept so existing `\Cref` references still resolve.
- Internal labels like `sec:write_bottleneck`, `fig:write_bottleneck`, `eq:mech_step2_attnwrite` were kept since renaming would break cross-references without user-visible benefit.
- The filename `plots/write_bottleneck_curves_all.png` cannot be changed (it's an image file).
- Remaining "write" occurrences: (a) the English verb "we write" / "writes a BOS anchor", (b) internal labels/filenames, (c) commented-out text.
- Line 184 uses Unicode smart quotes (U+201C / U+201D) — the `edit` tool can't match them directly; use Python `open()` + `replace()` instead.
- `\Cref` and `\cref` resolve by label name, not section number — moving labeled content between sections doesn't break cross-references.
- `\paragraph` headings are unnumbered in ICML style (secnumdepth defaults to 3). Labels on `\paragraph` render poorly with cleveref — point references to the parent `\subsection` instead.
- "OV-subspace" (hyphenated) as compound modifier, "OV subspace" (no hyphen) as standalone noun — both correct.
- `r0_gauge` and `gauge_grid` in label/filenames/appendix names are internal-only and left as-is.
