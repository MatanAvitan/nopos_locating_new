# NeurIPS 2026 Submission 28831: reviews as received

Stored verbatim so later sessions audit claims against the source rather than
against our own paraphrase. Scale is 1-6.

## Area Chair xvH6 (metareview)

a) Summary. This paper investigates how NoPE can recover absolute token
positions. The authors identify a research gap in the literature and propose a
five-step circuit mechanism to explain this phenomenon.

b) Strengths. The paper successfully identifies theoretical gaps in existing
explanations of emergent positionality. The authors move beyond purely
observational probing by proposing a theoretically motivated approach.

c) Weaknesses. The approach is validated only on shallow toy models explicitly
trained to predict position, severely limiting its practical relevance to actual
language models. The submission lacks a necessary limitations section.

d) The review panel leans toward rejection. While the reviewers acknowledge the
research gap this paper attempts to fill, the overall significance of the
contribution is undermined by the limited evidence provided.

## Reviewer sT2w. Rating 5 (Accept), Confidence 5. Quality 4, Clarity 3, Significance 3, Originality 4.

Summary. The Manuscript describe how autoregressive transformers contain
positional information by using 5 step circuit. In the first step the system
relies on beginning of sequence token (BOS) that starts rotating in a circle
after new tokens are added. its magnitude decays in the attention according to a
harmonic series. The full cycle of actions in the circuit are described in the
Appendix A and in Chapter 3. Using two extra layers in the proposed architecture
of the transformer, it becomes obvious that the positional encoding has a clear
intrinsic positional component. In the presented architecture the positional
encoding uses direction of the embedding vector as the major features for token
position. The Manuscript brings a valuable explainability component into the
behavior NoPE transformers.

Questions. I am wondering about the computational load of the 5 step circuit
compared with a rather simple positional encoding. What Is the tangible benefit
for making the change? Would improvement of the quality of infrerence when using
the 5 step circuit bring a strong rationale to use the architecture?

Limitations: Yes.

Formatting. The derivation of the 5-step circuit is separated into the Appendix
A and into the chapter 3. Jumping from one place to the other is a bit
irritating. On the other hands the Figures used are very good.

## Reviewer jKbL. Rating 2 (Reject), Confidence 3. Quality 2, Clarity 3, Significance 2, Originality 2.

Weakness 1. The mechanism is an over-formalization of trivial phenomena. For
example, in Step 2's condition (i) (directional separation) is nearly guaranteed
in high-dimensional space, two different inputs through a linear transformation
will almost surely produce non-parallel outputs. Condition (ii) (directional
coherence) is simply the law of large numbers in high dimensions: weighted
averages of many approximately random vectors concentrate around the mean
direction. And Step 3's observation that BOS attention weight decreases with
position is a basic property of softmax, adding positive terms to the
denominator makes the fraction smaller. None of these require any special
"conditions" to hold. So the novelty is still questionable for me.

Weakness 2. No connection to real NoPE LLMs. The mechanism is only validated on
toy 2-layer models explicitly trained to predict position. Whether actual NoPE
language models (trained on next-token prediction at scale) use the same
mechanism is entirely undiscussed, limiting practical relevance.

Questions. For weakness 1: I hope the authors could clarify which step of the
mechanism is extra novel, i.e., under what conditions it might fail to hold, or
whether alternative position-encoding mechanisms exist that the model did not
converge to. If all steps are inevitable consequences of the architecture, what
is the contribution? For weakness 2: I hope the authors could validate the
mechanism on a NoPE model trained with next-token prediction (rather than a
position-prediction objective), or at least discuss why conclusions from the toy
setting would transfer to practical scenarios.

Limitations: Do not contain a limitation discussion part.

## Reviewer 1JgN. Rating 3 (Borderline reject), Confidence 4. Quality 3, Clarity 3, Significance 2, Originality 2.

Weakness 1. The relationship to alternative accounts of emergent positionality
is underdeveloped. In particular, Zuo et al. propose that causal prefix
aggregation creates an adjacency geometry in which nearby token representations
have greater cosine similarity than distant ones. This explanation is also
directional and survives normalization, making it closely related to the present
account. The paper should clarify whether the BOS-to-nonBOS interpolation
produces the adjacency pattern, whether adjacency is a distinct mechanism, or
whether both are different geometric descriptions of the same prefix-aggregation
process.

Weakness 2. The evidence is limited to shallow models trained explicitly to
predict position. Even the "larger" experiment remains a two-layer Transformer,
rather than a pretrained language model in which positional information emerges
incidentally while optimizing next-token prediction. Consequently, while this
research convincingly demonstrates that the proposed circuit can emerge, it
provides little evidence that it is the mechanism used by practical NoPE
language models.

Questions. Does the proposed two-direction circuit mathematically imply the
adjacency pattern reported by Zuo et al.? How robust is the mechanism when the
model is not trained with direct supervision on absolute position? Does the same
BOS-dependent interpolation arise in NoPE models trained exclusively with a
language-modeling objective, and is it used causally by the model rather than
merely being available to a probe?

Limitations: No, the paper should at least discuss the limitations in terms of
task specificity and architectural scope.
