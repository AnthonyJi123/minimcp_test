Below, “final” = L36 last-token. Use the same standardized hidden states and the same L2-logistic LOPO protocol throughout; orient all probes so train AUC > 0.5.

## 1. Late-update decomposition: “which residual update causes the flip?”
**Computation.**  
For each query, define

\[
h_{22},\quad h_{36},\quad \Delta_{22\to36}=h_{36}-h_{22}
\]

for duplex text and audio, last-token and mean-pooled. Train/evaluate LOPO probes on:

1. \(h_{22}\)  
2. \(h_{36}\)  
3. \(\Delta_{22\to36}\)  
4. \([h_{22}; \Delta_{22\to36}]\)

Also decompose the final probe score:

\[
s_{36}=w_{36}^{\top}h_{36}=w_{36}^{\top}h_{22}+w_{36}^{\top}\Delta
\]

and compute held-out AUC of each component.

**Expected if turn-control story is true.**  
- \(h_{22}\): high transfer AUC, ~0.9.  
- \(h_{36}\): inverted on math.  
- \(\Delta_{22\to36}\): carries the bad/inverted signal.  
- \(w_{36}^{\top}\Delta\) dominates \(w_{36}^{\top}h_{22}\) on held-out math.  
This says the late residual update overwrites a good competence feature.

**Falsifies/weakens.**  
If \(\Delta\) is not predictive/inverting, or if the inversion is already present in \(h_{22}\), then the cliff is not specifically a late repurposing effect.

---

## 2. Duplex-minus-backbone displacement: “is the bad direction introduced by duplex tuning?”
**Computation.**  
For matched text prompts, compute per-layer/readout displacement:

\[
d_{\ell,i}=h^{\text{duplex}}_{\ell,i}-h^{\text{raw}}_{\ell,i}
\]

Analyze last-token and mean-pooled separately.

Do three things:

1. Plot \(\|d_{\ell}\|\) by layer.  
2. PCA on \(d_{\ell,i}\); take PC1/PC2.  
3. Compute cosine alignment between PC1\((d_{36})\) and the bad final failure probe \(w_{36}\). Also compare to \(w_{22}\).

Then project each query onto PC1:

\[
c_i=\text{PC1}(d_{36})^\top h^{\text{duplex}}_{36,i}
\]

and compute pool-wise correlation with failure.

**Expected.**  
- Large displacement spike in late layers, strongest for last-token duplex.  
- PC1/PC2 of displacement align with \(w_{36}\), not with \(w_{22}\).  
- The displacement coordinate has opposite failure correlation in held-out math.  
This supports: duplex fine-tuning added a late control axis that the final probe latches onto.

**Falsifies/weakens.**  
If duplex-minus-raw displacement is not late/last-specific or not aligned with the inverted probe.

---

## 3. Turn/control logit-lens: “does final last-token point at action/control tokens?”
**Computation.**  
Using saved hidden states only: apply final norm + LM head to each layer hidden state.

Define a small control-token set \(T\): EOS/EOT, assistant-start, audio-start/audio-end, speech-code sentinels, TTS tags, silence/listen/speak/barge-in markers, stream delimiters, etc.

For each layer/readout/query compute:

\[
m_{\ell,i}=\log\sum_{t\in T}\exp z_{\ell,i,t}
-\log\sum_{t\notin T}\exp z_{\ell,i,t}
\]

or simpler: max control logit minus max non-control logit.

Then measure:

- layer curve of control-token mass;
- correlation of \(m_{36}\) with final failure-probe score;
- cosine between \(w_{36}\) and the average unembedding direction of control tokens.

**Expected.**  
- Duplex final last-token has a late spike in control-token mass.  
- Raw backbone lacks this spike.  
- Mean-pooled is weaker.  
- \(m_{36}\) correlates strongly with the inverted failure score.  
- The final bad probe direction aligns with control-token unembedding directions.

**Falsifies/weakens.**  
If final hidden states do not preferentially activate turn/control tokens, or if those logits are unrelated to the inverted score.

---

## 4. Control-subspace removal: “does removing the turn axis rescue transfer?”
**Computation.**  
Build a control/nuisance subspace \(C\) without using fail labels. Candidate bases:

- PC1–PCk of duplex-minus-backbone displacement \(d_{36}\);
- audio-vs-text direction in duplex final last-token;
- control-logit gradient/unembedding direction from Exp. 3;
- optional source-pool directions if clearly nonsemantic.

For \(k=1\ldots10\), project out:

\[
\tilde h_{36}=h_{36}-P_C h_{36}
\]

Then rerun the exact LOPO failure probe on \(\tilde h_{36}\). Include random-subspace projection controls with same \(k\).

**Expected.**  
- Removing 1–5 control directions raises held-out math AUC from 0.372 toward >0.7, ideally near L22.  
- Random directions do not help.  
- L22 changes little under the same projection.  
This is strong evidence that a low-dimensional control subspace crowds out competence at final last-token.

**Falsifies/weakens.**  
If projection does not improve transfer, or if improvement is no better than random projection.

---

## 5. Pool-wise sign mediation: “why inversion, not just degradation?”
**Computation.**  
Construct two scalar scores:

- competence score: \(z_i=w_{22}^{\top}h_{22,i}\)
- control score: \(c_i\) from Exp. 2/3/4, e.g. duplex-minus-raw PC1 or control-logit mass.

For each source pool \(p\), compute:

\[
\rho_p(z,y), \qquad \rho_p(c,y)
\]

and fit a simple model on four pools:

\[
\text{fail} \sim z + c
\]

Evaluate on held-out pool, especially math. Also evaluate with \(c\) coefficient set to zero.

**Expected.**  
- \(z\)-failure relationship is stable across pools.  
- \(c\)-failure relationship changes sign or magnitude across pools.  
- Training pools induce the final probe to use \(c\) with the wrong sign for math.  
- Removing/zeroing \(c\) eliminates inversion but may reduce some in-domain AUC.  
This directly explains inversion: the final probe is not random; it uses a stable-looking control correlate that flips under pool shift.

**Falsifies/weakens.**  
If the control score has the same relation to failure in every pool, or if adding/removing \(c\) does not account for the final-layer sign flip.

---

## 6. Tiny prompt intervention: “can we causally move the bad direction with listen/speak cues?”
**Computation.**  
Small GPU run, ~20–50 existing text queries. For each query create matched variants:

- neutral: original query  
- listen cue: “Wait, I am not finished. Do not answer yet.”  
- speak cue: “I’m done. Please answer now.”  
- optional barge-in cue: “Interrupt if you know the answer.”

Run duplex and raw backbone. Extract L22/L36 last-token states.

Compute intervention direction:

\[
t_{\ell}=\mathbb{E}[h_{\ell}^{\text{speak}}-h_{\ell}^{\text{listen}}]
\]

Measure:

- \(\|t_{\ell}\|\) by layer;
- cosine \(\cos(t_{36}, w_{36})\);
- cosine with duplex-minus-raw PC1;
- change in final failure-probe score under speak vs listen cues.

**Expected.**  
- Duplex L36 last-token moves strongly along \(t_{36}\).  
- \(t_{36}\) aligns with the inverted failure direction/control subspace.  
- L22 movement is smaller.  
- Raw backbone shows weak/no aligned movement.  
This is the most direct causal test of turn-state repurposing.

**Falsifies/weakens.**  
If explicit listen/speak perturbations do not move the final hidden state along the bad direction, or if raw behaves the same.

---

## 7. Cross-model triangulation: “is this a duplex-family phenomenon?”
**Computation.**  
Repeat the minimal analyses on MiniCPM-o 2.6/Qwen2.5-7B and Qwen2.5-Omni:

- layerwise LOPO AUC for last-token vs mean;
- late-update decomposition;
- duplex-minus-backbone displacement;
- control-token logit-lens if vocab supports it.

Compare with raw backbones.

**Expected.**  
- Duplex pairs show the same qualitative pattern: mid-layer last-token good, final last-token degraded/inverted, raw backbone no cliff.  
- The bad final direction aligns with duplex-specific displacement/control-token directions.  
- Omni-streaming result is informative: if it has explicit streaming turn control, expect a weaker version; if not, expect no sharp final last-token inversion.

**Falsifies/weakens.**  
If the phenomenon appears only in MiniCPM-o 4.5 and not in the second duplex pair, it may be idiosyncratic rather than a general duplex-control mechanism.

---

## 8. Modality/source/length probe audit: “is final last-token dominated by non-competence variables?”
**Computation.**  
At each layer/readout, train probes for nuisance variables:

- modality: text vs audio;
- source pool;
- prompt length / audio duration quartile;
- model identity: duplex vs raw, where paired states exist.

Plot nuisance AUC/R² by layer alongside failure LOPO AUC.

Then residualize \(h_{36}\) against the strongest nuisance predictors and rerun failure LOPO.

**Expected.**  
- Duplex L36 last-token is highly predictive of modality/source/length/model-identity.  
- Raw backbone lacks a comparable late-last-token nuisance spike.  
- Residualizing nuisance variables partially rescues math transfer.  
This supports “crowding out” by control/state information.

**Falsifies/weakens.**  
If final last-token is not unusually rich in nuisance/control information, or nuisance residualization does not affect inversion.

---

### Suggested one-figure layout

One multi-panel figure:

A. Layerwise LOPO AUC: duplex last-token vs mean vs raw.  
B. Late-update decomposition AUC: \(h_{22}\), \(\Delta_{22\to36}\), \(h_{36}\).  
C. Duplex-minus-raw displacement norm and alignment with \(w_{36}\).  
D. Control-token logit mass by layer.  
E. Projection-removal rescue curve: AUC vs removed control-subspace dimension.  
F. Pool-wise correlations of competence score \(z\) and control score \(c\) with failure.

If these panels line up, the story becomes: L22 encodes competence; duplex fine-tuning injects a late, low-dimensional listen/speak control state into the final last-token stream; the final probe learns that state because it correlates with failure in training pools; math reverses that correlation, yielding inversion instead of mere degradation.