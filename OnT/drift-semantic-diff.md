# OnT Candidate: `drift` — EDCM-Scored Semantic Diff

| Field | State |
|---|---|
| Threshold status | Raised; awaiting ratification |
| Proposed destination | `The-Interdependency/skill-lib` |
| Working skill name | `drift` |
| Implementation | Absent — manifest-first; no code before ratification |
| Pipeline specification | Specified, not yet assembled or tested |
| Thresholds | Conjectural |
| Bone-sequence semantics | Defended in the proposal; must bind to exact EDCM canon |
| Calibration specimen | Identified, not yet frozen as a fixture |
| Proven claims | None |
| Provenance | Claude generated. Context, prompt by Erin Patrick Spencer. |
| Captured | 2026-07-26 |

## Why OnT raised

This is beyond a passing idea but has not entered implementation. It has a named skill, bounded purpose, ordered pipeline, output contract, calibration specimen, immediate targets, an upstream defect gate, and explicit claim-status discipline.

Raising OnT preserves the proposal and its unresolved constraints. It does not ratify the name, thresholds, status labels, implementation, or architectural placement.

## Proposal as submitted

Better than receipts — receipts are human labor at ship-time; this is machine labor at every diff. Ratifiable proposal:

**Skill: `drift` (skill-lib, manifest-first — no code until this is ratified)**

**Purpose:** EDCM-scored semantic diff. Git diff weights all lines equally; |∆|a bone-weighted diff distinguishes structural drift from cosmetic churn|∆|. Flesh edits are cheap; bone-sequence edits are load-bearing — |bones are sequence-bearing by type, bag-of-bones forbidden| — so the diff must be edit-distance over the extracted bone *sequence*, not set comparison.

**Pipeline:**
1. smash-join preprocess both versions (protects bone-order from scrambling)
2. extract bone sequence per version (bones_words_v1 + affixes + punct rules)
3. sequence-level diff over bones → structural-drift score
4. marker deltas across the 9 metrics → dissonance-drift vector
5. verdict: `PASS | DRIFT(metric citations) | STRUCTURAL(bone-diff hunks)`, exit nonzero on raise — same CI shape as the ratios runner, `--strict --root .`

**Verdict discipline:** every raise cites the exact bone hunks and marker IDs that fired — |no inherited prose figures| applies to the instrument's own output.

**Calibration — you already own a labeled specimen:** |the false-chain FLOW_DIRECTION docs vs. the corrected tensor-join docs| are a real drift pair with known ground truth. First test fixture. Thresholds start `conjectural`, upgrade to `defended` only after that pair separates cleanly from a cosmetic-edit control pair.

**Immediate targets once green:** |ZFAE living-spec vs. implementation drift|, and |the three-repo FLOW_DIRECTION propagation| — run drift before and after patching to verify the correction actually landed everywhere.

**Gate, stated plainly:** |§|the 15 collision priority violations sit upstream of this|§| — family assignment feeds marker deltas, so drift built on the current markers inherits the defect. Either discharge the 15 first, or ship with a `KNOWN_DEFECT` field in every verdict citing them. Silent inheritance is the one option that's out.

**Status tags at birth:** pipeline `test-backed`, thresholds `conjectural`, bone-diff semantics `defended` (follows from bones-are-vectors), nothing `proven`.

This also closes the recursion you like: EDCM measuring its own canon for drift is the article obeying the article's own law.

hmmm — a ruler that measures its own warping is either the beginning of metrology or the end of it; the difference is whether the calibration pair is outside the ruler. Yours is: the false chain already happened to you, which makes your worst architecture error your best test fixture. Waste nothing, not even the wounds.

## Ratification pins

These are preservation boundaries, not silent rewrites of the proposal.

1. **Name collision.** `drift` already names byte/editorial integrity tooling in skill-lib, including `tools/check_skill_lib_drift.py` and `tools/check_consumer_drift.py`. Ratification must either define the new semantic instrument as an intentional extension of that family or assign a non-colliding canonical identity.

2. **Smash-join scope.** Smash-joining preserves recognized multiword bone units through longest-match-first normalization. The ordered extractor and sequence diff preserve global bone order. The preprocessor alone does not prevent arbitrary sequence scrambling.

3. **Nine-metric evidence is mixed.** Current `markers_v1.json` does not make every metric marker-computable. Some dimensions are partial and require semantic comparison or embeddings; unavailable dimensions must remain typed absence or `NA`, never zero and never silently inferred. A verdict must cite both marker IDs and the evidence mode used for each dimension.

4. **Verdicts may overlap.** Metric drift and structural bone drift can occur in the same diff. Ratification must state whether `DRIFT` and `STRUCTURAL` are exclusive labels or independent evidence fields so one class cannot suppress the other.

5. **Pipeline status.** A specified pipeline cannot itself be `test-backed` before the assembled path has an executable test. Ratification may separately inherit test-backed status for existing components, but the new end-to-end pipeline begins unimplemented.

6. **Upstream defect provenance.** The asserted 15 collision-priority violations remain a hard gate, but the implementation handoff must bind that count to an exact EDCM commit, checker output, or issue. Otherwise the verdict can cite only an unverified inherited number.

7. **Calibration integrity.** Freeze the false-chain version, corrected tensor-join version, and cosmetic control as exact provenance-bearing fixtures. Control size and formatting should be close enough that the first successful separation cannot be explained by trivial edit volume alone.

## Promotion condition

Promote this candidate into skill-lib only after ratification fixes:

- canonical identity and relationship to existing drift checks;
- exact source surfaces and frozen calibration fixtures;
- bone extraction and normalization contract;
- mixed-evidence handling for all nine EDCM dimensions;
- verdict schema, including overlapping evidence and `KNOWN_DEFECT` behavior;
- initial status labels that match evidence actually present at birth.

## hmmm

The strongest part of the proposal is not the score. It is the requirement that every raise expose the exact structural hunk and measurement evidence that caused it. The instrument becomes useful when disagreement with its verdict can be localized, reproduced, and corrected rather than answered by another layer of prose.