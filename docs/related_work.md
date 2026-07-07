# Related Work

Working notes — this doubles as a draft for the paper's related-work section.

## Foundational

- **Gohr, A. (CRYPTO 2019)** — introduced neural differential distinguishers,
  applied to round-reduced SPECK32/64. Established the real-vs-random pair
  classification task and the ResNet-style architecture this whole line of
  work builds on. https://github.com/agohr/deep_speck

## Directly on ASCON

- **Shen, D., Song, Y., Lu, Y., Long, S., Tian, S. (2024).** *Neural
  differential distinguishers for GIFT-128 and ASCON.* J. Inf. Secur. Appl. 82.
  Improves 4-round ASCON accuracy from 50.69% to 69.25% — but via a
  **score-distribution-over-multiple-ciphertext-pairs** method, not a
  single-pair classifier. Also reports that lower Hamming-weight input
  differences perform better, which informs our Δ selection.
  **Difference from this work:** their reported number is not directly
  comparable to a single-pair distinguisher; we need our own single-pair
  baseline rather than citing 69.25% as the target to beat, unless we
  replicate their multi-pair setup exactly.

- **Yadav, T., Kumar, M. (SPACE 2024/2025).** *ML Based Improved Differential
  Distinguisher with High Accuracy: Application to GIFT-128 and ASCON.*
  Reaches 99.4% on 4-round ASCON via a differential-ML hybrid (combining a
  classical differential trail with an ML distinguisher), at 2^18 data
  complexity. Code: https://github.com/tarunyadav/Improved-Differential-Distinguisher-GIFT128-ASCON
  **Difference from this work:** hybrid classical+ML approach, single
  architecture, no architecture comparison or interpretability component.

- **(2026, SJER)** CNN + Residual Shrinkage Network with multi-scale fusion,
  reaches 53.54% max accuracy on 4-round ASCON using 32-ciphertext-pair
  inputs. Useful as another multi-pair data point but again not directly
  comparable to single-pair setups.

## Architecture comparisons (not on ASCON)

- **Bellini, E., Gerault, D., Hambitzer, A., Rossi, M. (2022).**
  *A Cipher-Agnostic Neural Training Pipeline with Automated Finding of Good
  Input Differences.* Introduces DBitNet (dilated convolutions) and compares
  it against Gohr's ResNet across several ciphers (SPECK, SIMON, XTEA, LEA,
  HIGHT, GIMLI, etc.) — **but not ASCON**, and without tying results back to
  cipher-specific structure via interpretability.
  Code: https://github.com/Crypto-TII/AutoND
  **Difference from this work:** closest precedent for our "controlled
  architecture comparison" framing, but ASCON is absent from their cipher
  set, and they don't do structural interpretability.

## Gap this project fills

No existing ASCON work combines (a) a matched-budget, multi-architecture
comparison and (b) interpretability explicitly grounded in ASCON's per-row
rotation offsets and S-box differential properties. That's the contribution.

## Open questions to resolve before writing the paper

- [ ] Decide single-pair vs. multi-pair input format (affects which prior
      numbers are legitimate baselines)
- [ ] Confirm whether AutoND's DBitNet code can be adapted directly for
      ASCON's 320-bit / 5-word state, or needs rework
- [ ] Track down whether Shen et al. or Yadav & Kumar report a *single-pair*
      accuracy number anywhere in their papers (would make direct comparison
      possible without replicating their full pipeline)
