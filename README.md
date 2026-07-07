# Neural Differential Cryptanalysis of ASCON: A Controlled Architecture Comparison

Training deep neural networks as differential distinguishers against the round-reduced
ASCON permutation, with two goals existing ASCON work hasn't combined:

1. **A controlled comparison** of multiple architectures (MLP, Gohr-style CNN,
   ResNet-style, dilated-conv) trained on *identical* data, splits, and budget, to
   determine which model family actually detects ASCON's residual structure best —
   rather than reporting one architecture in isolation.
2. **Interpretability tied to ASCON's actual design** — using saliency/attribution
   methods to check whether the trained distinguisher's attention aligns with ASCON's
   known per-row rotation offsets and S-box differential properties, rather than
   treating the model as a black box.

## Status

- [x] ASCON permutation implementation, validated against the official reference
      implementation (`pyascon`) across 5000 random (state, round) trials — see `tests/`
- [ ] Vectorized data generator (numpy-batched permutation)
- [ ] Baseline MLP
- [ ] Gohr-style CNN
- [ ] ResNet-style variant
- [ ] Dilated-conv (DBitNet-style) variant
- [ ] Architecture comparison sweep (rounds 1–8, multiple Δ)
- [ ] Interpretability analysis
- [ ] Paper draft

## Repo structure

```
src/
  ascon_core.py       # validated ASCON permutation (do not modify without re-running tests/)
  data_generator.py   # vectorized real-vs-random pair generation
  models/
    mlp.py
    cnn_gohr.py
    resnet.py
    dbitnet.py
  train.py            # shared training loop, used identically across all architectures
  interpret.py         # saliency/attribution + mapping back to ASCON structure
tests/
  test_ascon_core.py  # regression test against reference implementation (self-contained)
scripts/
  run_sweep.sh         # trains all architectures x all (round, delta) combos
results/                # metrics, checkpoints (gitignored, see below)
docs/
  related_work.md      # how this compares to Gohr 2019, Shen et al. 2024, AutoND/DBitNet
notebooks/
  results_analysis.ipynb
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/   # confirm ASCON implementation is still correct before doing anything else
```

## Method summary

We treat the ASCON permutation as a keyless public transform (standard for this line of
work) and train a binary classifier to distinguish:
- **Real pairs**: ciphertexts of (P, P ⊕ Δ) for a fixed input difference Δ
- **Random pairs**: ciphertexts of two independent random inputs

both passed through the same round-reduced permutation. Accuracy significantly above 50%
at round R indicates the permutation has not fully diffused Δ by round R.

## Related work / how this differs

See `docs/related_work.md`. Briefly: Gohr (CRYPTO 2019) introduced this methodology for
SPECK. Shen et al. (2024) applied it to ASCON using a score-distribution-over-multiple-pairs
approach (69.25% at 4 rounds). AutoND/DBitNet (Bellini et al. 2022) compares Gohr's CNN
against a dilated-conv architecture, but not on ASCON, and without structural interpretability.
This repo runs a matched-budget architecture comparison specifically on ASCON and adds
interpretability grounded in ASCON's rotation constants and S-box.

## Citation

If referencing this work before a formal paper draft exists, cite the repo directly.
Paper draft in progress — see `docs/`.

## License

MIT — see `LICENSE`.
