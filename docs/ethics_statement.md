# Ethics Statement

## Responsible Disclosure and Research Ethics

This research project, *Adversarial Attack Detection in Zero-Trust Networks*, was conducted with strict adherence to ethical research practices:

### Controlled Environment

All adversarial attack techniques were developed and tested **exclusively in a controlled, offline research environment**. No real network infrastructure, production systems, or live user traffic was targeted or compromised at any point during this research.

### Public Datasets

All experiments were conducted using **publicly available datasets**:

- **NSL-KDD**: A widely-used intrusion detection benchmark derived from the KDD Cup 1999 dataset, available from the Canadian Institute for Cybersecurity.
- **CICIDS-2017** (optional validation): Real network captures from a controlled testbed, also from the Canadian Institute for Cybersecurity.

These datasets contain no personally identifiable information and are explicitly released for research purposes.

### Defensive Purpose

The adversarial attack implementations (FGSM, PGD, and black-box transfer attacks) are included **solely to demonstrate and evaluate defensive mechanisms** — specifically, the Zero-Trust contextual policy engine proposed in this work. The attacks serve as evaluation tools for measuring the robustness of the proposed defense, not as offensive capabilities.

### No Novel Attack Techniques

This work does not introduce any novel attack algorithms. All adversarial methods used (FGSM, PGD, HopSkipJump) are well-established in the published literature and implemented via the open-source Adversarial Robustness Toolbox (ART).

### Reproducibility Commitment

Complete code, model checkpoints, and step-by-step reproduction instructions are provided (see `REPRODUCE.md`) to enable independent verification of all experimental claims.

---

*Department of Computer Science and Engineering, SRM Institute of Science and Technology, India*
