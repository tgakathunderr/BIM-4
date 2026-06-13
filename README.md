# BIM 4: The Digital Species

A fully autonomous, biological cognitive architecture. 

While conventional AI models (like LLMs) passively predict text using dense matrices and backpropagation, **BIM 4** operates as an active digital organism. It possesses simulated biological organs, intrinsically motivated learning, and the ability to consolidate memory offline.

## Why BIM 4 is different
1. **Zero Backpropagation**: BIM 4 learns sequentially via Numba-compiled Hebbian plasticity.
2. **Actor-Critic on SDRs**: Traditional RL requires massive neural networks. BIM 4's Basal Ganglia maps 16,384-dimensional Sparse Distributed Representations (SDRs) directly to expected Dopamine rewards, learning instantly.
3. **No Catastrophic Forgetting**: Because of 0.39% sparsity, learning new associations does not overwrite old ones.
4. **Sleep Consolidation**: A simulated Hippocampus logs episodes online and replays them during `[SLEEP]` offline, locking them into the Cortex with a high Acetylcholine (ACh) surge.
5. **CPU Native**: Built purely on NumPy and Numba. Takes < 300MB of RAM.

## The Organs
- **Thalamus**: Hash-based deterministic sensory gateway encoding raw bytes to SDRs.
- **Cortex**: Fast sparse predictive sequence memory.
- **Basal Ganglia**: Action gating based on Dopamine prediction. 
- **Brainstem**: Modulates Acetylcholine (surprise) and Dopamine (reward).
- **Hippocampus**: Episodic buffer for memory playback.

## Quickstart

### Installation
From inside this directory, install the dependencies using pip.
**Windows / macOS / Linux:**
```bash
python -m pip install -e .
```

### The Infant Curriculum
To see the biological substrate learn sequentially from motor babbling, to Pavlovian conditioning, to hippocampal sequence consolidation, run the test curriculum.
Run this command from the parent directory:
```bash
python -m bim4.test_curriculum
```

### Visual REPL
To raise your own baby, you can chat with it in the terminal and watch its brain chemistry in real time.
Run this command from the parent directory:
```bash
python -m bim4.visual_repl
```
*Note: Type `[REWARD]` or `[PAIN]` to spike dopamine, and `[SLEEP]` to trigger offline memory consolidation.*
