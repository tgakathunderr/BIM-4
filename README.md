<br/>
<div align="center">
  <h1 align="center">BIM 4: The Digital Species</h1>

  <p align="center">
    A biological cognitive architecture that learns continuously on a single CPU core.
    <br/>
    <br/>
    <a href="#about-the-project"><strong>Explore the philosophy »</strong></a>
    <br/>
    <br/>
    <a href="#getting-started">Getting Started</a>
    ·
    <a href="#the-biological-architecture">Architecture</a>
    ·
    <a href="#usage">Usage</a>
  </p>
</div>

<!-- Badges -->
<div align="center">
  <img src="https://img.shields.io/badge/Architecture-Biological_Substrate-blue?style=for-the-badge" alt="Architecture" />
  <img src="https://img.shields.io/badge/Hardware-CPU_Native-brightgreen?style=for-the-badge" alt="CPU Native" />
  <img src="https://img.shields.io/badge/License-MIT-orange?style=for-the-badge" alt="License" />
</div>

---

## Table of Contents
- [About The Project](#about-the-project)
- [Built With](#built-with)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [The Biological Architecture](#the-biological-architecture)
- [License](#license)

---

## About The Project

Modern AI is dominated by Large Language Models (LLMs) that passively predict text using massive dense matrices, requiring enormous GPU farms to train via backpropagation. 

**BIM 4 is the exact opposite.** 

It is an active, intrinsically motivated digital organism. It possesses simulated biological organs, learns sequentially from continuous interaction without catastrophic forgetting, and consolidates its memories offline. 

**Why BIM 4 is a Paradigm Shift:**
1. **Zero Backpropagation**: BIM 4 learns sequentially via Numba-compiled Hebbian plasticity in a fraction of a millisecond.
2. **Actor-Critic on SDRs**: Traditional RL requires massive neural networks. BIM 4's Basal Ganglia maps 16,384-dimensional Sparse Distributed Representations (SDRs) directly to expected Dopamine rewards, learning instantly.
3. **No Catastrophic Forgetting**: Because of its 0.39% representation sparsity, learning new associations does not overwrite old ones.
4. **Sleep Consolidation**: A simulated Hippocampus logs episodes online and replays them during `[SLEEP]` offline, permanently locking them into the Cortex.
5. **CPU Native**: Built purely on NumPy and Numba. The entire brain takes < 300MB of RAM.

## Built With

* [![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
* [![NumPy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
* [![Numba](https://img.shields.io/badge/Numba-00A6D6?style=for-the-badge&logo=numba&logoColor=white)](https://numba.pydata.org/)
* [![Rich](https://img.shields.io/badge/Rich-UI-FF69B4?style=for-the-badge)](https://github.com/Textualize/rich)

---

## Getting Started

To get your own digital species up and running locally, follow these simple steps.

### Prerequisites
You need Python 3.8+ installed on your system.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/your-username/bim4.git
   ```
2. Navigate to the directory
   ```sh
   cd bim4
   ```
3. Install the package
   ```sh
   python -m pip install -e .
   ```

---

## Usage 

There are two main ways to interact with BIM 4 out of the box:

### 1. The Visual REPL (Chat Interface)
Raise your own baby! Chat with it in the terminal and watch its brain chemistry (Acetylcholine and Dopamine) spike in real-time as it learns.
```bash
python -m bim4.visual_repl
```
> **Tip:** You can directly trigger the Brainstem by typing `[REWARD]`, `[PAIN]`, or `[SLEEP]` in the chat prompt.

### 2. The Infant Curriculum Test
To see a programmatic demonstration of the biological substrate learning sequentially—from motor babbling, to Pavlovian conditioning, to hippocampal sequence consolidation—run the testing script:
```bash
python -m bim4.test_curriculum
```

---

## The Biological Architecture

BIM 4 replaces the monolithic neural network with 5 distinct, highly optimized biological modules:

* **`thalamus.py`**: The sensory gateway. Uses a deterministic hash to encode raw environmental bytes into incredibly sparse (64 active out of 16,384) distributed representations (SDRs).
* **`cortex.py`**: A blazing-fast, Numba-compiled predictive sequence memory. Uses pure Hebbian plasticity gated strictly by surprise levels to wire sequential associations together.
* **`basal_ganglia.py`**: The action-selection engine. Replaces passive *argmax* predictions. It evaluates the current Cortical state against expected rewards to decide whether to confidently act or randomly "babble".
* **`brainstem.py`**: The chemical bath. Modulates **Acetylcholine (ACh)** based on cortical prediction errors to gate learning rates, and **Dopamine (DA)** based on environmental rewards to reinforce behavior.
* **`hippocampus.py`**: The episodic buffer. Stores recent sequences and replays them during `[SLEEP]` to permanently consolidate memories into the Cortex.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="center">
  <i>Built by UnikAI Lab</i>
</p>
