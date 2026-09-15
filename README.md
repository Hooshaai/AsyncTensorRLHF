# AsyncTensorRLHF

## High‑Throughput Asynchronous RLHF with Tensor‑Native Rewards

**AsyncTensorRLHF** is a research‑grade framework that implements the architecture described in the *Full English Implementation Guide*. It builds on proven open‑source stacks (vLLM, SGLang, verl, OpenRLHF) and adds two original contributions:

1. **Tensor‑native reward computation** – all reward logic runs on the GPU without any CPU‑side decoding, enabling sub‑millisecond latency for token‑level checks.
2. **Asynchronous rollout with off‑policy correction** – rollout workers generate tokens continuously while the trainer consumes experiences from a lock‑free replay buffer, applying PPO/GRPO loss with optional M2PO staleness handling.

The project is deliberately lightweight: it provides the scaffolding to plug in any LLM backend and any RLHF loss, while the core control‑plane (orchestrator) and data‑flow are fully implemented.

---

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Components](#components)
  - [Rollout Workers](#rollout-workers)
  - [Reward Engine](#reward-engine)
  - [Replay Buffers](#replay-buffers)
  - [Trainer Workers](#trainer-workers)
  - [Orchestrator](#orchestrator)
- [Configuration Files](#configuration-files)
- [Running the System](#running-the-system)
- [Extending / Customising](#extending--customising)
- [Testing & Profiling](#testing--profiling)
- [License & Citation](#license--citation)

---

## Architecture Overview
```
┌─────────────────────────────────────────────────────────────────────┐
│                         Orchestrator (Python)                     │
│  – Prompt queue (asyncio)                                          │
│  – Policy version bookkeeping (VersionManager)                     │
│  – Periodic weight‑sync (LoRA or full NCCL)                        │
└───────────────┬───────────────────────────────┬───────────────────┘
                │                               │
                ▼                               ▼
┌───────────────────────┐        ┌───────────────────────────────────────┐
│   Rollout Cluster      │        │            Trainer Cluster            │
│   (Ray actors)         │        │   (Ray actors)                        │
│   – AsyncEngine        │        │   – TrainerWorker (PPO / GRPO)        │
│   – GPU‑Reward Engine  │        │   – WeightSyncService                 │
│   – BoundedReplayBuffer│        │   – Optimiser, Scheduler              │
└───────────────┬───────┘        └───────────────┬─────────────────────┘
                │                            │
                ▼                            ▼
        Experience Replay Buffer (shared)   │
                │                            │
                └───────────────►────────────┘
```

* **Rollout → Reward → Replay Buffer** – all happen on the GPU and are non‑blocking.
* **Trainer → Weight Sync → Rollout** – trainer pulls batches, updates the model, then pushes the new LoRA adapters (or full parameters) to the rollout workers.
* **Orchestrator** keeps the system alive, monitors throughput, and bumps the policy version.

---

## Quick Start
```bash
# 1️⃣ Clone the repo (already in your workspace at RLFW)
cd /Users/tahamajs/Documents/uni/Research/RLFW

# 2️⃣ Install dependencies (Python 3.10+) – you can also use a conda env
pip install -r requirements.txt   # you may need to create this file later

# 3️⃣ Start Ray (head node) – the launch scripts will auto‑start if needed
ray start --head

# 4️⃣ Launch the three components (adjust GPU counts / model path as needed)
./scripts/launch_rollout.sh 2 /path/to/your/model
./scripts/launch_trainer.sh 2 /path/to/your/model
./scripts/launch_orchestrator.sh /path/to/your/model
```
The system will begin generating prompts, computing rewards on‑GPU, filling the replay buffer, and training the policy.

---

## Installation
Add a `requirements.txt` (optional) with the core libraries:
```text
torch>=2.2.0
ray[default]>=2.9.0
vllm>=0.5.0   # or sglang>=0.5.0 depending on backend
triton>=2.2.0
```
Install with `pip install -r requirements.txt`.

> **Note** – The repository only provides the scaffolding. You must supply a compatible LLM checkpoint (e.g., a HuggingFace model) and optionally a tokenizer.

---

## Components
### Rollout Workers (`src/rollout/`)
* **`AsyncEngine`** – thin wrapper around vLLM/SGLang that yields a `GenerationResult` as soon as EOS is reached.
* **`VersionManager`** – thread‑safe counter used by the orchestrator to tag each experience with the current policy version.
* **`launch_rollout.sh`** – convenience script that creates Ray actors; replace the placeholder with real engine initialisation.

### Reward Engine (`src/reward/`)
* **`tensor_native.py`** – two functions:
  * `gpu_reward_simple` – straightforward subsequence check (used in Phase 1).
  * `tensor_native_reward` – fully‑batched version ready for Triton migration.
* **GPU‑first design** – all tensors stay on the device; only a scalar reward is returned.

### Replay Buffers (`src/buffer/`)
* **`BoundedReplayBuffer`** – lock‑protected queue that drops the oldest entry when full (ideal for PPO).
* **`GroupAwareReplayBuffer`** – stores *G* responses per prompt, computes group‑wise advantages, and exposes completed groups to the trainer (for GRPO).

### Trainer Workers (`src/trainer/`)
* **`TrainerWorker`** – Ray actor that samples from the replay buffer, computes PPO or GRPO loss, back‑propagates and updates the model.
* **`ppo_loss.py`** – standard clipped PPO loss with optional M2PO second‑moment correction.
* **`grpo_loss.py`** – loss that expects group‑wise advantages.
* **`launch_trainer.sh`** – script to spin up one or more trainer actors.

### Orchestrator (`src/orchestrator/`)
* **`scheduler.py`** – maintains an async prompt queue, periodically bumps the policy version and calls a user‑provided `weight_sync_fn`.
* **`launch_orchestrator.sh`** – runs the orchestrator as a background Ray task.

---

## Configuration Files (`configs/`)
* **`phase1_sync.yaml`** – baseline synchronous PPO config (no disaggregation). Adjust `model.path`, `training` hyper‑parameters, rollout backend, etc.
* Extend with `phase2_async.yaml` and `phase3_disagg.yaml` as described in the roadmap.

---

## Running the System
1. **Start Ray** (head). All components connect via `--address=auto`.
2. **Launch Rollout Workers** – each worker creates its own `AsyncEngine`, attaches to a shared `BoundedReplayBuffer` (Ray actor handle).
3. **Launch Trainer Workers** – trainer actors pull batches from the same buffer.
4. **Launch Orchestrator** – feeds dummy prompts (replace with real data source) and triggers weight sync every `sync_interval` steps.
5. **Monitor** – you can query Ray’s dashboard (`http://localhost:8265`) for actor stats, or instrument the code with `torch.profiler`.

---

## Extending / Customising
* **Backends** – swap `engine=None` in `AsyncEngine` for a real vLLM (`VLLMEngine`) or SGLang (`SGLangEngine`).
* **Reward Functions** – replace `gpu_reward_simple` with a Triton kernel (see the roadmap). Add hybrid CPU‑based rewards for code‑execution tasks.
* **Off‑Policy Corrections** – enable M2PO by calling `compute_ppo_loss(..., use_m2po=True)`.
* **Weight Sync** – implement a LoRA‑only NCCL broadcast inside `weight_sync_fn` and pass it to the orchestrator.
* **Scaling** – adjust `NUM_WORKERS` and GPU allocation in the launch scripts; the architecture scales linearly across nodes.

---

## Testing & Profiling
* **Unit tests** – place them under `tests/` and use `pytest`. Example tests:
  * Verify `tensor_native_reward` matches a CPU reference.
  * Ensure `BoundedReplayBuffer` respects max size.
  * Check that `GroupAwareReplayBuffer` produces correctly normalised advantages.
* **Profiling** – insert `torch.profiler` blocks around the reward kernel and trainer step. Export traces to TensorBoard (`tensorboard --logdir=log`).

---

## License & Citation
This scaffold is released under the **Apache 2.0 License**. When you publish results using AsyncTensorRLHF, please cite the original roadmap paper (or your own pre‑print) and acknowledge the underlying projects (vLLM, verl, OpenRLHF).

```bibtex
@article{your2026asynctensorrlhf,
  title={AsyncTensorRLHF: High‑Throughput Asynchronous RLHF with Tensor‑Native Rewards},
  author={Your Name and …},
  journal={arXiv preprint arXiv:XXXX.XXXX},
  year={2026}
}
```

---

### Happy hacking! 🎉
Feel free to reach out if you need concrete implementations for the engine wrappers, weight‑sync logic, or any other piece of the pipeline.
# AsyncTensorRLHF
