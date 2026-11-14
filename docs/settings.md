# Training settings

The settings of the supervised training and of the PPO runs. `pipeline/train.sh` runs every model with them.

## Supervised learning (`training/sl.py`)

One run takes about 82 minutes on a single 8 GB GPU, with three runs sharing it.

| The same for every mode | |
|---|---|
| Optimiser | AdamW (Adam with decoupled weight decay) |
| Learning rate | 10⁻³, 1,000 warm-up steps, then cosine decay |
| Batch | 1,024 decisions |
| Length | 480,000 iterations (about 40 epochs) |

| Different per mode | Joint | Independent | Sequential |
|---|---|---|---|
| Weight decay | 0.05 | 0.05 | 0.01 |
| Concept weight α (`hard_coef`) | 10 | 4.2 | 4.2 |
| Concept reweighting (`reweight_concepts`) | yes | no | yes |
| Concepts-only policy layer β (`concept_only_coef`) | 0.1 | – | – |
| Concepts trained first (`concept_phase_iterations`) | – | – | 336,000 iterations |

## Reinforcement learning (`training/ppo.py`)

Next to brl's settings for fine-tuning its supervised network, which are from Kita et al. (2024) and, for value
and gradient clipping, from the defaults of brl's `ppo.py`.

| Where this differs from brl | This repository | brl |
|---|---|---|
| Starting model | the supervised model, plus a new layer for the value estimate | brl's supervised model |
| Partner and opponents | copies of the model being trained (self-play in all four seats) | self-play and fictitious self-play |
| KL anchor weight η (`sl_anchor_coef`) | 0.01 (anchored) or 0 (unanchored) | none |
| Concept step (`concept_coef`) | joint mode only: one extra step per update that refits the concepts to the rule engine's labels (weight 0.5) | none |
| Concept-step loss | concept BCE + 0.1 × concepts-only policy layer | – |
| Games in parallel | 1,024 | 8,192 |
| Minibatch | the whole batch | 1,024 |
| Learning rate | joint: 3 × 10⁻⁵; independent, sequential: 10⁻⁴ | 10⁻⁶ |
| When a run stops | by the convergence rule (`pipeline/converge.py`) | after 10,000 updates |
| Longest run | 30,000 updates (joint), 60,000 (independent, sequential) | – |
| Checkpoints | every 250 updates (joint), 1,000 (independent, sequential) | – |

| The same in both | |
|---|---|
| Reward | double-dummy score / 7,600 |
| Steps per game | 32 |
| Passes over the data | 10 epochs |
| Optimiser | Adam, ε = 10⁻⁵; gradient norm at most 0.5 |
| Policy clipping ε | 0.2 |
| Value loss | clipped, weight 0.5 |
| Entropy bonus | 0.001 |
| Discount γ | 1 (no discount) |
| GAE λ | 0.95 |
