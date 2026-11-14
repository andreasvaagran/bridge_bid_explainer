# bridge_bid_explainer
Explainable bridge bidding with concept bottleneck models.

This repository is the code of the master's thesis *Explainability in Cooperative-Competitive Multi-Agent Systems:
A Case Study in Bridge Bidding with Concept Bottlenecks, Counterfactuals and Natural-Language Explanations*
(Andreas Vagran, University of Oslo, 2026). A bidding network first predicts 146 yes/no concepts written from the
bidding system (such as "shows five or more spades"), then bids from them. Every explanation is built from the
model's own concepts and checked against the hand and the auction before it is shown.

The base bidding agent is [brl](https://github.com/harukaki/brl) (Kita et al., 2024). `sl.py`, `ppo.py`, `eval.py`
and `src/` are brl's code; `src/evaluation.py`, `src/roll_out.py` and `src/utils.py` are adapted to pgx 2.6.

The training settings, the calibration of the rules to WBridge5 and the concept list are in [`docs/`](docs/README.md).

## Installation
Python 3.11. The exact environment of the thesis is `requirements-lock.txt` (jax 0.4.38, pgx 2.6.0); install jax
and jaxlib for your CUDA version. The explainer (Qwen3-8B with QLoRA) also needs `requirements-nlg.txt` and a GPU
with 8 GB.
```bash
pip install -r requirements-lock.txt
pip install -r requirements-nlg.txt   # only for the explainer
```

## Data
- **WBridge5 deals.** Download `train.txt` and `test.txt` of OpenSpiel's WBridge5 dataset (see
  [bridge_supervised_learning.py](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/examples/bridge_supervised_learning.py))
  to `data/wbridge5/official_train.txt` and `data/wbridge5/official_test.txt`, then label them with the rule engine:
  ```bash
  python -m training.prepare_sl_data --train data/wbridge5/official_train.txt \
    --test data/wbridge5/official_test.txt --out data/wbridge5/prepared
  ```
- **Double-dummy results.** `dds_results_10M.npy` (self-play deals) and `dds_results_500K.npy` (held-out deals):
  ```py
  from pgx.bridge_bidding import download_dds_results
  download_dds_results()
  ```
- **brl's networks.** Download the five `.pkl` files of [brl's bridge_models](https://github.com/harukaki/brl/tree/main/bridge_models)
  to `bridge_models/`.

## Models
No trained weights are published; `pipeline/train.sh` trains every model below. The score is IMPs per board against
brl SL in duplicate matches on 10,000 held-out deals (thesis, Table 4.3).

| Model                         | Description                                                        | vs brl SL     |
|-------------------------------|--------------------------------------------------------------------|---------------|
| SL joint                      | Concepts and the hidden layer both feed the bid                    | +0.15 IMPs/b  |
| SL independent                | True bottleneck: the bid is made from the concepts alone           | +0.06 IMPs/b  |
| SL sequential                 | True bottleneck: concepts trained first, then the bid from them    | −0.06 IMPs/b  |
| SL plain                      | The same network without concepts                                  | +0.14 IMPs/b  |
| PPO joint / ind. / seq.       | Self-play PPO with a KL anchor to the SL model                     | +0.56 / +0.30 / +0.19 IMPs/b |
| PPO … unanchored              | Self-play PPO without the anchor                                   | +1.09 / +0.32 / +0.22 IMPs/b |
| distilled joint / ind. / seq. | The SL recipe trained on brl RL+FSP's games instead of WBridge5's  | see thesis, Table 4.10 |

## Concepts and the rule engine
`concepts/` decodes a position into the hand and the auction and defines the 146 concepts (`concepts/rules.py`),
calibrated to WBridge5's bidding in its SAYC setting (`notes/wbridge5_system.md`). All concepts and their meanings
are listed in `notes/concept_catalogue.md`. Check the rules against real deals:
```bash
python scripts/audit_concepts.py --deals 20000 --out audit.md
```

## Supervised learning
One concept model per mode (`joint`, `independent`, `sequential`); the settings of each mode are in
`pipeline/train.sh`.

Example
```bash
python -m training.sl prepared_path=data/wbridge5/prepared save_path=runs/sl/joint bottleneck_mode=joint \
  actor_model_type=DeepMind iterations=480000 train_batch=1024 lr_schedule=cosine warmup_steps=1000 \
  learning_rate=0.001 hard_coef=10 weight_decay=0.05 rng_seed=44
```

## Reinforcement learning
Four-seat self-play PPO from an SL model, with a KL anchor of weight `sl_anchor_coef` to it. `pipeline/converge.py`
stops a run when it no longer gains strength and its KL has levelled off.

Example
```bash
python -m training.ppo game_mode=competitive bottleneck_mode=joint load_initial_model=True \
  initial_model_path=runs/sl/joint/params-best.pkl sl_anchor_path=runs/sl/joint/params-best.pkl \
  sl_anchor_coef=0.01 lr=3e-5 num_envs=1024 num_steps=32 minibatch_size=32768 reward_scale=7600 \
  illegal_action_l2norm_coef=0 concept_coef=0.5 log_path=runs/ppo/joint_a0.01_lr3e-5 exp_name=ppo
```

## Training everything
```bash
bash pipeline/train.sh             # SL, seed replicates, plain network, distillation, PPO
bash pipeline/train_explainer.sh   # teacher explanations, then the Qwen explainer
```

## Explanations
`nlg/teacher.py` writes an explanation from the model's decision, `nlg/consistency_checks.py` checks every claim
against the hand and the auction, and the fine-tuned explainer (`nlg/train_qwen.py`) writes the text that is shown.

Explain every bid of two deals:
```bash
python -m nlg.explain_deals --model_path runs/sl/independent/params-best.pkl \
  --opponents bridge_models/model-sl.pkl --adapter runs/qwen/adapter-best --deals 2
```

## Evaluation
Duplicate match between two models (concept models or brl networks):
```bash
python -m evaluation.duplicate_match --a runs/sl/joint/params-best.pkl --b bridge_models/model-sl.pkl
```
The full evaluation of the thesis, then its tables and figures:
```bash
bash pipeline/evaluate.sh
python -m evaluation.report --eval_dir runs/evaluation --out_dir results
```

## Web interface
Seat any model or yourself at each seat, and read for each bid its explanation, the concepts with their weights,
card counterfactuals and answers to follow-up questions.
```bash
python -m gui.server   # then open http://127.0.0.1:8765
```

## Tests
```bash
JAX_PLATFORMS=cpu pytest tests/
```
Tests that need the data, the DDS results or trained models are skipped without them.

## License

Apache 2.0, as brl.

## Citation

```
@mastersthesis{Vagran2026,
        title        = {{Explainability in Cooperative-Competitive Multi-Agent Systems: A Case Study in Bridge Bidding with Concept Bottlenecks, Counterfactuals and Natural-Language Explanations}},
        author       = {Vagran, Andreas},
        year         = 2026,
        school       = {University of Oslo, Department of Informatics},
}
```
brl:
```
@inproceedings{Kita2024,
        title        = {{A Simple, Solid, and Reproducible Baseline for Bridge Bidding AI}},
        author       = {Kita, Haruka and Koyamada, Sotetsu and Yamaguchi, Yotaro and Ishii, Shin},
        year         = 2024,
        booktitle    = {IEEE Conference on Games},
}
```
