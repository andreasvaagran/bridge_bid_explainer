#!/usr/bin/env bash
# Trains every model the thesis evaluates (evaluation/common.py reads them from these folders):
#   runs/sl/<mode>                SL concept model per mode (joint, independent, sequential)
#   runs/sl/<mode>_s{1,2}         its two seed replicates
#   runs/sl/plain, plain_s{1,2}   the plain network: the joint architecture with every concept loss off
#   runs/distill/prepared         brl RL+FSP's own games, labelled by the rule engine
#   runs/sl/distilled_<mode>      the mode's SL recipe trained on those games instead of WBridge5's
#   runs/ppo/<run>                PPO per mode, with the KL anchor (eta 0.01) and without it (eta 0), and from the
#                                 plain network without it
# Needs data/wbridge5/prepared (training/prepare_sl_data.py), dds_results/ and brl's networks in bridge_models/.
# Done runs are skipped, so the script can be rerun after a failure. Run from the repository root:
#   bash pipeline/train.sh
# The explainer is trained separately (pipeline/train_explainer.sh).
set -u
cd "$(dirname "$0")/.."
mkdir -p runs
LOG=runs/train.log
source pipeline/lib.sh

J="learning_rate=0.001 hard_coef=10 weight_decay=0.05"                     # joint
I="learning_rate=0.001 reweight_concepts=False weight_decay=0.05"          # independent
S="learning_rate=0.001 weight_decay=0.01"                                  # sequential
P="learning_rate=0.001 weight_decay=0.05 hard_coef=0 concept_only_coef=0"  # plain (joint architecture, no concept loss)
D=runs/sl
FSP=prepared_path=runs/distill/prepared

log "SL"
# shellcheck disable=SC2086
{
  queue sl_run $D/joint joint 480000 $J rng_seed=44
  queue sl_run $D/independent independent 480000 $I rng_seed=44
  queue sl_run $D/sequential sequential 480000 $S rng_seed=45
  for s in 1 2; do
    queue sl_run $D/joint_s$s joint 480000 $J rng_seed=$s
    queue sl_run $D/independent_s$s independent 480000 $I rng_seed=$s
    queue sl_run $D/sequential_s$s sequential 480000 $S rng_seed=$s
  done
  queue sl_run $D/plain joint 480000 $P rng_seed=44
  for s in 1 2; do queue sl_run $D/plain_s$s joint 480000 $P rng_seed=$s; done
}
wait

log "distillation data: brl RL+FSP's games, labelled by the rule engine"
$PY -u scripts/distill_rl_fsp.py --stop_after data >> "$LOG" 2>&1
# shellcheck disable=SC2086
{
  queue sl_run $D/distilled_joint joint 480000 $J rng_seed=44 $FSP
  queue sl_run $D/distilled_independent independent 480000 $I rng_seed=44 $FSP
  queue sl_run $D/distilled_sequential sequential 480000 $S rng_seed=45 $FSP
}
wait

log "PPO"
O=runs/ppo
SLOTS=${PPO_SLOTS:-3}
#     function name              mode        SL checkpoint                  eta  lr   cap   save seed extra
queue ppo_run $O joint_a0.01_lr3e-5 joint       $D/joint/params-best.pkl       0.01 3e-5 30000 250  0 concept_coef=0.5
queue ppo_run $O joint_a0_lr3e-5    joint       $D/joint/params-best.pkl       0    3e-5 30000 250  0 concept_coef=0.5
queue ppo_run $O ind_a0.01_lr1e-4   independent $D/independent/params-best.pkl 0.01 1e-4 60000 1000 0 concept_coef=0
queue ppo_run $O ind_a0_lr1e-4      independent $D/independent/params-best.pkl 0    1e-4 60000 1000 0 concept_coef=0
queue ppo_run $O seq_a0.01_lr1e-4   sequential  $D/sequential/params-best.pkl  0.01 1e-4 60000 1000 0 -
queue ppo_run $O seq_a0_lr1e-4      sequential  $D/sequential/params-best.pkl  0    1e-4 60000 1000 0 -
queue ppo_run $O plain_a0_lr3e-5    joint       $D/plain/params-best.pkl       0    3e-5 30000 250  0 concept_coef=0
wait
log "training finished"
