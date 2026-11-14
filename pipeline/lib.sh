#!/usr/bin/env bash
# Helpers shared by pipeline/train.sh: one SL run, one PPO run with its stopping rule (pipeline/converge.py), and a
# queue that runs at most $SLOTS jobs side by side.
PY=${PY:-python}
SLOTS=${SLOTS:-3}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

sl_run() {  # sl_run <save_path> <mode> <iterations> [key=value ...]
  # The SL recipe: batch 1,024, cosine schedule with 1,000 warm-up steps, validation every 5,000 iterations, no early
  # stop. The sequential mode trains the concepts alone for the first 70% of the iterations.
  local dir=$1 mode=$2 it=$3; shift 3
  if grep -q '"completed": true' "$dir/summary.json" 2>/dev/null; then log "[$dir] done already"; return; fi
  local extra=""; [ "$mode" = sequential ] && extra="concept_phase_iterations=$(( it * 7 / 10 ))"
  mkdir -p "$(dirname "$dir")"
  log "start SL $dir ($mode, $it iterations, $*)"
  local t0; t0=$(date +%s)
  # shellcheck disable=SC2086
  $PY -u -m training.sl prepared_path=data/wbridge5/prepared save_path=$dir bottleneck_mode=$mode \
    actor_model_type=DeepMind iterations=$it train_batch=1024 lr_schedule=cosine warmup_steps=1000 \
    eval_every=5000 early_stop_patience=0 save_every=0 print_every=5000 "$@" $extra > "$dir.log" 2>&1
  local rc=$?
  log "end SL $dir rc=$rc after $(( ($(date +%s) - t0) / 60 )) min"
}

ppo_run() {  # ppo_run <out> <name> <mode> <sl checkpoint> <eta> <lr> <cap> <save every> <seed> <extra|->
  # Four-seat self-play PPO from the SL checkpoint: 1,024 environments x 32 steps, minibatch 32,768, KL anchor of
  # weight eta to the same SL checkpoint. converge.py stops the run when it no longer gains strength and its KL has
  # levelled off, and writes converged.json; <cap> is the update limit.
  local out=$1 name=$2 mode=$3 sl=$4 eta=$5 lr=$6 cap=$7 save=$8 seed=$9 extra=${10}
  [ "$extra" = - ] && extra=""
  if [ -f "$out/$name/done" ]; then log "[$name] done already"; return; fi
  mkdir -p "$out/$name"
  log "start PPO $name ($mode from $sl, eta $eta, lr $lr, seed $seed, cap $cap)"
  # shellcheck disable=SC2086
  $PY -u -m training.ppo game_mode=competitive bottleneck_mode=$mode load_initial_model=True \
    initial_model_path=$sl sl_anchor_coef=$eta sl_anchor_path=$sl lr=$lr seed=$seed \
    log_path=$out/$name exp_name=ppo num_envs=1024 num_steps=32 minibatch_size=32768 \
    save_model_interval=$save reward_scale=7600 illegal_action_l2norm_coef=0 \
    total_timesteps=$(( cap * 32768 )) $extra > "$out/$name.log" 2>&1 &
  local pid=$!
  $PY -u pipeline/converge.py --run_dir "$out/$name" --sl "$sl" --pid $pid --cap "$cap" > "$out/$name.converge.log" 2>&1
  wait $pid
  pkill -TERM -f "log_path=$out/$name exp_name" 2>/dev/null   # forked labelling workers outlive the trainer
  if [ -f "$out/$name/converged.json" ]; then
    log "end PPO $name: $($PY -c "import json; d = json.load(open('$out/$name/converged.json')); print(d['reason'], 'at update', d['update'])")"
    date -Iseconds > "$out/$name/done"
  else
    log "PPO $name FAILED, see $out/$name.log"
  fi
}

queue() {  # queue <function> <args>...: run in the background once fewer than $SLOTS queued jobs are running
  while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do wait -n; done
  "$@" &
  sleep 60
}
