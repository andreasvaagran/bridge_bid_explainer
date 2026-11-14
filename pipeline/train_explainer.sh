#!/usr/bin/env bash
# Trains the explainer: Qwen3-8B with QLoRA adapters (rank 16) on the teacher's explanations, learning rate 1e-4,
# cosine schedule, 8 examples per step. Needs the SL and PPO models of pipeline/train.sh, requirements-nlg.txt and a
# GPU with 8 GB. The adapter goes to runs/qwen/adapter-best, where the interface and the evaluation read it.
#   1. teacher datasets: explanations of WBridge5's positions and of the PPO model's own self-play
#   2. stage 1: the explanations
#   3. stage 2: the follow-up questions, continuing from stage 1 and replaying explanations so both tasks are kept
# The thesis's adapter was trained in more stages, each continuing from the one before with revised teacher texts;
# this script runs the two main stages on the current teacher.
#   bash pipeline/train_explainer.sh
set -eu
cd "$(dirname "$0")/.."
PY=${PY:-python}
SL=runs/sl/independent/params-best.pkl
PPO=$($PY -c "from evaluation.common import MODELS; print(MODELS['PPO independent'])")
N=runs/explainer_data
Q=runs/qwen

$PY -u -m nlg.build_teacher_dataset --model_path $SL --out_dir $N/teacher
$PY -u -m nlg.build_teacher_dataset --model_path "$PPO" --selfplay_deals 200000 --per_category 400 \
  --random_vulnerability --dds_path dds_results/dds_results_10M.npy --out_dir $N/teacher_selfplay
$PY -u -m nlg.followup_questions --build --model_path $SL --teacher_dir $N/teacher --out_dir $N/followup

COMMON=(--learning_rate 1e-4 --lr_schedule cosine --warmup_steps 20 --batch_size 1 --grad_accum_steps 8
        --eval_every 50 --eval_rows 150)
$PY -u -m nlg.train_qwen "${COMMON[@]}" --save_dir $Q/stage1 --max_steps 2000 \
  --labels $N/teacher/train_labels.jsonl $N/teacher_selfplay/train_labels.jsonl \
  --explanations $N/teacher/train_frames.jsonl $N/teacher_selfplay/train_frames.jsonl \
  --val_labels $N/teacher/val_labels.jsonl $N/teacher_selfplay/val_labels.jsonl \
  --val_explanations $N/teacher/val_frames.jsonl $N/teacher_selfplay/val_frames.jsonl
$PY -u -m nlg.train_qwen "${COMMON[@]}" --save_dir $Q --max_steps 850 --init_adapter $Q/stage1/adapter-best \
  --labels $N/teacher/train_labels.jsonl $N/teacher_selfplay/train_labels.jsonl \
  --explanations $N/teacher/train_frames.jsonl $N/teacher_selfplay/train_frames.jsonl \
  --val_labels $N/teacher/val_labels.jsonl $N/teacher_selfplay/val_labels.jsonl \
  --val_explanations $N/teacher/val_frames.jsonl $N/teacher_selfplay/val_frames.jsonl \
  --explain_rows 12000 --sft $N/followup/sft_train.jsonl --sft_val $N/followup/sft_val.jsonl
