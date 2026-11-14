#!/usr/bin/env bash
# The explainer on held-out decisions of every main concept model (3 SL, 3 anchored and 3 unanchored PPO): WBridge5
# test-split positions for every model, and each PPO model's own self-play on the held-out deals. 10 decisions per
# bidding situation, a greedy first attempt, then up to 3 sampled retries. The explainer sees only the facts of the
# decision, so one adapter serves every model; this measures how the shift in decisions affects it.
# Called by pipeline/evaluate.sh (Q = output folder).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
Q=${Q:-runs/evaluation/qwen/models}
ADAPTER=runs/qwen/adapter-best
mkdir -p $Q
say() { echo "[$(date '+%F %T')] $*" >> $Q/run.log; }

# name <TAB> checkpoint <TAB> is_ppo
mapfile -t ROWS < <(JAX_PLATFORMS=cpu $PY -c "
from evaluation.common import MODELS, CONCEPT_MODELS
for n in CONCEPT_MODELS: print(n, MODELS[n], int(n.startswith('PPO')), sep='\t')" 2>/dev/null)
say "models: ${#ROWS[@]}"

for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r name ckpt ppo <<< "$row"
  tag=${name// /_}
  srcs=(wb5); [ "$ppo" = 1 ] && srcs+=(selfplay)
  for src in "${srcs[@]}"; do
    d=$Q/$tag/$src; mkdir -p $d
    if [ ! -f $d/data/test_frames.jsonl ]; then
      say "build $tag/$src"
      if [ $src = wb5 ]; then
        JAX_PLATFORMS=cpu $PY -m nlg.build_teacher_dataset --model_path "$ckpt" --split test --per_category 30 \
          --val_frac 0 --test_frac 1 --out_dir $d/data > $d.build.log 2>&1 || { say "build FAILED $tag/$src"; continue; }
      else
        JAX_PLATFORMS=cpu $PY -m nlg.build_teacher_dataset --model_path "$ckpt" --selfplay_deals 20000 \
          --random_vulnerability --dds_path dds_results/dds_results_500K.npy --per_category 30 \
          --val_frac 0 --test_frac 1 --out_dir $d/data > $d.build.log 2>&1 || { say "build FAILED $tag/$src"; continue; }
      fi
    fi
    if [ ! -f $d/eval/eval.json ]; then
      say "eval $tag/$src"
      $PY -m nlg.eval_qwen --adapter $ADAPTER --labels $d/data/test_labels.jsonl --frames $d/data/test_frames.jsonl \
        --per_category 10 --retries 3 --out_dir $d/eval > $d.eval.log 2>&1 || say "eval FAILED $tag/$src"
    fi
  done
done

$PY - "$Q" <<'EOF' >> $Q/run.log
import json, sys
from pathlib import Path
q = Path(sys.argv[1]); out = {}
for e in sorted(q.glob("*/*/eval/eval.json")):
    d = json.loads(e.read_text())
    key = f"{e.parts[-4]}/{e.parts[-3]}"
    out[key] = {k: d.get(k) for k in ("n", "pass_rate", "pass_rate_with_retries", "token_f1")}
(q / "summary.json").write_text(json.dumps(out, indent=2))
low = [k for k, v in out.items() if v["pass_rate"] is not None and v["pass_rate"] < 0.98]
for k, v in out.items():
    print(f"  {k:<40} first try {v['pass_rate']}  with retries {v['pass_rate_with_retries']}  F1 {v['token_f1']}")
print("below 98% at the first try: " + ", ".join(low) if low else "at least 98% at the first try everywhere")
EOF
say "finished"
