#!/usr/bin/env bash
# The thesis's evaluation of every model of pipeline/train.sh and brl's five networks, into runs/evaluation.
# Each step runs once (a marker in done/, its output in logs/<step>.log); a failed step does not stop the others, and
# a rerun retries what failed. The tables and figures are then made with
#   python -m evaluation.report --eval_dir runs/evaluation --out_dir <thesis>/results
# Run from the repository root, with nothing else on the GPU (the explainer fills most of an 8 GB GPU):
#   bash pipeline/evaluate.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
EV=${EV:-runs/evaluation}
L=$EV/logs
LOG=$EV/run.log
mkdir -p $EV/done $L
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
CPU="env JAX_PLATFORMS=cpu $PY -u"
GPU="$PY -u"
SLD=runs/sl
DIST=runs/distill

MAIN=("SL joint" "SL independent" "SL sequential" "PPO joint" "PPO independent" "PPO sequential"
      "PPO joint unanchored" "PPO independent unanchored" "PPO sequential unanchored")
SEEDS=("SL joint s1" "SL joint s2" "SL independent s1" "SL independent s2" "SL sequential s1" "SL sequential s2")
PLAIN=("SL plain" "SL plain s1" "SL plain s2")
BRL=("brl SL" "brl RL" "brl RL+FSP" "brl RL+PFSP" "brl RL scratch")
CONCEPT_ALL=("${MAIN[@]}" "${SEEDS[@]}")

step() {  # step <id> <command...>
  local id=$1; shift
  [ -f "$EV/done/$id" ] && return 0
  log "start: $id"
  if "$@" > "$L/$id.log" 2>&1; then touch "$EV/done/$id"; log "done:  $id"
  else log "FAILED: $id, see $L/$id.log"; fi
}
ckpt() { $CPU -c "from evaluation.common import MODELS; print(MODELS['$1'])" 2>/dev/null; }
t() { echo "${1// /_}"; }

missing=$($CPU -c "from evaluation.common import MISSING_CURRENT; print(', '.join(MISSING_CURRENT))" 2>&1 | tail -1)
if [ -n "$missing" ]; then log "models missing: $missing (run pipeline/train.sh first)"; exit 1; fi

log "evaluation start"
step tests env JAX_PLATFORMS=cpu $PY -m pytest -q tests/

# strength: duplicate matches between every pair of models, the anchor ablation, the seed spread
mkdir -p $EV/strength $EV/ablation $EV/seeds $EV/plain $EV/plain_ppo
step strength $GPU -m evaluation.strength_matches --all_pairs --out $EV/strength/strength.json
step ablation $GPU -m evaluation.anchor_ablation --runs runs/ppo --points 6 --out $EV/ablation/ablation.json
step seeds $GPU -m evaluation.seed_variance --seeds_dir $SLD --out $EV/seeds/seeds.json
# the plain network against the concept models of the same seed (44 is the main models' seed)
for s in 44 1 2; do
  suf=$([ $s = 44 ] && echo "" || echo "_s$s")
  joint=$([ $s = 44 ] && echo $SLD/joint || echo $SLD/joint_s$s)/params-best.pkl
  ind=$([ $s = 44 ] && echo $SLD/independent || echo $SLD/independent_s$s)/params-best.pkl
  plain=$([ $s = 44 ] && echo $SLD/plain || echo $SLD/plain_s$s)/params-best.pkl
  step plain_vs_joint$suf $GPU -m evaluation.duplicate_match --a $plain --b $joint --vulnerability none \
    --out $EV/plain/plain_vs_joint$suf.json
  step independent_vs_plain$suf $GPU -m evaluation.duplicate_match --a $ind --b $plain --vulnerability none \
    --out $EV/plain/independent_vs_plain$suf.json
  step plain_vs_brlsl$suf $GPU -m evaluation.duplicate_match --a $plain --b bridge_models/model-sl.pkl \
    --vulnerability none --out $EV/plain/plain_vs_brlsl$suf.json
done
step scratch_vs_brlsl $GPU -m evaluation.duplicate_match --a bridge_models/model-from-scratch-rl.pkl \
  --b bridge_models/model-sl.pkl --vulnerability random --out $EV/plain/scratch_vs_brlsl.json
pp=$(ckpt "PPO plain unanchored")
step ppoplain_vs_start $GPU -m evaluation.duplicate_match --a "$pp" --b runs/sl/plain/params-best.pkl \
  --vulnerability random --out $EV/plain_ppo/vs_start.json
step ppoplain_vs_brlsl $GPU -m evaluation.duplicate_match --a "$pp" --b bridge_models/model-sl.pkl \
  --vulnerability random --out $EV/plain_ppo/vs_brlsl.json
# the distilled models: agreement on RL+FSP's and WBridge5's test games, then their matches
step distill_eval $GPU scripts/distill_rl_fsp.py --out $DIST --eval_dir $EV --stop_after eval
step strength_distill $GPU -m evaluation.strength_matches --distill --out $EV/strength/strength_distill.json

# agreement, concept quality, influence, counterfactuals
mkdir -p $EV/test_split $EV/influence $EV/concepts $EV/counterfactuals
for m in "${CONCEPT_ALL[@]}" "${PLAIN[@]}" "PPO plain unanchored"; do
  step "test_$(t "$m")" $CPU -m evaluation.split_accuracy --model_path "$(ckpt "$m")" \
    --split data/wbridge5/prepared/test --out "$EV/test_split/$(t "$m").json"
done
step compare_modes $CPU -m evaluation.compare_modes --joint $SLD/joint/params-best.pkl \
  --independent $SLD/independent/params-best.pkl --sequential $SLD/sequential/params-best.pkl \
  --out $EV/test_split/compare_modes.json
for m in "${CONCEPT_ALL[@]}"; do
  step "influence_test_$(t "$m")" $CPU -m evaluation.concept_influence --models "$m" --out_dir $EV/influence
done
for m in "${MAIN[@]}"; do
  step "influence_selfplay_$(t "$m")" $CPU -m evaluation.concept_influence --models "$m" --source selfplay \
    --random_vulnerability --out_dir $EV/influence
done
step concept_quality $CPU -m evaluation.concept_f1 --models "${CONCEPT_ALL[@]}" --out $EV/concepts/concept_quality.json
for m in "${MAIN[@]}" "brl SL" "brl RL" "brl RL+FSP" "brl RL+PFSP" "SL plain" "PPO plain unanchored"; do
  step "counterfactuals_$(t "$m")" $CPU -m evaluation.counterfactual_quality --models "$m" --n 2000 \
    --out_dir $EV/counterfactuals
done
step descriptor_refit $CPU -m evaluation.suit_descriptor_refit --out $EV/influence/descriptor_refit.json

# explanations, card swaps, attribution
mkdir -p $EV/explanations $EV/card_swaps
for m in "${MAIN[@]}"; do
  step "expl_test_$(t "$m")" $CPU -m evaluation.explanation_faithfulness --model "$m" --out_dir $EV/explanations
  step "expl_selfplay_$(t "$m")" $CPU -m evaluation.explanation_faithfulness --model "$m" --source selfplay \
    --random_vulnerability --out_dir $EV/explanations
done
step card_swaps $CPU -m evaluation.card_swap_study --models "${MAIN[@]}" "brl SL" "brl RL+FSP" "SL plain" --n 1500 \
  --out $EV/card_swaps/card_swap_study.json --md $EV/card_swaps/card_swap_study.md
step attribution_deletion $CPU -m evaluation.attribution_deletion --models "${MAIN[@]}" "brl SL" "brl RL+FSP" "SL plain" \
  --n 1500 --out $EV/card_swaps/attribution_deletion.json --md $EV/card_swaps/attribution_deletion.md

# drift from WBridge5's system, convention cards, completeness, leakage, erasure, probes
mkdir -p $EV/drift $EV/convention_trees $EV/completeness $EV/leakage $EV/erasure $EV/probes
agents=("${BRL[@]}" "${MAIN[@]}" "SL plain" "PPO plain unanchored")
step drift $CPU -m evaluation.convention_drift --deals 3000 --agents "${agents[@]}" --out_dir $EV/drift
step cards $CPU -m evaluation.convention_ranges --deals 6000 --agents "${agents[@]}" \
  --out $EV/drift/convention_cards.json --save_cards nlg/convention_cards
step trees $CPU -m evaluation.convention_cards_from_play --deals 50000 --wb5_deals 50000 \
  --agents "${MAIN[@]}" "${BRL[@]}" "SL plain" "PPO plain unanchored" --out_dir $EV/convention_trees
step card_report $CPU -m evaluation.convention_card_figures --cards_dir $EV/convention_trees \
  --thesis_dir $EV/convention_trees/thesis
step card_ebu $CPU -m evaluation.convention_card_ebu_form --cards_dir $EV/convention_trees
step completeness $CPU -m evaluation.concept_completeness --models "${CONCEPT_ALL[@]}" "${BRL[@]}" "${PLAIN[@]}" \
  --out $EV/completeness/completeness.json
step leakage $CPU -m evaluation.concept_leakage --models "${CONCEPT_ALL[@]}" --out $EV/leakage/leakage.json
step erasure $CPU -m evaluation.concept_erasure --models "${BRL[@]}" "SL plain" "${MAIN[@]}" "PPO plain unanchored" \
  --out $EV/erasure/erasure.json
step erasure_seeds $CPU -m evaluation.concept_erasure --models "SL plain s1" "SL plain s2" "SL joint s1" "SL joint s2" \
  --layers 4 --no_per_concept --out $EV/erasure/erasure_seeds.json
step probes $CPU -m evaluation.concept_probes --out $EV/probes/probes.json
step probes_fsp $CPU -m evaluation.concept_probes --brl "brl RL+FSP" --out $EV/probes/probes.json
trunks=()
for m in "SL sequential" "PPO joint" "PPO independent" "PPO sequential" "PPO joint unanchored" \
         "PPO independent unanchored" "PPO sequential unanchored"; do trunks+=("$m (trunk)=$(ckpt "$m")"); done
step probes_trunks $CPU -m evaluation.concept_probes --trunks "${trunks[@]}" --out $EV/probes/probes.json
ptr=()
for m in "SL plain" "SL plain s1" "SL plain s2" "SL joint" "SL joint s1" "SL joint s2"; do ptr+=("$m (trunk)=$(ckpt "$m")"); done
step probes_plain $CPU -m evaluation.concept_probes --trunks "${ptr[@]}" --out $EV/probes/probes_plain.json

# the explainer, follow-up questions, trust calibration, the checker
mkdir -p $EV/qwen $EV/followup $EV/nlg $EV/checker
ind=$SLD/independent/params-best.pkl
adapter=runs/qwen/adapter-best
step qwen_testsplit_data $CPU -m nlg.build_teacher_dataset --model_path $ind \
  --split test --per_category 30 --max_deals 2048 --val_frac 0 --test_frac 1 --out_dir $EV/qwen/testsplit_data
step qwen_adapter_testsplit $GPU -m nlg.eval_qwen --adapter $adapter \
  --labels $EV/qwen/testsplit_data/test_labels.jsonl --frames $EV/qwen/testsplit_data/test_frames.jsonl \
  --per_category 10 --retries 3 --out_dir $EV/qwen/eval_adapter_testsplit
step qwen_base_testsplit $GPU -m nlg.eval_qwen --no_adapter \
  --labels $EV/qwen/testsplit_data/test_labels.jsonl --frames $EV/qwen/testsplit_data/test_frames.jsonl \
  --per_category 10 --retries 0 --out_dir $EV/qwen/eval_base_testsplit
step qwen_models env Q=$EV/qwen/models PY="$PY" bash pipeline/qwen_models.sh
step followup $GPU -m nlg.followup_questions_eval --adapter $adapter \
  --model_path $ind --systems rule tuned base --n 400 --n_hand 150 --out_dir $EV/followup
step trust_calibration bash -c "$CPU -m nlg.followup_questions --calibrate --wb5_card --model_path $ind \
  && cp nlg/trust_calibration.json nlg/wbridge5_card.json $EV/nlg/"
step checker $CPU -m evaluation.checker_coverage --qwen_dir $EV/qwen --out $EV/checker/checker_coverage.json

# the distilled models on their own play; the listener test; natural and artificial bids of every card
step distill_explain $GPU scripts/distill_rl_fsp.py --out $DIST --eval_dir $EV
dj=$($CPU -c "from evaluation.common import DISTILLED; print(DISTILLED['distilled joint'])" 2>/dev/null)
di=$($CPU -c "from evaluation.common import DISTILLED; print(DISTILLED['distilled independent'])" 2>/dev/null)
mkdir -p $EV/bid_meaning
step listener $CPU -m evaluation.bid_meaning_tests swap runs/distill/prepared/test \
  "brl RL+FSP;distilled independent=$di;distilled joint=$dj;brl SL;SL independent" $EV/bid_meaning/listener_fsp.json
cards=()
for f in $EV/convention_trees/*.json; do [ "$(basename "$f")" = wbridge5.json ] || cards+=("$f"); done
for f in $DIST/convention_trees/distilled_*.json; do cards+=("$f"); done
step natural $CPU -m evaluation.bid_meaning_tests artificial $EV/convention_trees/wbridge5.json "${cards[@]}" \
  --out $EV/bid_meaning/natural.json

# measurements for the Discussion, runtime, the worked examples
mkdir -p $EV/remeasure $EV/runtime $EV/runtime_gpu $EV/examples
step remeasure_cuebid env EV=$EV $CPU pipeline/remeasure_cuebid.py
step reason_ranking $CPU pipeline/reason_ranking_effect.py "SL independent,SL joint,SL sequential" 1500 \
  $EV/remeasure/reason_ranking_effect.json
step runtime $CPU -m evaluation.runtime_cost --part all --out $EV/runtime
step runtime_gpu $GPU -m evaluation.runtime_cost --part ppo --out $EV/runtime_gpu
step examples $CPU -m evaluation.thesis_example_figures --eval_dir $EV --out $EV/examples/examples.json \
  --tex_dir $EV/examples/tex
log "evaluation finished: $(ls $EV/done | wc -l) steps done"
