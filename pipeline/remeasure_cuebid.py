"""How often an explanation calls a bid in the opponents' suit "a strong, forcing call" for a hand under 10 HCP
(Discussion, "Meanings come from the rules"): in the explainer's training data and in the test-split explanations of
the evaluation in $EV (default runs/evaluation). Run from the repository root."""
import json, os, pickle, sys
sys.path.insert(0, ".")
from evaluation.common import load_test_split
from concepts.hand import decode_hand
T = 10  # "weak": under 10 HCP
# the explainer's training data (labels + frames, line by line)
L = [json.loads(l) for l in open("runs/explainer_data/teacher/train_labels.jsonl")]
F = [json.loads(l) for l in open("runs/explainer_data/teacher/train_frames.jsonl")]
n = sum(1 for l, f in zip(L, F) if "strong, forcing" in l["explanation"] and f["hcp"] < T)
print(f"training: {n} of {len(L)} = {100*n/len(L):.3f}%  (all strong-forcing: {sum('strong, forcing' in l['explanation'] for l in L)})")
pos = load_test_split()
for m in ("SL_independent", "SL_joint"):
    d = pickle.load(open(f"{os.environ.get('EV', 'runs/evaluation')}/explanations/{m}__test_records.pkl", "rb"))
    for part in ("uniform", "stratified"):
        hits = [r for r in d[part] if "strong, forcing" in r["text"] and decode_hand(pos.obs[int(r["row"])]).hcp < T]
        print(m, part, len(hits), "of", len(d[part]), "same as WBridge5:", sum(r["bid"] == r["wb5_bid"] for r in hits))
