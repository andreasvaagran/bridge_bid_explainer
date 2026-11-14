"""How ranking reasons by their contribution over all alternatives (nlg.explanation_inputs.contribution_key), rather
than by the margin over the runner-up alone, changes the explanations' reasons on held-out WBridge5 decisions: the
share with at least one hand-dependent reason, and the share whose only reason restates the bid.

    python pipeline/reason_ranking_effect.py "SL independent,SL joint" 1500 out.json
"""
import sys, json
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
from evaluation.common import MODELS, SEATS, load_test_split, sample_rows
from nlg.explanation_frames import decision_frame
from nlg.explanation_inputs import hand_dependent_concepts, select_reasoned_concepts
from nlg.decision_record import Bidder

pos = load_test_split()
rows = sample_rows(pos, int(sys.argv[2]) if len(sys.argv) > 2 else 1500, seed=1)
out = {}
for model in sys.argv[1].split(","):
    b = Bidder(str(MODELS[model]))
    stats = {"decisions": 0, "confident": 0}
    for key in ("contribution", "contribution_alt"):
        stats[key] = {"hand_reason": 0, "only_restatement": 0, "no_reason": 0}
    for i in rows:
        obs = pos.obs[i].astype(np.float32); legal = pos.legal[i]
        o = b.outputs(obs)
        before, seat, dl = pos.auction(i)
        fr = decision_frame(b.enricher, obs, legal, o, int(pos.deal[i]), int(pos.step[i]), seat, SEATS[dl], dl, "None",
                            before, counterfactuals=False)
        stats["decisions"] += 1
        if (fr.get("runner_up") or {}).get("p", 1) < 0.05:
            stats["confident"] += 1
        dep = hand_dependent_concepts(fr)
        for key in ("contribution", "contribution_alt"):
            # contribution: the runner-up margin only; contribution_alt: over all alternatives, as deployed
            f2 = fr if key == "contribution_alt" else dict(fr, concept_evidence=[
                {k: v for k, v in e.items() if k != "contribution_alt"} for e in fr["concept_evidence"]])
            rs = select_reasoned_concepts(f2)
            if not rs:
                stats[key]["no_reason"] += 1
            elif any(dep.get(r) for r in rs):
                stats[key]["hand_reason"] += 1
            else:
                stats[key]["only_restatement"] += 1
    out[model] = stats
    print(model, json.dumps(stats), flush=True)
json.dump(out, open(sys.argv[3] if len(sys.argv) > 3 else "reason_ranking_effect.json", "w"), indent=1)
