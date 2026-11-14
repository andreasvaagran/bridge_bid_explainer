"""The experiments chapter's figures (<out_dir>/figures/*.pdf, .png) and tables (<out_dir>/tables/<label>.tex, each
\\input where the chapter discusses it), rebuilt from the evaluations' JSON outputs in --eval_dir:

    python -m evaluation.report --eval_dir runs/evaluation --out_dir <thesis>/results

The convention cards (evaluation/convention_card_figures.py, convention_card_ebu_form.py) and the worked examples
(evaluation/thesis_example_figures.py) write their own outputs."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from evaluation.report import rq1_cost, rq1_faithful, rq2_words, rq3_convention, rq3_gain

SECTIONS = [rq1_cost, rq1_faithful, rq2_words, rq3_convention, rq3_gain]   # chapter 4's sections in order


def fit_width(table: str) -> str:
    """A table never wider than the text block (\\usepackage{adjustbox} in the thesis preamble)."""
    if "adjustbox" in table:   # the heatmaps box themselves (their cells nest tabulars)
        return table
    return re.sub(r"(\\begin\{tabular\}.*?\\end\{tabular\})",
                  lambda m: "\\begin{adjustbox}{max width=\\linewidth}\n" + m.group(1) + "\n\\end{adjustbox}",
                  table, flags=re.S)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_dir", default="runs/evaluation")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--only", nargs="*", help="figure or table function names")
    args = ap.parse_args()
    ev, out = Path(args.eval_dir), Path(args.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    for section in SECTIONS:
        for f in section.FIGURES:
            if not args.only or f.__name__ in args.only:
                f(ev, out / "figures")
                print(f"[report] {f.__name__} done", flush=True)
        for f in section.TABLES:
            if args.only and f.__name__ not in args.only:
                continue
            for t in f(ev):
                label = re.search(r"\\label\{tab:([\w-]+)\}", t).group(1)
                (out / "tables" / f"{label}.tex").write_text(fit_width(t))
                print(f"[report] tab:{label} done", flush=True)


if __name__ == "__main__":
    main()
