#!/usr/bin/env python3
"""LaTeX table of the launch configurations (cap, tx, ty) behind fig:evox.

Reads results/rerun_v3/analysis/augmented_vs_evox.csv (plot_evox_comparison.py) and writes
results/rerun_v3/analysis/evox_launch_configs.tex: one table* with a row pair per (I, N, D)
cell -- the tuned (best) configuration and the heuristic's -- and one column per
operator and variant.
"""
import argparse
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = [(op, fam) for op in ("de-gen", "ga-gen", "ux-bf") for fam in ("scalar", "vec4")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(HERE, "..", "results", "rerun_v3", "analysis", "augmented_vs_evox.csv"))
    ap.add_argument("--tex", default=os.path.join(HERE, "..", "results", "rerun_v3", "analysis", "evox_launch_configs.tex"))
    a = ap.parse_args()

    cfg, cells = {}, set()
    for r in csv.DictReader(open(a.csv)):
        # D_cell: the cell's D in 32-bit words (ux-bf runs it with 32x the genes, as bits)
        k = (r["op"], r["family"], int(r["I"]), int(r["N"]), int(r.get("D_cell") or r["D"]))
        cfg[k] = ((r["cap"], r["tx"], r["ty"]), (r["heur_cap"], r["heur_tx"], r["heur_ty"]))
        cells.add(k[2:])
    cells = sorted(cells, key=lambda c: c[2])

    fmt = lambda c: "%s, %s, %s" % c  # noqa: E731
    L = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
         r"\caption{Launch configurations $(s, t_x, t_y)$ of the kernels in "
         r"Figure~\ref{fig:evox}. For each $(I, N, D)$ cell, \emph{tuned} is the best "
         r"configuration of the ablation grid and \emph{heuristic} the configuration of "
         r"Equation~\eqref{eq:heuristic}. The kernel launches "
         r"$\min(s, \lceil U/t_x \rceil)$ blocks per genome, so a split factor above "
         r"that value has no effect. \texttt{ux-bf} runs each cell with $32D$ genes (bits), "
         r"the same memory. -- marks a cell without a valid launch.}",
         r"\label{tab:evox-configs}",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{} rrr l cc cc cc @{}}",
         r"\toprule",
         r"& & & & \multicolumn{2}{c}{\texttt{de}} & \multicolumn{2}{c}{\texttt{sbx-pm}} "
         r"& \multicolumn{2}{c}{\texttt{ux-bf}} \\",
         r"\cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10}",
         r"$D$ & $I$ & $N$ & & scalar & \texttt{vec4} & scalar & \texttt{vec4} & scalar & \texttt{vec4} \\",
         r"\midrule"]
    for n, (I, N, D) in enumerate(cells):
        for j, (name, which) in enumerate((("tuned", 0), ("heuristic", 1))):
            vals = [fmt(cfg[(op, fam, I, N, D)][which]) if (op, fam, I, N, D) in cfg else "--"
                    for op, fam in COLS]
            head = (r"$2^{%d}$ & %d & %d" % (D.bit_length() - 1, I, N)) if j == 0 else "& &"
            L.append(f"{head} & {name} & " + " & ".join(vals) + r" \\")
        if n < len(cells) - 1:
            L.append(r"\addlinespace[2pt]")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    with open(a.tex, "w") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", a.tex)


if __name__ == "__main__":
    main()
