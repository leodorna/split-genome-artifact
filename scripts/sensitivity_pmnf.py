#!/usr/bin/env python3
"""Launch-parameter sensitivity per (op, family, D): measured penalty + PMNF model.

Two complementary measures per knob k in (cap, tx, ty):

  penalty   (model-free, headline)  For every level v of k, re-tune the other two
            knobs: best_k(v) = min { t(c) : c_k = v }.  The penalty is
            max_v best_k(v) / t_best -- the slowdown of the worst setting of k when
            the other knobs are still tuned (1 = the knob does not matter).  Also
            reported: penalty_oat, the same ratio with the other two knobs frozen at
            the best config (one-at-a-time).  The configuration grid is the whole
            design, not a sample, so the 95% CI comes from a bootstrap over seeds.

  elasticity (model)  Extra-P-style PMNF regression of the raw time,
                t(x) = c0 + sum_i c_i * prod_j x_j^{e_ij} * (log2 x_j)^{l_ij},
            with x = (g, tx, ty), where g = min(cap, ceil(U/tx)) is the *effective*
            number of blocks per row (U = units per row): nominal caps above
            ceil(U/tx) launch the identical kernel, so aliased configs are pooled
            into one launch point.  Single terms e in {-2,-1,-0.5,0,0.5,1,2},
            l in {0,1}; pairwise products e in {-1,0,1,2}, l = 0.  Terms by forward
            BIC (OLS on standardized features), coefficients by Huber RLM.
            Elasticity E_k = d log t / d log x_k at the geometric mean of the launch
            points.  95% CI from a bootstrap over launch points that re-selects the
            terms in every replicate (term_stability = share of replicates that pick
            the full-sample term set).  cv_mape = 10-fold CV with term selection
            inside each fold; r2 is in-sample.

Outputs (results/rerun_v3/analysis/):
  sensitivity_coeffs_pmnf.csv
  fig_sensitivity_pmnf.png      (row 1: penalty, row 2: elasticity; per op x family)
"""
import argparse
import csv
import math
import os
import statistics as st
from collections import Counter, defaultdict

import numpy as np
import statsmodels.api as sm
from statsmodels.robust.norms import HuberT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context="paper", style="ticks", font_scale=0.95)
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
KNOBS = ["cap", "tx", "ty"]
KN = {"cap": 0, "tx": 1, "ty": 2}
E_LIST = [-2, -1, -0.5, 0.5, 1, 2]
PROD_E = [-1, 0, 1, 2]
LN2 = math.log(2.0)
OPS = ["de-gen", "ga-gen", "ux-bf"]
FAMS = [("highdim", "scalar"), ("highdim_vec4", "vec4")]


# ---------- term library ----------
def library():
    terms = []
    for k in range(3):
        for e in E_LIST + [0]:
            for l in (0, 1):
                if e == 0 and l == 0:
                    continue
                terms.append(((k, e, l),))
    for a in range(3):
        for b in range(a + 1, 3):
            for e1 in PROD_E:
                for e2 in PROD_E:
                    if e1 == 0 and e2 == 0:
                        continue
                    terms.append(((a, e1, 0), (b, e2, 0)))
    return terms


TERMS = library()


def tval(term, x):
    v = 1.0
    for (k, e, l) in term:
        v *= (x[k] ** e) * (math.log2(x[k]) ** l)
    return v


def tderiv(term, x, dk):
    """d term / d x_{dk}."""
    hit = [f for f in term if f[0] == dk]
    if not hit:
        return 0.0
    k, e, l = hit[0]
    xk = x[dk]
    if l == 0:
        df = e * (xk ** (e - 1))
    else:
        df = (xk ** (e - 1)) * (e * math.log2(xk) + 1.0 / LN2)
    v = 1.0
    for (kk, ee, ll) in term:
        if kk == dk:
            continue
        v *= (x[kk] ** ee) * (math.log2(x[kk]) ** ll)
    return v * df


def units_per_row(op, fam, D):
    binary = op == "ux-bf"
    if fam == "vec4":
        return D // 128 if binary else D // 4
    return D // 32 if binary else D


def load(path):
    """(op, fam, I, N, D, cap, tx, ty) -> {seed: median over reps}."""
    per = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(path)):
        if r["status"] != "ok":
            continue
        fam = "vec4" if r["layout"].endswith("vec4") else "scalar"
        k = (r["op"], fam, int(r["I"]), int(r["N"]), int(r["D"]),
             int(r["cap"]), int(r["tx"]), int(r["ty"]))
        per[k][r["seed"]].append(float(r["time_us"]))
    return {k: {s: st.median(v) for s, v in d.items()} for k, d in per.items()}


# ---------- PMNF fit ----------
def design(points):
    """Raw term matrix for launch points [(g, tx, ty), ...]."""
    return np.array([[tval(t, p) for t in TERMS] for p in points])


def select_terms(Z, y, max_terms=6):
    """Forward selection by BIC; same criterion as statsmodels OLS().bic (up to a constant)."""
    n = len(y)
    max_terms = min(max_terms, n - 3)  # keep >= 2 residual dof for the Huber scale
    cand = [j for j in range(Z.shape[1]) if np.std(Z[:, j]) > 1e-12]

    def bic(cols):
        X = np.column_stack([np.ones(n)] + [Z[:, c] for c in cols])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        rss = max(float(np.sum((y - X @ beta) ** 2)), 1e-300)
        return n * math.log(rss / n) + X.shape[1] * math.log(n)

    chosen, best = [], bic([])
    while len(chosen) < max_terms:
        scores = [(bic(chosen + [j]), j) for j in cand if j not in chosen]
        if not scores:
            break
        b, j = min(scores)
        if b < best - 1e-6:
            chosen.append(j); best = b
        else:
            break
    return chosen


def huber(Z, y, sel):
    return sm.RLM(y, sm.add_constant(Z[:, sel], has_constant="add"), M=HuberT()).fit()


def predict(r, Z, sel):
    return sm.add_constant(Z[:, sel], has_constant="add") @ r.params


def elasticities(r, sel, mu, sd, ref, active):
    """E_k = (x_k / t) dt/dx_k at ref, from the fit on standardized features."""
    gamma = r.params[1:] / sd[sel]
    t = r.params[0] + sum(gamma[i] * (tval(TERMS[j], ref) - mu[j]) for i, j in enumerate(sel))
    E = {}
    for kn in KNOBS:
        if not active[kn]:
            E[kn] = float("nan")
            continue
        d = sum(gamma[i] * tderiv(TERMS[j], ref, KN[kn]) for i, j in enumerate(sel))
        E[kn] = (ref[KN[kn]] / t) * d
    return E


# ---------- model-free penalty ----------
def penalties(t):
    """t: {(cap, tx, ty): time}. Returns {knob: (penalty_retuned, penalty_oat)}."""
    best = min(t, key=t.get)
    tb = t[best]
    out = {}
    for i, kn in enumerate(KNOBS):
        levels = sorted({c[i] for c in t})
        retuned = max(min(v for c, v in t.items() if c[i] == lv) for lv in levels) / tb
        oat = [v for c, v in t.items() if all(c[j] == best[j] for j in range(3) if j != i)]
        out[kn] = (retuned, max(oat) / tb)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default=os.path.join(RES, "rerun_v3", "phase3_unified_final_v3.csv"))
    ap.add_argument("--dir", default=os.path.join(RES, "rerun_v3", "analysis"))
    ap.add_argument("-B", "--boot", type=int, default=200)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    data = load(a.final)

    groups = defaultdict(list)
    for k in data:
        groups[(k[0], k[1], k[2], k[3], k[4])].append(k)  # op, family, I, N, D

    rows = []
    series = defaultdict(dict)  # (op, fam) -> D -> {knob: {...}}
    for (op, fam, I, N, D) in sorted(groups, key=lambda g: (g[0], g[1], g[4])):
        keys = groups[(op, fam, I, N, D)]
        U = units_per_row(op, fam, D)
        seeds = sorted({s for k in keys for s in data[k]})

        # --- measured penalty, seed-bootstrap CI
        t_cfg = {k[5:8]: st.median(data[k].values()) for k in keys}
        pen = penalties(t_cfg)
        pboot = {kn: [] for kn in KNOBS}
        for _ in range(a.boot):
            pick = rng.choice(seeds, size=len(seeds))
            tb = {k[5:8]: st.median([data[k][s] for s in pick if s in data[k]] or data[k].values())
                  for k in keys}
            for kn, (p, _) in penalties(tb).items():
                pboot[kn].append(p)

        # --- PMNF on effective launch points (aliased caps pooled)
        pooled = defaultdict(list)
        for k in keys:
            cap, tx, ty = k[5:8]
            pooled[(min(cap, math.ceil(U / tx)), tx, ty)].append(t_cfg[(cap, tx, ty)])
        pts = sorted(pooled)
        y = np.array([st.median(pooled[p]) for p in pts])
        Xraw = design(pts)
        mu = Xraw.mean(axis=0); sd = Xraw.std(axis=0); sd[sd == 0] = 1.0
        Z = (Xraw - mu) / sd
        active = {kn: len({p[KN[kn]] for p in pts}) > 1 for kn in KNOBS}
        ref = [2.0 ** st.mean(math.log2(p[i]) for p in pts) for i in range(3)]

        sel = select_terms(Z, y)
        r = huber(Z, y, sel)
        E = elasticities(r, sel, mu, sd, ref, active)
        r2 = 1 - np.sum((y - r.fittedvalues) ** 2) / np.sum((y - y.mean()) ** 2)

        # 10-fold CV, terms re-selected inside each fold
        folds = np.array_split(rng.permutation(len(pts)), min(a.folds, len(pts)))
        prd = np.empty(len(pts))
        for f in folds:
            m = np.ones(len(pts), bool); m[f] = False
            s_f = select_terms(Z[m], y[m])
            prd[f] = predict(huber(Z[m], y[m], s_f), Z[f], s_f)
        cv_mape = float(np.mean(np.abs(prd - y) / y))

        # bootstrap over launch points, terms re-selected per replicate
        eboot = {kn: [] for kn in KNOBS}
        sets = Counter()
        for _ in range(a.boot):
            idx = rng.integers(len(pts), size=len(pts))
            s_b = select_terms(Z[idx], y[idx])
            sets[tuple(sorted(s_b))] += 1
            try:
                Eb = elasticities(huber(Z[idx], y[idx], s_b), s_b, mu, sd, ref, active)
            except (np.linalg.LinAlgError, ValueError):
                continue
            for kn in KNOBS:
                eboot[kn].append(Eb[kn])
        stability = sets[tuple(sorted(sel))] / a.boot

        terms_str = " ; ".join(
            "*".join(f"{('g' if k == 0 else KNOBS[k])}^{e}" + ("*log2" if l else "") for (k, e, l) in TERMS[j])
            for j in sel)
        for kn in KNOBS:
            pl, ph = np.percentile(pboot[kn], [2.5, 97.5])
            ev = [v for v in eboot[kn] if not math.isnan(v)]
            el, eh = np.percentile(ev, [2.5, 97.5]) if ev else (float("nan"), float("nan"))
            nlev = len({k[5 + KN[kn]] for k in keys})
            rows.append([op, fam, I, N, D, kn,
                         f"{pen[kn][0]:.4f}", f"{pl:.4f}", f"{ph:.4f}", f"{pen[kn][1]:.4f}", nlev,
                         f"{E[kn]:+.4f}", f"{el:+.4f}", f"{eh:+.4f}",
                         len(keys), len(pts), len(seeds), f"{r2:.3f}", f"{cv_mape:.4f}",
                         f"{stability:.3f}", terms_str])
            series[(op, fam)].setdefault(D, {})[kn] = dict(
                pen=(pen[kn][0], pl, ph), el=(E[kn], el, eh))
        print(f"{op:7}{fam:7}D=2^{D.bit_length()-1:<2} configs={len(keys):3d} launches={len(pts):3d} "
              f"R2={r2:.2f} CV-MAPE={cv_mape:.3f} stab={stability:.2f}  "
              + "  ".join(f"{kn}: pen {pen[kn][0]:.2f}x E {E[kn]:+.3f}" for kn in KNOBS), flush=True)

    out_csv = os.path.join(a.dir, "sensitivity_coeffs_pmnf.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "family", "I", "N", "D", "knob",
                    "penalty", "pen_ci_lo", "pen_ci_hi", "penalty_oat", "n_levels",
                    "elasticity", "ci_lo", "ci_hi",
                    "n_configs", "n_launches", "n_seeds", "r2", "cv_mape", "term_stability", "terms"])
        w.writerows(rows)

    KCOL = {"cap": "#2a78d6", "tx": "#eb6834", "ty": "#1baf7a"}
    fig, axes = plt.subplots(2, 6, figsize=(18.0, 6.4), sharex=True)
    for c_, (op, (lay, fam)) in enumerate([(o, f) for o in OPS for f in FAMS]):
        for r_, (m, ylab) in enumerate([("pen", "penalty (×)\nworst level, others re-tuned"),
                                        ("el", "PMNF elasticity\n$\\partial\\log t/\\partial\\log x$")]):
            ax = axes[r_, c_]
            xs = sorted(series[(op, fam)])
            for kn in KNOBS:
                v = np.array([series[(op, fam)][D][kn][m] for D in xs], dtype=float)
                err = np.abs(np.vstack([v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]]))
                ax.errorbar([math.log2(D) for D in xs], v[:, 0], yerr=err, fmt="o-", color=KCOL[kn],
                            capsize=2, ms=3.5, lw=1.2, label=kn)
            ax.axhline(1 if m == "pen" else 0, color="0.3", lw=0.8, ls=":")
            if m == "pen":
                ax.set_yscale("log")
                ax.set_title(f"{op} / {fam}", fontsize=9)
            ax.set_xticks([10, 12, 14, 15, 17, 19, 20, 22, 24])
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.25, lw=0.3); sns.despine(ax=ax)
            if c_ == 0:
                ax.set_ylabel(ylab, fontsize=8)
            if r_ == 1:
                ax.set_xlabel("$\\log_2 D$", fontsize=8)
    axes[0, 0].legend(fontsize=7, title="knob", title_fontsize=7)
    fig.suptitle("Launch-parameter sensitivity: measured penalty (95% seed-bootstrap CI) and "
                 "PMNF elasticity on effective launches (95% config-bootstrap CI)", fontsize=10)
    fig.tight_layout()
    p = os.path.join(a.dir, "fig_sensitivity_pmnf.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote", p)
    print("wrote", out_csv)


if __name__ == "__main__":
    main()
