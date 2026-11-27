#!/usr/bin/env python3
"""Measure the speedup of one case over the baseline and compare it with the paper.

A case is an operator, a variant, an objective function and a cell (I, N, D), with D as
in Table 2 of the paper (genes per individual; ux-bf runs 32 D bits, the same memory).
The script times our kernel (code/bin/bench_generation_v2) at the heuristic's launch
configuration, the paper's tuned one or a given one, then the island-local baseline
(python/step_bench, torch.compile max-autotune-no-cudagraphs), and prints both times,
the speedup and the paper's value for the same case. It prints the exact command of
each side and the time of every timed generation of every seed, and writes them to
check_speedup/<case>.times.csv and <case>.commands.txt.

    scripts/check_speedup.py --op sbx-pm --variant vec4 --function sphere --I 32 --N 256 --D 131072
    scripts/check_speedup.py --op de --function cec-f12 --I 1024 --N 64 --D 1000 --config tuned
    scripts/check_speedup.py --op ux-bf --function knapsack --I 32 --N 256 --D 131072 --config 4,512,1

Operators: the paper's names de, sbx-pm and ux-bf (the code's de-gen and ga-gen are also
accepted). Functions: sphere (de, sbx-pm), onemax (ux-bf), cec-f1, cec-f2, cec-f3, cec-f12
(de, sbx-pm; generated shift, as in the timing runs) and knapsack (ux-bf).
The paper's values come from results/speedup_reference.csv (written with
--write-reference from the analysis outputs). On the paper's GPU the check passes when
the measured speedup is within --tol of the paper's; elsewhere it only reports.
Needs an idle GPU, the built kernels (make -C code bench) and torch + evox for the
baseline (EVOX_SRC may point to an EvoX checkout). Exit status: 0 pass or reported,
1 outside the tolerance, 2 usage or run error.
"""
import argparse
import csv
import os
import statistics as st
import subprocess
import sys
import shlex
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import launch_heuristic as lh  # noqa: E402

REF = os.path.join(ROOT, "results", "speedup_reference.csv")
PAPER_GPU = "RTX PRO 6000"
FUNCS = {"de-gen": ("sphere", "cec-f1", "cec-f2", "cec-f3", "cec-f12"),
         "ga-gen": ("sphere", "cec-f1", "cec-f2", "cec-f3", "cec-f12"),
         "ux-bf": ("onemax", "knapsack")}
PAPER = {"de-gen": "de", "ga-gen": "sbx-pm", "ux-bf": "ux-bf"}   # code name -> paper name
MODE = "max-autotune-no-cudagraphs"
USTAR, OCC = 9, True   # the paper's heuristic (Equation 7): u* = 9 and the occupancy term b


def write_reference():
    """results/speedup_reference.csv from the outputs of reproduce_figures.sh."""
    rows = []
    sph = os.path.join(ROOT, "results", "rerun_v3", "analysis", "augmented_vs_evox.csv")
    if not os.path.exists(sph):
        sys.exit(f"missing {sph}: run ./reproduce_figures.sh first")
    for r in csv.DictReader(open(sph)):
        rows.append(dict(function="onemax" if r["op"] == "ux-bf" else "sphere", op=r["op"],
                         variant=r["family"], I=r["I"], N=r["N"], D=r["D_cell"],
                         heur_cfg=f"{r['heur_cap']},{r['heur_tx']},{r['heur_ty']}",
                         tuned_cfg=f"{r['cap']},{r['tx']},{r['ty']}",
                         ours_heur_us=r["heur_us"], ours_tuned_us=r["tuned_us"],
                         baseline_us=r["evox_island_us"], speedup_heur=r["speedup_heuristic"],
                         speedup_tuned=r["speedup_tuned"]))
    for r in csv.DictReader(open(os.path.join(ROOT, "results", "transfer", "transfer_cells.csv"))):
        D = int(r["D"]) // 32 if r["op"] == "ux-bf" else int(r["D"])
        rows.append(dict(function=r["fitness"], op=r["op"], variant=r["family"], I=r["I"], N=r["N"],
                         D=D, heur_cfg=r["heur_cfg"].replace("/", ","),
                         tuned_cfg=r["tuned_cfg"].replace("/", ","), ours_heur_us=r["heur_us"],
                         ours_tuned_us=r["tuned_us"], baseline_us=r["evox_us"],
                         speedup_heur=r["speedup_heur"], speedup_tuned=r["speedup_tuned"]))
    with open(REF, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {REF} ({len(rows)} cases)")


def reference(a):
    if not os.path.exists(REF):
        return None
    for r in csv.DictReader(open(REF)):
        if (r["function"], r["op"], r["variant"], int(r["I"]), int(r["N"]), int(r["D"])) == \
                (a.function, a.op, a.variant, a.I, a.N, a.D):
            return r
    return None


def gpu():
    try:
        import torch
        p = torch.cuda.get_device_properties(0)
        return p.name, p.multi_processor_count
    except Exception:
        name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip().splitlines()
        return (name[0] if name else "unknown"), None


def run_side(a, side, cmds, cwd=None, env=None):
    """Run one command per seed; each writes the time of every timed generation to a file.
    Returns the per-seed lists of times (us), or (None, error)."""
    per_seed = []
    for s, cmd in enumerate(cmds):
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
            cmd = cmd + ["--dump-times", tmp.name]
            a.log_cmd.write(f"# {side}, seed {s}\n{shlex.join(cmd)}\n")
            p = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)
            times = [float(x) for x in open(tmp.name).read().split()] if p.returncode == 0 else []
        if not times:
            err = (p.stderr.strip().splitlines() or p.stdout.strip().splitlines() or [f"exit {p.returncode}"])[-1]
            return None, err
        per_seed.append(times)
    return per_seed, None


def ours_cmds(a, cfg, bits):
    cap, tx, ty = cfg
    layout = "highdim_vec4" if a.variant == "vec4" else "highdim"
    fit = [] if a.function in ("sphere", "onemax") else ["--fitness", a.function]
    return [[a.bin, "--mode", "generation", "--algo", a.op, "--layout", layout,
             "--islands", str(a.I), "--n", str(a.N), "--d", str(bits),
             "--xthreads", str(tx), "--yblock", str(ty), "--hd-cap", str(cap),
             "--reps", "10", "--warmup", "5", "--seed", str(42 + s * 1000003)] + fit
            for s in range(a.seeds)]


def baseline_cmds(a, bits):
    cmds = []
    for s in range(a.seeds):
        cmd = [sys.executable, "-m", "step_bench.bench_generation", "--op", a.op,
               "--n", str(a.I * a.N), "--d", str(bits), "--islands", str(a.I), "--seed", str(s),
               "--generations", "20", "--warmup", "5", "--compiled-only", "--compile-mode", MODE]
        if a.function not in ("sphere", "onemax"):   # the defaults: Sphere, and OneMax for ux-bf
            cmd += ["--problem", a.function]
        cmds.append(cmd)
    return cmds


def report(a, side, per_seed):
    """Print every timed generation of every seed and write them to the raw CSV."""
    meds = [st.median(t) for t in per_seed]
    for s, t in enumerate(per_seed):
        print(f"  seed {s}: {len(t)} generations, median {meds[s]:.1f} us, min {min(t):.1f}, max {max(t):.1f}")
        print("    " + " ".join(f"{x:.1f}" for x in t))
        for g, x in enumerate(t):
            a.log_csv.writerow([side, s, g, f"{x:.3f}"])
    m = st.median(meds)
    print(f"  {side}: {m:.1f} us per generation (median over {len(meds)} seeds of the per-seed medians)")
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--op", choices=["de", "sbx-pm", "ux-bf", "de-gen", "ga-gen"],
                    help="the paper's names de, sbx-pm, ux-bf (or the code's de-gen, ga-gen)")
    ap.add_argument("--variant", choices=["scalar", "vec4"], default="scalar")
    ap.add_argument("--function", default=None, help="default: sphere, or onemax for ux-bf")
    ap.add_argument("--I", type=int)
    ap.add_argument("--N", type=int)
    ap.add_argument("--D", type=int, help="genes per individual as in Table 2 (ux-bf: 32 D bits)")
    ap.add_argument("--config", default="heuristic", help="heuristic | tuned | s,tx,ty")
    ap.add_argument("--seeds", type=int, default=3, help="seeds per side (the paper: 10 baseline, 5-10 ours)")
    ap.add_argument("--sm", type=int, default=None, help="SM count for the heuristic (default: this GPU's)")
    ap.add_argument("--tol", type=float, default=0.15, help="relative tolerance on the paper's GPU")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "check_speedup"),
                    help="where the raw times and the exact commands are written")
    ap.add_argument("--bin", default=os.path.join(ROOT, "code", "bin", "bench_generation_v2"))
    ap.add_argument("--write-reference", action="store_true", help="write results/speedup_reference.csv and exit")
    a = ap.parse_args()
    if a.write_reference:
        return write_reference()
    if None in (a.op, a.I, a.N, a.D):
        ap.error("--op, --I, --N and --D are required")
    a.op = {"de": "de-gen", "sbx-pm": "ga-gen"}.get(a.op, a.op)
    a.function = a.function or ("onemax" if a.op == "ux-bf" else "sphere")
    if a.function not in FUNCS[a.op]:
        ap.error(f"{PAPER[a.op]} runs {', '.join(FUNCS[a.op])}")
    if not os.access(a.bin, os.X_OK):
        sys.exit(f"missing {a.bin}: build the kernels with make -C code bench")
    bits = 32 * a.D if a.op == "ux-bf" else a.D
    name, sm = gpu()
    sm = a.sm or sm
    ref = reference(a)

    if a.config == "heuristic":
        if sm is None:
            sys.exit("cannot read the SM count without torch: pass --sm")
        cfg = lh.launch_config(a.op, a.variant, a.I, a.N, bits, sm, 1.0, USTAR, OCC)
    elif a.config == "tuned":
        if ref is None:
            sys.exit("no tuned configuration for this case in results/speedup_reference.csv")
        cfg = tuple(int(x) for x in ref["tuned_cfg"].split(","))
    else:
        cfg = tuple(int(x) for x in a.config.split(","))
    if a.config == "heuristic" and ref is not None and PAPER_GPU in name \
            and ",".join(map(str, cfg)) != ref["heur_cfg"]:
        print(f"note: the paper's heuristic configuration for this case is {ref['heur_cfg']} "
              "(a narrower block where the register limit forbids the rule's)")
    print(f"GPU: {name} ({sm} SMs)")
    code = f" (--algo/--op {a.op} in the commands below)" if PAPER[a.op] != a.op else ""
    print(f"case: {PAPER[a.op]}{code} {a.variant}, {a.function}, I={a.I} N={a.N} D={a.D}"
          + (f" ({bits} bits)" if a.op == "ux-bf" else "") + f", {a.seeds} seeds per side")

    tag = f"{PAPER[a.op]}_{a.variant}_{a.function}_I{a.I}_N{a.N}_D{a.D}_{a.config.replace(',', '-')}"
    os.makedirs(a.outdir, exist_ok=True)
    a.log_cmd = open(os.path.join(a.outdir, tag + ".commands.txt"), "w")
    raw = open(os.path.join(a.outdir, tag + ".times.csv"), "w", newline="")
    a.log_csv = csv.writer(raw)
    a.log_csv.writerow(["side", "seed", "generation", "time_us"])
    print("timing, both sides: CUDA events around one generation, after warm-up, excluding "
          "compilation;\n  ours: one fused-kernel launch per generation, 5 warm-up + 10 timed; "
          f"baseline: one compiled step, 5 warm-up + 20 timed")

    while True:
        print(f"\nours: s={cfg[0]} t_x={cfg[1]} t_y={cfg[2]} ({a.config})", flush=True)
        cmds = ours_cmds(a, cfg, bits)
        print(f"  command (seed 0): {shlex.join(cmds[0])}")
        per_seed, err = run_side(a, "ours", cmds)
        if per_seed is not None:
            break
        # as in the analysis: a block that does not fit the registers falls back to a narrower one
        if a.config == "heuristic" and cfg[1] > 32:
            print(f"  not launchable ({err}); trying t_x={cfg[1] // 2}")
            cfg = (cfg[0], cfg[1] // 2, cfg[2])
            continue
        sys.exit(f"kernel run failed: {err}")
    ours = report(a, "ours", per_seed)

    print(f"\nbaseline: island-local EvoX, torch.compile {MODE} (compiled once per seed)", flush=True)
    env = dict(os.environ)
    py = os.path.join(ROOT, "python")
    env["PYTHONPATH"] = py + (":" + env["EVOX_SRC"] if env.get("EVOX_SRC") else "")
    cmds = baseline_cmds(a, bits)
    print(f"  command (seed 0, in python/): {shlex.join(cmds[0])}")
    per_seed, err = run_side(a, "baseline", cmds, cwd=py, env=env)
    if per_seed is None:
        sys.exit(f"baseline run failed: {err}")
    base = report(a, "baseline", per_seed)
    raw.close()
    a.log_cmd.close()

    x = base / ours
    print(f"\nspeedup = baseline / ours = {base:.1f} / {ours:.1f} = {x:.2f}x")
    print(f"raw times: {os.path.join(a.outdir, tag + '.times.csv')}; commands: "
          f"{os.path.join(a.outdir, tag + '.commands.txt')}")

    if ref is None:
        print("paper: this case is not among the measured ones (results/speedup_reference.csv)")
        return 0
    key = {"heuristic": "speedup_heur", "tuned": "speedup_tuned"}.get(a.config)
    if key is None:
        print(f"paper: heuristic {float(ref['speedup_heur']):.2f}x ({ref['heur_cfg']}), "
              f"tuned {float(ref['speedup_tuned']):.2f}x ({ref['tuned_cfg']}); "
              f"baseline {float(ref['baseline_us']):.1f} us")
        return 0
    px = float(ref[key])
    ptime = float(ref["ours_heur_us" if a.config == "heuristic" else "ours_tuned_us"])
    print(f"paper ({PAPER_GPU}): ours {ptime:.1f} us, baseline {float(ref['baseline_us']):.1f} us, "
          f"speedup {px:.2f}x; measured / paper = {x / px:.2f}")
    if PAPER_GPU not in name:
        print("not the paper's GPU: reported only")
        return 0
    ok = abs(x / px - 1) <= a.tol
    print(f"{'PASS: within' if ok else 'FAIL: outside'} {100 * a.tol:.0f}% of the paper's speedup")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
