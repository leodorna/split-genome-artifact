#!/usr/bin/env python3
"""Parse the roofline profiles (ncu_roofline_v3.sh, ncu_ceilings_v3.sh).

Writes, into --dir:
  roofline_ceilings_v3.csv  one row per ceiling kernel: time, clocks, DRAM/L2 bytes,
                            warp instructions, and the rates they give
                            (DRAM_GBps, L2_GBps, warp_GIPS, IPC per SM)
  roofline_v3.csv           one row per profiled config: the kernels of one
                            generation summed (rates and percentages weighted by
                            kernel time), plus
    inst_per_dram_byte, inst_per_l2_byte   the roofline x-axes
    warp_GIPS, dram_GBps                   attained rates
    sol_dram           dram_GBps / DRAM ceiling (measured bytes)
    sol_issue          warp_GIPS / issue ceiling
    sol_rng            (binary_rng time for the same words) / time, ux-bf only
    bound              the largest of these three speed-of-light fractions
    compulsory_bytes   read the population once and write it once
                       (2 * I * N * D * 4 B; D / 32 words for ux-bf)
    sol_compulsory     (compulsory_bytes / DRAM ceiling) / time. A reference, not
                       a bound: tournament selection never reads the rows it does
                       not pick, so the GAs can move fewer bytes than this.
"""
import argparse
import csv
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_DIR = os.path.join(HERE, "..", "results", "rerun_v3", "roofline")
SUM = {
    "time_ns": "gpu__time_duration.sum",
    "dram_read": "dram__bytes_op_read.sum",
    "dram_write": "dram__bytes_op_write.sum",
    "l2_bytes": "lts__t_bytes.sum",
    "l1_bytes": "l1tex__t_bytes.sum",
    "warp_inst": "smsp__inst_executed.sum",
    "thread_inst": "smsp__thread_inst_executed.sum",
    "pipe_alu": "smsp__inst_executed_pipe_alu.sum",
    "pipe_fma": "smsp__inst_executed_pipe_fma.sum",
    "pipe_fmaheavy": "smsp__inst_executed_pipe_fmaheavy.sum",
    "pipe_xu": "smsp__inst_executed_pipe_xu.sum",
    "pipe_lsu": "smsp__inst_executed_pipe_lsu.sum",
    "ffma": "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "fadd": "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "fmul": "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "integer": "sm__sass_thread_inst_executed_op_integer_pred_on.sum",
}
AVG = {  # weighted by kernel time
    "sm_hz": "sm__cycles_elapsed.avg.per_second",
    "dram_hz": "dram__cycles_elapsed.avg.per_second",
    "l2_hit_pct": "lts__t_sector_hit_rate.pct",
    "issue_active_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "alu_busy_pct": "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active",
    "fmaheavy_busy_pct": "sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_active",
    "dram_sol_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "stall_long_sb_pct": "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "stall_math_pct": "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
    "stall_lg_pct": "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
    "stall_wait_pct": "smsp__warp_issue_stalled_wait_per_warp_active.pct",
}
ISSUE_KERNELS = ["issue_fp32", "issue_int", "issue_imad", "issue_mix"]


def num(s):
    try:
        return float(str(s).replace(",", ""))
    except ValueError:
        return float("nan")


def read_kernels(path):
    """Kernel rows of an ncu raw-page CSV (header, units row, one row per kernel)."""
    lines = [ln for ln in open(path) if ln.startswith('"')]
    rows = list(csv.DictReader(lines))
    return [r for r in rows if r.get("ID", "").isdigit()]


def combine(kernels):
    """Sum counters over the kernels of one generation; time-weight the rest."""
    v = {k: sum(num(r.get(m, "nan")) for r in kernels) for k, m in SUM.items()}
    t = [num(r.get(SUM["time_ns"], "nan")) for r in kernels]
    for k, m in AVG.items():
        v[k] = sum(ti * num(r.get(m, "nan")) for ti, r in zip(t, kernels)) / sum(t)
    v["kernels"] = "+".join(r.get("Kernel Name", "?").split("(")[0].split("<")[0] for r in kernels)
    v["fused_time_ns"] = t[0]
    v["dram_bytes"] = v["dram_read"] + v["dram_write"]
    v["flops"] = 2 * v["ffma"] + v["fadd"] + v["fmul"]
    return v


def ceilings(rawdir, sms):
    ev = {}
    p = os.path.join(rawdir, "ceilings_events.csv")
    if os.path.exists(p):
        ev = {r["kernel"]: r for r in csv.DictReader(open(p))}
    out = {}
    for f in sorted(glob.glob(os.path.join(rawdir, "ceil_*.csv"))):
        name = os.path.basename(f)[5:-4]
        ks = read_kernels(f)
        if not ks:
            continue
        v = combine(ks)
        s = v["time_ns"] * 1e-9
        v.update(kernel=name,
                 events_ms=num(ev.get(name, {}).get("best_ms", "nan")),
                 work=num(ev.get(name, {}).get("work", "nan")),
                 work_unit=ev.get(name, {}).get("work_unit", ""),
                 dram_GBps=v["dram_bytes"] / s / 1e9,
                 l2_GBps=v["l2_bytes"] / s / 1e9,
                 warp_GIPS=v["warp_inst"] / s / 1e9,
                 ipc_per_sm=v["warp_inst"] / (s * v["sm_hz"] * sms))
        v["work_per_s"] = v["work"] / s if v["work"] == v["work"] else float("nan")
        out[name] = v
    return out


def roofs(ceil):
    r = {}
    if "copy_dram" in ceil:
        r["dram_GBps"] = ceil["copy_dram"]["dram_GBps"]
    if "copy_l2" in ceil:
        r["l2_GBps"] = ceil["copy_l2"]["l2_GBps"]
    issue = [ceil[k]["warp_GIPS"] for k in ISSUE_KERNELS if k in ceil]
    if issue:
        r["issue_GIPS"] = max(issue)
    for k in ("binary_rng", "binary_rng_vec4", "philox_draws"):
        if k in ceil:
            r[k + "_GIPS"] = ceil[k]["warp_GIPS"]
            r[k + "_draws_per_s"] = ceil[k]["work_per_s"]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEF_DIR, help="roofline output dir (raw/ inside)")
    ap.add_argument("--configs", default="", help="configs file (tags per config); default <dir>/configs.txt")
    ap.add_argument("--sm", type=int, default=188)
    ap.add_argument("--words", type=int, default=1 << 30, help="--words of the binary_rng ceiling")
    a = ap.parse_args()
    raw = os.path.join(a.dir, "raw")

    ceil = ceilings(raw, a.sm)
    rf = roofs(ceil)
    ckeys = ["kernel", "events_ms", "time_ns", "sm_hz", "dram_hz", "dram_bytes", "l2_bytes", "warp_inst",
             "dram_GBps", "l2_GBps", "warp_GIPS", "ipc_per_sm", "work", "work_unit", "work_per_s",
             "issue_active_pct", "occupancy_pct", "alu_busy_pct", "fmaheavy_busy_pct", "pipe_alu",
             "pipe_fma", "pipe_fmaheavy"]
    with open(os.path.join(a.dir, "roofline_ceilings_v3.csv"), "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(ckeys)
        for v in ceil.values():
            w.writerow([v[k] if isinstance(v[k], str) else f"{v[k]:.6g}" for k in ckeys])
    print("roofs:", {k: round(x, 3) for k, x in rf.items()})

    tags = {}
    cfg = a.configs or os.path.join(a.dir, "configs.txt")
    if os.path.exists(cfg):
        for line in open(cfg):
            f = line.split()
            if len(f) == 9:
                tags[tuple(f[1:])] = f[0]

    pat = re.compile(r"(de-gen|ga-gen|ux-bf)_(highdim|highdim_vec4)_I(\d+)_N(\d+)_D(\d+)_tx(\d+)_ty(\d+)_cap(\d+)\.csv$")
    rows = []
    for f in sorted(glob.glob(os.path.join(raw, "*.csv"))):
        m = pat.search(os.path.basename(f))
        if not m:
            continue
        ks = read_kernels(f)
        if not ks:
            print("no kernels in", f)
            continue
        op, lay, I, N, D, tx, ty, cap = m.groups()
        v = combine(ks)
        s = v["time_ns"] * 1e-9
        I_, N_, D_ = int(I), int(N), int(D)
        words = I_ * N_ * (D_ // 32 if op == "ux-bf" else D_)
        comp = 2 * 4 * words
        v.update(tags=tags.get((op, lay, I, N, D, tx, ty, cap), ""), op=op, layout=lay,
                 family="vec4" if lay.endswith("vec4") else "scalar",
                 I=I_, N=N_, D=D_, tx=int(tx), ty=int(ty), cap=int(cap),
                 warp_GIPS=v["warp_inst"] / s / 1e9, dram_GBps=v["dram_bytes"] / s / 1e9,
                 inst_per_dram_byte=v["warp_inst"] / v["dram_bytes"],
                 inst_per_l2_byte=v["warp_inst"] / v["l2_bytes"],
                 compulsory_bytes=comp)
        v["sol_dram"] = v["dram_GBps"] / rf.get("dram_GBps", float("nan"))
        v["sol_compulsory"] = comp / (rf.get("dram_GBps", float("nan")) * 1e9) / s
        v["sol_issue"] = v["warp_GIPS"] / rf.get("issue_GIPS", float("nan"))
        v["sol_rng"] = float("nan")
        if op == "ux-bf":
            k = "binary_rng_vec4" if v["family"] == "vec4" else "binary_rng"
            if k in ceil:
                v["sol_rng"] = ceil[k]["time_ns"] * 1e-9 * words / a.words / s
        sols = {"dram": v["sol_dram"], "issue": v["sol_issue"], "rng": v["sol_rng"]}
        sols = {k: x for k, x in sols.items() if x == x}
        v["bound"] = max(sols, key=sols.get) if sols else ""
        rows.append(v)

    keys = ["tags", "op", "layout", "family", "I", "N", "D", "tx", "ty", "cap", "kernels",
            "time_ns", "fused_time_ns", "sm_hz", "dram_hz", "dram_read", "dram_write", "dram_bytes",
            "l2_bytes", "l1_bytes", "l2_hit_pct", "warp_inst", "thread_inst", "pipe_alu", "pipe_fma",
            "pipe_fmaheavy", "pipe_xu", "pipe_lsu", "flops", "integer", "issue_active_pct",
            "occupancy_pct", "alu_busy_pct", "fmaheavy_busy_pct", "dram_sol_pct", "stall_long_sb_pct",
            "stall_math_pct", "stall_lg_pct", "stall_wait_pct", "warp_GIPS", "dram_GBps",
            "inst_per_dram_byte", "inst_per_l2_byte", "sol_dram", "sol_issue", "sol_rng", "bound",
            "compulsory_bytes", "sol_compulsory"]
    out = os.path.join(a.dir, "roofline_v3.csv")
    with open(out, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(keys)
        for v in rows:
            w.writerow([v[k] if isinstance(v[k], (str, int)) else f"{v[k]:.6g}" for k in keys])
    print(f"wrote {out} ({len(rows)} configs), roofline_ceilings_v3.csv ({len(ceil)} ceilings)")


if __name__ == "__main__":
    main()
