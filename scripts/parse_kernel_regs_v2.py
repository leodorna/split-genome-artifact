"""Parse `ptxas -v` output for the v2 kernels into results/kernel_regs_v2.csv.

Emits columns: kernel,layout,vec4,regs where
  kernel = operator token (DE1bin | SBXCrossover | UX)
  layout = individual | highdim
  vec4   = 1/0
and regs is the MAX across the FusedVariation and FusedGeneration instantiations
(conservative, so both pipelines fit). Reads ptxas output from a file or stdin;
an optional second argument is the output path.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OPS = ["SBXCrossover", "DE1bin", "UX"]


def demangle(m):
    try:
        return subprocess.run(["c++filt", m], capture_output=True, text=True).stdout.strip()
    except Exception:
        return m


def main():
    text = open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()
    cur = None
    rows = {}
    for line in text.splitlines():
        m = re.search(r"Function properties for (\S+)", line)
        if m:
            cur = m.group(1)
            continue
        m = re.search(r"Used (\d+) registers", line)
        if m and cur:
            d = demangle(cur)
            if ("FusedGeneration" in d or "FusedVariation" in d) and ("Individual" in d or "HighDim" in d):
                layout = "individual" if "Individual" in d else "highdim"
                vec4 = "1" if re.search(r",\s*true>", d) else "0"
                for tok in OPS:
                    if tok in d:
                        k = (tok, layout, vec4)
                        rows[k] = max(rows.get(k, 0), int(m.group(1)))
                        break
            cur = None

    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(RES, "kernel_regs_v2.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write("kernel,layout,vec4,regs\n")
        for (name, layout, vec4), r in sorted(rows.items()):
            f.write(f"{name},{layout},{vec4},{r}\n")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
