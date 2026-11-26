#!/usr/bin/env bash
# Download the official CEC 2013 LSGO shift vectors (D = 1000) used by the transfer
# campaign's D = 1000 runs: F1, F2, F3 and F12 xopt files from the suite's code
# (github.com/dmolina/cec2013lsgo, GPL-3.0, so they are not redistributed here).
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)/data/cec2013lsgo"
REV=12ad8573947156458902e5dcde3a73cb10f4f14a
mkdir -p "$DIR"
for f in 1 2 3 12; do
  curl -fsSL -o "$DIR/F$f-xopt.txt" \
    "https://raw.githubusercontent.com/dmolina/cec2013lsgo/$REV/cec2013lsgo/cdatafiles/F$f-xopt.txt"
done
wc -l "$DIR"/F*-xopt.txt
