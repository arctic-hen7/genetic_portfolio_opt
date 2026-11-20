#!/bin/bash

set -euo pipefail

# It must print a line like "Average annual return: 24.969%" as its last line.
CMD="cargo run --release"

NUM=10
OUTFILE="asset_data/results.txt"

# start fresh
: > "$OUTFILE"

sum=0

for i in $(seq 1 $NUM); do
  echo "=== Run $i ===" >> "$OUTFILE"
  # capture entire output
  fullout=$($CMD)
  echo "$fullout" >> "$OUTFILE"
  # extract the last line
  last=$(printf "%s\n" "$fullout" | tail -n1)
  # pull out the number before the '%' (handles floats)
  ret=$(printf "%s\n" "$last" | sed -E 's/.*: *([0-9]+(\.[0-9]+)?)%.*/\1/')
  # add to running total
  sum=$(echo "$sum + $ret" | bc -l)
  echo "---" >> "$OUTFILE"
done

# compute arithmetic mean
avg=$(echo "scale=6; $sum / $NUM" | bc -l)

# report it (both to stdout and append to results.txt)
echo "Overall average annual return over $NUM runs: ${avg}%" | tee -a "$OUTFILE"
