#!/bin/bash
set -euo pipefail

# Usage: collect_timings_to_csv.sh [output.csv] [directory]
# Scans timing files named like timings_*probNN*_*.txt, extracts the values after the first ':' on
# each line and outputs a CSV where the header row contains the probability values (e.g. 0.15)

out=${1:-timings_summary.csv}
dir=${2:-.}

tmpd=$(mktemp -d)
trap 'rm -rf "$tmpd"' EXIT

files=()
probs=()

# find files containing 'timings' and 'prob' in name
while IFS= read -r -d '' f; do
  files+=("$f")
done < <(find "$dir" -maxdepth 1 -type f -name 'timings_*' -print0 | sort -z)

if [ ${#files[@]} -eq 0 ]; then
  echo "No timing files found in $dir" >&2
  exit 1
fi

# Process each file: extract numeric prob from filename and values after ':'
for f in "${files[@]}"; do
  base=$(basename "$f")
  # extract digits after 'prob' (e.g. prob15 -> 15)
  probnum=$(echo "$base" | sed -n 's/.*prob\([0-9]\+\).*/\1/p' || true)
  if [ -z "$probnum" ]; then
    # skip files without prob in name
    continue
  fi
  # format probability as decimal with two digits
  p=$(awk -v n="$probnum" 'BEGIN{printf "%.2f", n/100}')
  probs+=("$p")

  # extract values after ':' from lines that contain a ':' and save to tmp file
  outcol="$tmpd/col_${p}.txt"
  awk -F":" '/:/{
      # take the part after the first colon, trim leading/trailing spaces
      s=$2
      for(i=3;i<=NF;i++) s=s":"$i
      gsub(/^ +| +$/,"",s)
      print s
  }' "$f" > "$outcol"
done

if [ ${#probs[@]} -eq 0 ]; then
  echo "No timing files with 'prob' found." >&2
  exit 1
fi

# sort columns by numeric probability ascending
IFS=$'\n' sorted=($(printf "%s\n" "${probs[@]}" | sort -n))
unset IFS

# build list of tmp files in sorted order
cols=()
for p in "${sorted[@]}"; do
  cols+=("$tmpd/col_${p}.txt")
done

# header row: comma-separated probabilities
(
  IFS=','; echo "${sorted[*]}"
) > "$out"

# paste columns (rows correspond to metric lines), using comma as delimiter
paste -d ',' "${cols[@]}" >> "$out"

echo "Wrote summary to $out"