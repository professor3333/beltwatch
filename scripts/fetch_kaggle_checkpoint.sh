#!/usr/bin/env bash
# Download a private Kaggle Dataset of training runs and print the folder of one run.
#
#   KAGGLE_API_TOKEN=... scripts/fetch_kaggle_checkpoint.sh <owner/dataset> <run-name> <dir>
#
# Used by the benchmark and release workflows. The run folder must hold best.pt.
# The weights stay in <dir>; nothing is printed except the folder's path.
set -euo pipefail
dataset=$1 name=$2 dir=$3
test -n "${KAGGLE_API_TOKEN:-}" || { echo "KAGGLE_API_TOKEN is not set" >&2; exit 1; }
mkdir -p "$dir"
# Pinned CLI; its dependencies resolve as of a fixed date.
uvx --exclude-newer 2026-10-01 --from kaggle==2.2.4 \
  kaggle datasets download "$dataset" -p "$dir" --unzip --quiet >&2
find "$dir" -name '*.zip' -exec unzip -q -o {} -d "$dir" \;
run=$(find "$dir" -type d -name "$name" | head -n 1)
test -n "$run" && test -f "$run/best.pt" || { echo "no $name/best.pt in $dataset" >&2; exit 1; }
echo "$run"
