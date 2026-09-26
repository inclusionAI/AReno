#!/usr/bin/env bash
# Download Open-Jev v1.1 (training set) from ModelScope at pinned revisions,
# verify sha256, and convert it to JevForge records.
#
#   bash examples/classify/jev/data/fetch_open_jev.sh [DEST] [RECORDS]
#
# DEST     parquet directory (default ~/data/open-jev-v1.1, ~59 MB)
# RECORDS  converted records directory (default ~/data/jev-records/open-jev-v1.1, ~720 MB)
#
# Source: ModelScope ZefanCai/Open-Jev-v1.1, config community-hard-mix-v2-redistributable.
# License: original Open-Jev content CC0 1.0; WANLI-derived rows CC BY 4.0
# (Liu et al., 2022). See the dataset's LICENSE-DATA.md and README.md, which this
# script downloads next to the parquet files. The data is not vendored in git
# because of its size.
set -euo pipefail

DEST="${1:-$HOME/data/open-jev-v1.1}"
RECORDS="${2:-$HOME/data/jev-records/open-jev-v1.1}"
REPO="https://www.modelscope.cn/api/v1/datasets/ZefanCai/Open-Jev-v1.1/repo"
PREFIX="data/community-hard-mix-v2-redistributable"
HERE="$(cd "$(dirname "$0")" && pwd)"

# split  revision (per-file ModelScope commit)       sha256
FILES="
train       00fb0860ace9215762888831cf80cfa817ae8671 fa3b32a7052c451754885d734b4d6710c76354aed7999c8edbbf774f169321a3
validation  00fb0860ace9215762888831cf80cfa817ae8671 42fc93f9b5a063ca5f2f61bba10f7c580a9291d9a1403d0e0ae2747b5ee40aa2
calibration 11bacc8904cf279f6e27bbdf50d6304618611075 f101eafd1aea3c30ecfe83c5f94f1bd62fe9091b9c8e85aea794c6509cc00254
test        e2eff18b0e7b8f16da429c63f599d9d25c5f8678 c53beff495ed35d69e30eb584b01b971b663624081718cc10e9f1cca5d6003f6
ood         e2eff18b0e7b8f16da429c63f599d9d25c5f8678 d36ba3b51f3e86154a61e2e1f60511d596a83e012de5744e59b510072aa576ee
"

mkdir -p "$DEST"
while read -r split revision sha; do
  [[ -z "$split" ]] && continue
  file="$DEST/$split.parquet"
  if [[ ! -s "$file" ]] || ! echo "$sha  $file" | sha256sum -c --status; then
    echo "download $split ($revision)"
    curl -sfL --retry 3 -o "$file" "$REPO?Revision=$revision&FilePath=$PREFIX/$split-00000-of-00001.parquet"
  fi
  echo "$sha  $file" | sha256sum -c --quiet || { echo "sha256 mismatch for $file" >&2; exit 1; }
done <<< "$FILES"
for doc in LICENSE-DATA.md README.md; do
  [[ -s "$DEST/$doc" ]] || curl -sfL --retry 3 -o "$DEST/$doc" "$REPO?Revision=master&FilePath=$doc"
done
echo "parquet verified in $DEST"

if [[ ! -s "$RECORDS/manifest.json" ]]; then
  python "$HERE/../convert_datasets.py" open-jev --src "$DEST" --out "$RECORDS"
fi
echo "records in $RECORDS"
