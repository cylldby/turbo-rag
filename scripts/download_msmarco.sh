#!/usr/bin/env bash
# Download MS MARCO passage subset for large-scale benchmarks.
# Downloads 100k passages by default. Set N_PASSAGES=1000000 for 1M.
set -euo pipefail

N_PASSAGES=${N_PASSAGES:-100000}
DEST="data/msmarco"

if [ -f "$DEST/corpus.jsonl" ]; then
    COUNT=$(wc -l < "$DEST/corpus.jsonl")
    echo "MS MARCO already downloaded: $COUNT passages at $DEST/corpus.jsonl"
    exit 0
fi

echo "Downloading MS MARCO passages (first $N_PASSAGES)..."
mkdir -p "$DEST"

# MS MARCO passages v1 — full collection ~8.8M, we take the first N
URL="https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz"
TMP=$(mktemp -d)
curl -L "$URL" -o "$TMP/collection.tar.gz"
tar -xzf "$TMP/collection.tar.gz" -C "$TMP"

# Convert TSV (pid\ttext) to JSONL {id, text}
head -n "$N_PASSAGES" "$TMP/collection.tsv" | \
    awk -F'\t' '{printf "{\"id\":%s,\"text\":\"%s\",\"metadata\":{\"source\":\"msmarco\"}}\n", $1, $2}' | \
    sed 's/"/\\"/g; s/\\"/"/g' \
    > "$DEST/corpus.jsonl"

rm -rf "$TMP"
COUNT=$(wc -l < "$DEST/corpus.jsonl")
echo "MS MARCO ready: $COUNT passages → $DEST/corpus.jsonl"
