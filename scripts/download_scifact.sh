#!/usr/bin/env bash
# Download BEIR SciFact dataset: 5,183 docs, 300 test queries, relevance judgments.
# Outputs: data/scifact/corpus.jsonl, data/scifact/queries.jsonl, data/scifact/qrels.tsv
set -euo pipefail

DEST="data/scifact"
URL="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"

if [ -f "$DEST/corpus.jsonl" ]; then
    echo "SciFact already downloaded at $DEST/corpus.jsonl"
    exit 0
fi

echo "Downloading SciFact from BEIR..."
mkdir -p "$DEST"
TMP=$(mktemp -d)
curl -L "$URL" -o "$TMP/scifact.zip"
unzip -q "$TMP/scifact.zip" -d "$TMP"
rm -rf "$TMP/scifact.zip"

# BEIR format: corpus.jsonl, queries.jsonl, qrels/test.tsv
cp "$TMP/scifact/corpus.jsonl"        "$DEST/corpus.jsonl"
cp "$TMP/scifact/queries.jsonl"       "$DEST/queries.jsonl"
mkdir -p "$DEST/qrels"
cp "$TMP/scifact/qrels/test.tsv"      "$DEST/qrels/test.tsv"
rm -rf "$TMP"

CORPUS_COUNT=$(wc -l < "$DEST/corpus.jsonl")
QUERY_COUNT=$(wc -l < "$DEST/queries.jsonl")
echo "SciFact ready: $CORPUS_COUNT docs, $QUERY_COUNT queries → $DEST/"
