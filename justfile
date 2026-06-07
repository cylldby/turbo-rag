# turbo-rag task runner — requires `just` (cargo install just)

# ── Dev ────────────────────────────────────────────────────────────────────────
dev:
    docker compose -f docker/compose.yml --profile core up -d --wait

dev-full:
    docker compose -f docker/compose.yml --profile core --profile observability up -d --wait

dev-down:
    docker compose -f docker/compose.yml down

# ── Tests ──────────────────────────────────────────────────────────────────────
test-unit:
    cargo nextest run --workspace

test-unit-verbose:
    cargo nextest run --workspace --no-capture

test-integration:
    docker compose -f docker/compose.yml --profile core --profile test up -d --wait
    cargo nextest run --workspace --features integration; \
    docker compose -f docker/compose.yml down

test-e2e: download-scifact
    docker compose -f docker/compose.yml --profile core --profile test up -d --wait
    cargo nextest run --workspace --features e2e; \
    docker compose -f docker/compose.yml down

# ── Benchmarks ────────────────────────────────────────────────────────────────
bench:
    cargo bench --workspace

bench-quick:
    cargo bench -p bench --bench turbovec_search -- --sample-size 10
    cargo bench -p bench --bench compress_speed  -- --sample-size 10

bench-open:
    open target/criterion/report/index.html

# ── CLI shortcuts ──────────────────────────────────────────────────────────────
doctor:
    cargo run -p cli -- doctor

ingest-sample:
    cargo run -p cli -- ingest --input data/fixtures/corpus.jsonl

ingest-scifact:
    cargo run -p cli -- ingest --input data/scifact/corpus.jsonl

query text="What is machine learning?" top_k="5":
    cargo run -p cli -- query "{{text}}" --top-k {{top_k}}

query-bm25 text="neural network":
    cargo run -p cli -- query "{{text}}" --search-type bm25

query-hybrid text="deep learning transformers":
    cargo run -p cli -- query "{{text}}" --search-type hybrid --mode cold

bench-live:
    cargo run -p cli -- bench --tui --scales 10000,100000

# ── Datasets ──────────────────────────────────────────────────────────────────
download-scifact:
    bash scripts/download_scifact.sh

download-msmarco:
    bash scripts/download_msmarco.sh

# ── Checks ────────────────────────────────────────────────────────────────────
check:
    cargo check --workspace

clippy:
    cargo clippy --workspace -- -D warnings

fmt:
    cargo fmt --all

ci: fmt clippy test-unit
