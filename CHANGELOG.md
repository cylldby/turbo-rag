# Changelog

## [0.1.1](https://github.com/cylldby/turbo-rag/compare/v0.1.0...v0.1.1) (2026-06-07)


### Features

* **blob:** InMemory, LocalFs, and S3/MinIO blob backends ([1a0d587](https://github.com/cylldby/turbo-rag/commit/1a0d587c61fca46d6077b56abc85ab8d835601db))
* **cli:** ingest, query, bench, and doctor subcommands with layered config ([aa24261](https://github.com/cylldby/turbo-rag/commit/aa242611b73ff53df01e7a5baacb1ebcbc4f2708))
* **common:** define EmbeddingBackend, BlobBackend, and VectorStore traits ([258382a](https://github.com/cylldby/turbo-rag/commit/258382a986a321db7f83c029ef5e3dab29f76c18))
* **compressor:** TurboQuant index wrapper with blob serialisation ([bb0e6ea](https://github.com/cylldby/turbo-rag/commit/bb0e6eab2840524379e07d2fd8f200da44cfdb97))
* **docker:** MinIO, WireMock, Prometheus, and Grafana compose stack ([af61675](https://github.com/cylldby/turbo-rag/commit/af616752091f06574b5baf2fa78f3207a1a1b887))
* **embedder:** FastEmbed, OpenAI-compat, Synthetic, and Mock backends ([1268eda](https://github.com/cylldby/turbo-rag/commit/1268eda2a9aa788b4f9c72528a76beaa75fbcbe9))
* **pipeline:** rayon preprocessing and buffered-async ingestion pipeline ([921052e](https://github.com/cylldby/turbo-rag/commit/921052eb9af827f1dc069b82231df91e16ab3309))
* **store:** TurboVecStore, LanceDbStore, and HybridStore with BM25 and hybrid RRF ([7df1efc](https://github.com/cylldby/turbo-rag/commit/7df1efc35be3ccaf28ae65b242c3c3cb324194ab))


### Bug Fixes

* **ci:** add libopenblas-dev, fix clippy &PathBuf lint, fix rustfmt ([f5ab09f](https://github.com/cylldby/turbo-rag/commit/f5ab09f3f6803983831638703d42f9fef14e8ddf))
* **ci:** rustfmt all files, fix print_literal and ptr_arg clippy lints ([756a478](https://github.com/cylldby/turbo-rag/commit/756a478cb03e89c7d0df5a36d78bd958d11db2b9))


### Performance Improvements

* **bench:** criterion suites for search, compression, embed, and pipeline throughput ([436ee4e](https://github.com/cylldby/turbo-rag/commit/436ee4ef7f2f981cf00bbf63f4f249c113c06d4b))


### Documentation

* fix badge and clone URLs to cylldby/turbo-rag ([6c3f52a](https://github.com/cylldby/turbo-rag/commit/6c3f52ae65b76a4cf7923d8de4cd4904e26e69bc))
* README, architecture diagrams, ingestion, retrieval, and RAG strategies ([9ccccf7](https://github.com/cylldby/turbo-rag/commit/9ccccf74abcd924aaf970048b0b1dd8cb5195265))
