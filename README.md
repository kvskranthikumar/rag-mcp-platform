# RAG MCP Platform

A configurable retrieval-augmented generation service that indexes user-supplied documents in PostgreSQL with `pgvector` and exposes retrieval through MCP, a CLI, and an optional web interface.

## Capabilities

- Ingest Markdown, text, HTML, DOCX, and PDF files.
- Extract structure, split into bounded chunks, and create embeddings.
- Retrieve with PostgreSQL vector search and full-text search fused by Reciprocal Rank Fusion.
- Optionally rerank results and generate answers through a configurable LLM provider.
- Connect an MCP client using stdio or streamable HTTP.
- Inspect extraction and chunking before indexing; audit an existing index; evaluate extraction using a JSON manifest.

The project contains only synthetic generic sample documents under `examples/docs/`. The empty `docs-source/` directory is ignored so local document corpora are not accidentally committed. Add your files there or supply another path with `--path`.

## Architecture

```text
Documents -> loaders -> structure-aware chunker -> embedding provider -> PostgreSQL/pgvector
                                                                        |
Question -> query embedding -> vector + full-text search -> RRF -> cited passages
                                                                        |
                                      +---------------------------------+----------------+
                                      | MCP retrieval tool                               |
                                      | Optional web app: retrieve + answer LLM         |
                                      +--------------------------------------------------+
```

## Requirements

- Python 3.10 or later.
- PostgreSQL 14 or later with the `vector` extension available.
- An embedding provider and, if using generated answers, a supported answer-LLM provider.

## Install and Configure

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[web,ingest-ui,mcp-http]"
Copy-Item .env.example .env
```

Set `DATABASE_URL` and provider settings in `.env`. For Azure OpenAI, configure `EMBEDDING_PROVIDER=azure`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and `AZURE_EMBEDDING_DEPLOYMENT`. Hosted providers are blocked by default. Set `ALLOW_EXTERNAL_DATA=true` only after approving the transfer: embedding providers receive document chunks and search queries, while answer providers receive the question and retrieved passages. PII replacement is not a confidentiality control. For local inference, use `EMBEDDING_PROVIDER=local` with a compatible local model and `LLM_PROVIDER=ollama` at its default loopback URL. Set `LLM_PROVIDER` and the matching LLM settings only when using the web app's answer endpoint or the CLI `ask` command. MCP search returns passages; the MCP host's model composes the response.

Langfuse tracing is optional. Install it with `python -m pip install -e ".[observability]"`, set `LANGFUSE_ENABLED=true`, and configure `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL`. Ingestion, retrieval, embedding, reranking, and answer-generation spans are correlated and flushed at process exit. Traces include stage names, counts, durations, model identifiers, token usage, and safe error classes; document text, queries, prompts, model outputs, paths, and exception messages are not sent.

Create the schema, then inspect and ingest your documents:

```powershell
rag-mcp-platform init-db
rag-mcp-platform inspect --path .\examples\docs
rag-mcp-platform ingest --path .\examples\docs
rag-mcp-platform status
```

`inspect` is read-only and does not contact the database or embedding provider. Add `--preview` to include extracted chunks in its JSON output. To use the module form instead of the console script:

```powershell
python -m rag_mcp_platform inspect --path .\examples\docs --preview
```

The installed distribution and console commands are named `rag-mcp-platform`; Python imports use `rag_mcp_platform`.

Other CLI commands include `search`, `ask`, and `evaluate`. Run `rag-mcp-platform --help` for the complete list. A starter evaluation manifest is available at `examples/evaluation.json`:

```powershell
rag-mcp-platform evaluate .\examples\evaluation.json
```

## Configuration

Settings are loaded from OS environment variables and, for local development, the project `.env` file. Containers set `RAG_MCP_PLATFORM_RUNTIME=container` and use injected OS environment only. The distribution and console commands are named `rag-mcp-platform`; the importable Python module is `rag_mcp_platform`.

| Setting | Purpose |
| --- | --- |
| `DATABASE_URL` | PostgreSQL connection. |
| `DOCS_SOURCE_DIR` | Default source directory; defaults to `docs-source`. |
| `EMBEDDING_PROVIDER` | `azure`, `openai`, or `local`. |
| `EMBEDDING_MODEL`, `EMBEDDING_DIM` | Embedding model identity and vector dimension. Default dimension is 1536. |
| `EMBEDDING_BATCH_SIZE` | Maximum hosted inputs per request. |
| `EMBEDDING_MAX_INPUT_TOKENS`, `EMBEDDING_BATCH_MAX_TOKENS` | Per-input and per-request token limits. |
| `EMBEDDING_MAX_RETRIES`, `EMBEDDING_TIMEOUT_SECONDS` | Hosted SDK retry and timeout bounds. |
| `ALLOW_EXTERNAL_DATA` | Explicitly permit hosted inference or a non-loopback Ollama endpoint to receive document/query content; defaults to `false`. |
| `EMBEDDING_REUSE_ENABLED` | Reuse exact matching chunk embeddings from the active index; defaults to false. |
| `CHUNK_MAX_TOKENS`, `CHUNK_OVERLAP_TOKENS` | Chunk bounds and overlap. |
| `INGESTION_PIPELINE_VERSION` | Processing version used to detect stale chunks; defaults to `10`. |
| `LLM_PROVIDER`, `LLM_MODEL` | Optional answer-generation provider and model. |
| `LANGFUSE_ENABLED` | Enable optional Langfuse tracing; defaults to `false`. |
| `MCP_API_KEY` | Bearer token required for MCP HTTP on non-loopback hosts. |

See `.env.example` for provider-specific variables. Model or embedding-version changes require a complete isolated index rebuild; `--full` cannot bypass the embedding-space compatibility guard.

## Interfaces

### MCP

Local stdio:

```powershell
rag-mcp-platform-mcp
```

Hosted streamable HTTP:

```powershell
rag-mcp-platform-mcp-http
```

The MCP endpoint is `/mcp`; `GET /health` is available for container health checks. The server exposes `search_documentation` and `docs://documents`.

### Web app

```powershell
rag-mcp-platform-web
```

The FastAPI application serves a small static chat UI and `/api/ask`, `/api/search`, `/api/documents`, and `/api/health` endpoints.

### Ingestion UI

```powershell
rag-mcp-platform-ingest-ui
```

The Streamlit upload UI stages files in a temporary directory, indexes them, then removes the staged originals. Re-submit a file to retry it; there is no upload archive or background-job resumption.

## Containers

Build from the repository root:

```powershell
docker build -f docker/mcp/Dockerfile -t rag-mcp-platform-mcp .
docker build -f docker/ingest-ui/Dockerfile -t rag-mcp-platform-ingest-ui .
```

Containers connect to an external PostgreSQL instance and embedding service. Inject secrets at runtime; do not bake credentials or `.env` into images. Generic deployment examples are in [docker/README.md](docker/README.md).

## Ingestion Quality and Data Handling

- HTML headings, tables, code, and allowlisted metadata are retained. DOCX paragraphs and tables preserve order. PDF text and detected tables retain physical page provenance.
- Oversized prose is split within token bounds; tables, code, and lists split at structural boundaries where possible.
- The default PII policy replaces detected English identifiers with entity placeholders before chunking, embedding, storage, or preview, then continues processing. It remains rule-based and non-comprehensive. Screening or anonymization failures stop processing; previously indexed content is not automatically purged.
- Hosted inference is disabled unless `ALLOW_EXTERNAL_DATA=true`. Even after PII replacement, confidential business text may remain and is sent to the configured endpoint when that opt-in is enabled. PostgreSQL still stores document content; use an approved database endpoint and restrict network egress at the host or deployment boundary for a strict no-egress requirement.
- OCR remains manual. Prepare text-enabled PDFs before ingestion.
- Audience metadata is descriptive only; document- or tenant-level authorization is not implemented.
- Original uploads are not retained after staging cleanup.

### Presidio PII Screening

PII detection uses Microsoft Presidio's `presidio-analyzer` package with an
explicit set of recognizers, called directly rather than through the default
`AnalyzerEngine`. Built-in email, phone, card, and US SSN checks are combined with
custom English label-based name/address patterns and conservative email-shaped
source/SSN patterns. Overlapping findings in a category are counted once.
Presidio's built-in card recognizer validates checksums. Phone results use a 0.4
score cutoff; other selected recognizers use 0.5.

This configuration loads no NLP model, downloads no model weights, and makes no
PII service calls. Presidio may install NLP-related Python dependencies, but their
model-backed engines are not used. `presidio-anonymizer` replaces detected spans
with placeholders such as `<PERSON>`, `<EMAIL_ADDRESS>`, `<PHONE_NUMBER>`,
`<POSTAL_ADDRESS>`, `<US_SSN>`, and `<CREDIT_CARD>`.

Ingestion and inspection use the **replace-and-continue** policy. Useful text and
field labels are retained where possible; only the sanitized document is hashed,
chunked, embedded, stored, or previewed. Metadata is sanitized recursively,
including keys, values, and lists. PDF text and transient raw page text are
sanitized independently while retaining physical page numbers. Existing
source files are not modified, and no original-upload archive is introduced.

Sensitive source identities become stable `sanitized://` hash references instead
of storing the original path. Sensitive URLs are removed rather than creating
invalid links containing placeholders. File-result labels are sanitized, and
`pii_sanitization` metadata records replacement categories/counts without values.
Counts represent replacements across document representations, not unique people.

Sanitization runs before skip/deduplication checks and embeddings, including
inspection previews and explicit `--full` ingestion. Detector/anonymizer failures
or incomplete sanitization stop processing without publishing a partial record.
Audit and explicit maintenance still use detection-only checks; auditing does not
rewrite existing data.

Existing database content is not automatically purged. Reprocess existing sources
to apply the policy. Legacy records whose sensitive identities change to hashed
references require an audit and approved identity migration/cleanup; this step
does not delete them automatically.

Unlabelled names, unusual address formats, images, and unsupported identifiers
can be missed; rules are not a compliance guarantee. No prompt-injection or
content-moderation model is added by this change.

Update dependencies with `python -m pip install -e .`, restart ingestion processes,
and update any explicit pipeline-version override to `10` before reprocessing.
The embedding-space signature is unchanged and no schema migration is required.

## Auditing, Maintenance, and Evaluation

```powershell
rag-mcp-platform audit
rag-mcp-platform audit --pii
rag-mcp-platform reconcile --path C:\path\to\documents
rag-mcp-platform rename DOCUMENT_UUID --expected-version 3 --to new/source/path.md
rag-mcp-platform delete DOCUMENT_UUID --expected-version 3
rag-mcp-platform evaluate evaluation.json
```

Audit and reconcile are read-only. Rename and delete are previews unless `--apply` is explicitly supplied; both require the current document version. These PostgreSQL commands have not been verified against your live database.

The offline `evaluate` command checks extraction, metadata, chunk hierarchy, and size against expected results. It does not evaluate retrieval quality. Supply representative documents and expected outcomes for a useful evaluation set.

## Development

```powershell
python -m unittest discover -s tests
```

The tests use temporary files and simulated database/provider clients. They do not require a live database or make hosted embedding calls. This project is distributed under the MIT License; see [LICENSE](LICENSE).