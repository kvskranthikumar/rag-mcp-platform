# Getting Started

This sample document demonstrates a small, generic knowledge base for a document-processing service.

## Configure a workspace

Create a workspace and choose a storage location. Store credentials in a secret manager or environment variables rather than in source files.

| Setting | Description |
| --- | --- |
| Workspace name | A short identifier used in logs and dashboards. |
| Storage location | A database or object store accessible to the service. |
| Access token | A secret supplied at runtime; do not commit it. |

## Submit a document

A document submission should include the source file and any metadata needed for discovery, such as a title, language, or revision. Validate the file before processing and report a safe error if it cannot be read.

```text
submit --path ./documents/guide.md --title "Example Guide"
```

## Review the result

Inspect extracted text and chunk boundaries before enabling automated ingestion. Confirm that citations can be traced to the original document and page when available.
