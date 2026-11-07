# Operations Guide

## Monitor processing

Track the number of submitted files, rejected files, generated chunks, embedding requests, and elapsed time by stage. Keep logs free of source text, secrets, and detected personal information.

## Handle a failed task

1. Read the error category and processing stage.
2. Confirm that the source file is readable and supported.
3. Check connectivity to configured storage and model providers.
4. Retry after correcting the underlying issue.

## Protect indexed data

Back up the database according to the service's retention policy. Test restores in a separate environment. Review document retention and access controls before connecting confidential sources.

## Change an embedding model

Build a complete replacement index in a separate database. Validate its contents and retrieval quality before switching clients. Do not mix vectors from different embedding spaces in one active index.
