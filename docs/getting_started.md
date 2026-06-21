# Getting Started with the Knowledge Retrieval Agent

This document is a sample knowledge-base entry used for local demos and the
production ingestion walkthrough.

## What the Agent Does
The knowledge retrieval agent ingests unstructured documents, splits them into
overlapping chunks, embeds those chunks into a vector store, and answers natural
language questions with grounded, citation-backed responses.

## Supported Document Types
The agent supports PDF, plain text, Markdown, and Word documents. Each document is
routed to a format-specific loader and chunked with a configurable size and overlap.

## Asking Questions
Questions are answered using only the content of the ingested documents. If no
sufficiently relevant content is found, the agent declines to answer rather than
guessing, keeping the hallucination rate low.
