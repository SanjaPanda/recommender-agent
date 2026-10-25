"""Apparel ingestion, embeddings, offline enrichment, and retrieval."""

# Serialize catalog imports and embedding writes across maintenance processes.
CATALOG_WRITE_LOCK = 0x524147
