"""
Medical Literature Search Module

A modular search engine for medical literature providing:
- Semantic vector search using FAISS
- BM25 keyword search for author keywords, MeSH terms, and title/abstract
- Hybrid search combining vector and BM25 approaches
- Database lookup utilities for full article retrieval
"""

from .engine import SearchEngine, get_full_entries

__version__ = "1.0.0"
__all__ = ["SearchEngine", "get_full_entries"]