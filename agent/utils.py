#!/usr/bin/env python3
"""
utils.py
--------

Utility classes and functions for the queryome agent system.
Contains FileLogger and search tool functions.
"""

import os
import json
from datetime import datetime
from typing import Any, Dict

# Global search engine and database path
search_engine = None
db_path = None


class FileLogger:
    """A simple logger to write to console and structured files."""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.log_file_path = os.path.join(self.base_dir, "run_log.txt")

    def log(self, message: str):
        """Prints to console and appends to the main log file."""
        print(message)
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] {message}\n")

    def log_json(self, file_name: str, data: Any, indent: int = 2):
        """Saves a Python object as a JSON file in the log directory."""
        path = os.path.join(self.base_dir, file_name)
        self.log(f"Saving JSON to {path}")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

    def log_text(self, file_name: str, text: str):
        """Saves a string as a text file in the log directory."""
        path = os.path.join(self.base_dir, file_name)
        self.log(f"Saving text to {path}")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def get_sub_logger(self, sub_dir_name: str):
        """Creates a new logger for a subdirectory."""
        safe_sub_dir_name = "".join(c for c in sub_dir_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
        safe_sub_dir_name = safe_sub_dir_name.replace(' ', '_')[:50]
        sub_dir_path = os.path.join(self.base_dir, safe_sub_dir_name)
        return FileLogger(sub_dir_path)


def hybrid_search(query: str, k: int = 10, alpha: float = 0.5) -> str:
    """Perform hybrid search combining FAISS vector search and BM25 title+abstract search."""
    from search.engine import get_full_entries
    try:
        results = search_engine.hybrid_search(
            query=query,
            k=k,
            alpha=alpha
        )
        if not results:
            return json.dumps({
                "query": query, "search_type": "hybrid",
                "parameters": {"k": k, "alpha": alpha},
                "results_count": 0, "results": [], "message": f"No results found for query: '{query}'"
            }, indent=2, ensure_ascii=False)

        # Fetch full article data
        articles = get_full_entries(db_path, results)
        articles_by_id = {a['vector_id']: a for a in articles}

        formatted_results = []
        for i, result in enumerate(results, 1):
            vector_id = result.get('vector_id') or result.get('chunk_id')
            article = articles_by_id.get(vector_id, {})
            formatted_results.append({
                "rank": i, "score": result['score'], "search_type": "hybrid",
                "vector_id": vector_id, "chunk_id": vector_id,
                "pmid": article.get('pmid'), "title": article.get('title'),
                "year": article.get('year'), "journal": article.get('journal'),
                "authors": article.get('authors', []), "mesh_terms": article.get('mesh_terms', []),
                "keywords": article.get('keywords', []), "citations": article.get('citations', []),
                "text": article.get('text')
            })

        return json.dumps({
            "query": query, "search_type": "hybrid",
            "parameters": {"k": k, "alpha": alpha},
            "results_count": len(results), "results": formatted_results
        }, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "query": query, "search_type": "hybrid", "error": f"ERROR in hybrid_search: {str(e)}",
            "results_count": 0, "results": []
        }, indent=2, ensure_ascii=False)


def bm25_author_keywords_search(query: str, k: int = 10) -> str:
    """Perform BM25 search on author keywords index."""
    from search.engine import get_full_entries
    try:
        results = search_engine.bm25_author_keywords_search(query=query, k=k)
        if not results:
            return json.dumps({
                "query": query, "search_type": "bm25_author_keywords",
                "parameters": {"k": k}, "results_count": 0, "results": [],
                "message": f"No results found for author keywords query: '{query}'"
            }, indent=2, ensure_ascii=False)

        # Fetch full article data
        articles = get_full_entries(db_path, results)
        articles_by_id = {a['vector_id']: a for a in articles}

        formatted_results = []
        for i, result in enumerate(results, 1):
            vector_id = result.get('vector_id') or result.get('chunk_id')
            article = articles_by_id.get(vector_id, {})
            formatted_results.append({
                "rank": i, "score": result['score'], "search_type": "bm25_author_keywords",
                "vector_id": vector_id, "chunk_id": vector_id,
                "pmid": article.get('pmid'), "title": article.get('title'),
                "year": article.get('year'), "journal": article.get('journal'),
                "authors": article.get('authors', []), "mesh_terms": article.get('mesh_terms', []),
                "keywords": article.get('keywords', []), "citations": article.get('citations', []),
                "text": article.get('text')
            })

        return json.dumps({
            "query": query, "search_type": "bm25_author_keywords",
            "parameters": {"k": k}, "results_count": len(results),
            "results": formatted_results
        }, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "query": query, "search_type": "bm25_author_keywords", "error": f"ERROR in bm25_author_keywords_search: {str(e)}",
            "results_count": 0, "results": []
        }, indent=2, ensure_ascii=False)


def bm25_mesh_terms_search(query: str, k: int = 10) -> str:
    """Perform BM25 search on MeSH terms index."""
    from search.engine import get_full_entries
    try:
        results = search_engine.bm25_mesh_terms_search(query=query, k=k)
        if not results:
            return json.dumps({
                "query": query, "search_type": "bm25_mesh_terms",
                "parameters": {"k": k}, "results_count": 0, "results": [],
                "message": f"No results found for MeSH terms query: '{query}'"
            }, indent=2, ensure_ascii=False)

        # Fetch full article data
        articles = get_full_entries(db_path, results)
        articles_by_id = {a['vector_id']: a for a in articles}

        formatted_results = []
        for i, result in enumerate(results, 1):
            vector_id = result.get('vector_id') or result.get('chunk_id')
            article = articles_by_id.get(vector_id, {})
            formatted_results.append({
                "rank": i, "score": result['score'], "search_type": "bm25_mesh_terms",
                "vector_id": vector_id, "chunk_id": vector_id,
                "pmid": article.get('pmid'), "title": article.get('title'),
                "year": article.get('year'), "journal": article.get('journal'),
                "authors": article.get('authors', []), "mesh_terms": article.get('mesh_terms', []),
                "keywords": article.get('keywords', []), "citations": article.get('citations', []),
                "text": article.get('text')
            })

        return json.dumps({
            "query": query, "search_type": "bm25_mesh_terms",
            "parameters": {"k": k}, "results_count": len(results),
            "results": formatted_results
        }, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "query": query, "search_type": "bm25_mesh_terms", "error": f"ERROR in bm25_mesh_terms_search: {str(e)}",
            "results_count": 0, "results": []
        }, indent=2, ensure_ascii=False)


def initialize_search_engine(device: str = "cuda:0",
                             faiss_index_path: str = "indices/vector_db/faiss.index",
                             bm25_author_keywords_path: str = "indices/bm25_author_keywords",
                             bm25_mesh_terms_path: str = "indices/bm25_mesh_terms",
                             bm25_title_abstract_path: str = "indices/bm25_title_abstract",
                             articles_db_path: str = "indices/vector_db/articles.db"):
    """Initialize the medical search engine

    Args:
        device: Device for embedding model (e.g. "cuda:0")
        faiss_index_path: Path to FAISS vector index
        bm25_author_keywords_path: Path to BM25 author keywords index
        bm25_mesh_terms_path: Path to BM25 MeSH terms index
        bm25_title_abstract_path: Path to BM25 title/abstract index
        articles_db_path: Path to SQLite articles database
    """
    global search_engine, db_path
    print("[INIT] Initializing search engine...")
    try:
        from search.engine import SearchEngine
        search_engine = SearchEngine(
            faiss_index_path=faiss_index_path,
            bm25_author_keywords_path=bm25_author_keywords_path,
            bm25_mesh_terms_path=bm25_mesh_terms_path,
            bm25_title_abstract_path=bm25_title_abstract_path,
            device=device,
            use_fp16=True
        )
        db_path = articles_db_path
        print("[INIT] Search engine initialized successfully")
        return True
    except Exception as e:
        import traceback
        print(f"[INIT] Error initializing search engine: {e}")
        traceback.print_exc()
        return False