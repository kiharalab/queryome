#!/usr/bin/env python3
"""
Medical Literature Search Engine

A modular search engine that provides semantic vector search, BM25 keyword search,
and hybrid search capabilities for medical literature.
"""

import numpy as np
import bm25s
import Stemmer
import faiss
import sqlite3
import torch
from pathlib import Path
from typing import List, Dict, Any, Union
from sentence_transformers import SentenceTransformer


class SearchEngine:
    """
    Medical Literature Search Engine
    
    Loads all search indices during initialization for fast querying.
    Provides vector search, BM25 search, and hybrid search methods.
    """
    
    def __init__(self, 
                 faiss_index_path: str,
                 bm25_author_keywords_path: str,
                 bm25_mesh_terms_path: str,
                 bm25_title_abstract_path: str,
                 device: str = "cuda:1",
                 use_fp16: bool = True):
        """
        Initialize search engine with all indices
        
        Args:
            faiss_index_path: Path to FAISS vector index
            bm25_author_keywords_path: Path to BM25 author keywords index
            bm25_mesh_terms_path: Path to BM25 MeSH terms index  
            bm25_title_abstract_path: Path to BM25 title/abstract index
            device: Device for embedding model
            use_fp16: Whether to use half precision for embeddings
        """
        self.device = device
        self.use_fp16 = use_fp16
        
        # Load embedding model
        print("Loading embedding model...")
        target_device = torch.device(device)
        self.embedding_model = SentenceTransformer(
            "Linq-AI-Research/Linq-Embed-Mistral",
            device="cpu"
        )
        self.embedding_model.half()
        self.embedding_model = self.embedding_model.to(target_device)
        self.embedding_model._target_device = target_device  # ensure encode() schedules on GPU
        
        # Load FAISS index
        print("Loading FAISS index...")
        self.faiss_index = faiss.read_index(faiss_index_path)
        
        # Load BM25 indices
        print("Loading BM25 indices...")
        self.bm25_author_keywords = bm25s.BM25.load(bm25_author_keywords_path, mmap=True, load_corpus=True)
        self.bm25_author_keywords.backend = "numba"
        
        self.bm25_mesh_terms = bm25s.BM25.load(bm25_mesh_terms_path, mmap=True, load_corpus=True) 
        self.bm25_mesh_terms.backend = "numba"
        
        self.bm25_title_abstract = bm25s.BM25.load(bm25_title_abstract_path, mmap=True, load_corpus=True)
        self.bm25_title_abstract.backend = "numba"
        
        # Initialize stemmer for BM25 queries
        self.stemmer = Stemmer.Stemmer("english")
        
        print("Search engine initialized successfully!")
    
    def _embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query using the loaded embedding model
        
        Args:
            query: Query string
            
        Returns:
            1D numpy array embedding
        """
        task_instruction = "Given a prompt, retrieve medical research passages that relate most closely to the prompt."
        prompt = f"Instruct: {task_instruction}\nPrompt: "
        
        embedding = self.embedding_model.encode([query], prompt=prompt)
        return embedding[0].astype(np.float32)
    
    def vector_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of results with vector IDs and scores
        """
        # Embed query
        query_vector = self._embed_query(query)
        query_vector = query_vector.reshape(1, -1)
        faiss.normalize_L2(query_vector)
        
        # Search
        scores, indices = self.faiss_index.search(query_vector, k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append({
                    'vector_id': int(idx),
                    'score': float(score)
                })
        
        return results
    
    def bm25_author_keywords_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search author keywords using BM25
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of results with chunk IDs and scores
        """
        # Tokenize query
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer, stopwords="en")
        
        # Search
        results = self.bm25_author_keywords.retrieve(
            query_tokens,
            k=k,
            return_as="tuple",
            show_progress=False
        )
        
        # Format results
        search_results = []
        for i in range(results.documents.shape[1]):
            doc = results.documents[0, i]
            score = float(results.scores[0, i])
            search_results.append({
                'chunk_id': doc['chunk_id'],
                'score': score
            })
        
        return search_results
    
    def bm25_mesh_terms_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search MeSH terms using BM25
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of results with chunk IDs and scores
        """
        # Tokenize query
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer, stopwords="en")
        
        # Search
        results = self.bm25_mesh_terms.retrieve(
            query_tokens,
            k=k,
            return_as="tuple",
            show_progress=False
        )
        
        # Format results
        search_results = []
        for i in range(results.documents.shape[1]):
            doc = results.documents[0, i]
            score = float(results.scores[0, i])
            search_results.append({
                'chunk_id': doc['chunk_id'],
                'score': score
            })
        
        return search_results
    
    def bm25_title_abstract_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search title and abstract using BM25
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of results with chunk IDs and scores
        """
        # Tokenize query
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer, stopwords="en")
        
        # Search
        results = self.bm25_title_abstract.retrieve(
            query_tokens,
            k=k,
            return_as="tuple",
            show_progress=False
        )
        
        # Format results
        search_results = []
        for i in range(results.documents.shape[1]):
            doc = results.documents[0, i]
            score = float(results.scores[0, i])
            search_results.append({
                'chunk_id': doc['chunk_id'],
                'score': score
            })
        
        return search_results
    
    def hybrid_search(self, 
                     query: str,
                     k: int = 10,
                     alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity and BM25 search
        
        Args:
            query: Search query string
            k: Number of final results to return
            alpha: Weight for semantic search (0.0 = only BM25, 1.0 = only vector)
            
        Returns:
            List of hybrid results with chunk IDs and combined scores
        """
        print(f"[ENGINE] Hybrid search started for query='{query}' k={k} alpha={alpha}")

        retrieval_k = k * 100
        print(f"[ENGINE] Retrieval depth set to {retrieval_k}")
        
        # Vector search results
        print("[ENGINE] Running vector search...")
        vector_results = self.vector_search(query, k=retrieval_k)
        print(f"[ENGINE] Vector search returned {len(vector_results)} results")
        
        # BM25 search results (using title/abstract as default)
        print("[ENGINE] Running BM25 (title/abstract) search...")
        bm25_results = self.bm25_title_abstract_search(query, k=retrieval_k)
        print(f"[ENGINE] BM25 search returned {len(bm25_results)} results")
        
        # Convert vector results to use chunk_id instead of vector_id for consistency
        vector_dict = {}
        for result in vector_results:
            chunk_id = result['vector_id']  # Since vector_id = chunk_id
            vector_dict[chunk_id] = result['score']
        print(f"[ENGINE] Vector dictionary populated with {len(vector_dict)} entries")
        
        # Convert BM25 results to dictionary
        bm25_dict = {}
        for result in bm25_results:
            chunk_id = result['chunk_id']
            bm25_dict[chunk_id] = result['score']
        print(f"[ENGINE] BM25 dictionary populated with {len(bm25_dict)} entries")
        
        # Get all unique chunk IDs from both searches
        all_chunk_ids = set(vector_dict.keys()) | set(bm25_dict.keys())
        print(f"[ENGINE] Unique chunk IDs combined: {len(all_chunk_ids)}")
        
        # Normalize scores
        # Vector scores are already in [-1, 1] range, normalize to [0, 1]
        vector_scores = list(vector_dict.values())
        if vector_scores:
            vector_min, vector_max = min(vector_scores), max(vector_scores)
            vector_range = vector_max - vector_min if vector_max != vector_min else 1.0
        else:
            vector_min, vector_range = 0.0, 1.0
        print(f"[ENGINE] Vector score range: min={vector_min} range={vector_range}")
        
        # BM25 scores are positive, normalize to [0, 1]
        bm25_scores = list(bm25_dict.values())
        if bm25_scores:
            bm25_min, bm25_max = min(bm25_scores), max(bm25_scores)
            bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1.0
        else:
            bm25_min, bm25_range = 0.0, 1.0
        print(f"[ENGINE] BM25 score range: min={bm25_min} range={bm25_range}")
        
        # Calculate hybrid scores
        hybrid_results = []
        for chunk_id in all_chunk_ids:
            # Normalize vector score to [0, 1]
            vector_score = vector_dict.get(chunk_id, vector_min)
            normalized_vector = (vector_score - vector_min) / vector_range
            
            # Normalize BM25 score to [0, 1]  
            bm25_score = bm25_dict.get(chunk_id, bm25_min)
            normalized_bm25 = (bm25_score - bm25_min) / bm25_range
            
            # Calculate hybrid score
            hybrid_score = alpha * normalized_vector + (1 - alpha) * normalized_bm25
            
            hybrid_results.append({
                'chunk_id': chunk_id,
                'score': hybrid_score
            })
        print(f"[ENGINE] Hybrid scores computed for {len(hybrid_results)} entries")
        
        # Sort by hybrid score (descending) and return top k
        hybrid_results.sort(key=lambda x: x['score'], reverse=True)
        final_results = hybrid_results[:k]
        print(f"[ENGINE] Returning top {len(final_results)} hybrid results")
        return final_results


def get_full_entries(db_path: str,
                    ids: Union[List[int], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Lookup full article entries from SQLite database using vector_id

    Args:
        db_path: Path to SQLite database
        ids: List of IDs (integers) or search results (dicts with chunk_id/vector_id)

    Returns:
        List of full article entries from database
    """
    # Extract IDs from input
    if isinstance(ids[0], dict):
        # Input is search results - extract IDs (vector_id == chunk_id)
        id_list = [result.get('vector_id') or result.get('chunk_id') for result in ids]
    else:
        # Input is list of integers
        id_list = ids

    # Connect to database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enable dict-like access
    cursor = conn.cursor()

    # Query using vector_id
    placeholders = ','.join('?' * len(id_list))
    query = f"SELECT * FROM articles WHERE vector_id IN ({placeholders})"

    # Execute query
    cursor.execute(query, id_list)
    rows = cursor.fetchall()

    # Convert to list of dictionaries
    results = []
    for row in rows:
        entry = dict(row)
        # Parse JSON fields
        import json
        for json_field in ['authors', 'mesh_terms', 'keywords', 'citations']:
            if entry.get(json_field):
                try:
                    entry[json_field] = json.loads(entry[json_field])
                except json.JSONDecodeError:
                    entry[json_field] = []
        results.append(entry)

    conn.close()
    return results
