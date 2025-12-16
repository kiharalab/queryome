#!/usr/bin/env python3
"""
queryome.py
-----------

Python interface for the Queryome medical research system.

This module provides a clean Python API for integrating Queryome into other applications.

Example usage:
    from queryome import Queryome, batch_research
    
    # Single query - Initialize the research system
    queryome = Queryome()
    result = queryome.research("What are the latest treatments for Type 2 diabetes?")
    print(result)
    
    # Multiple queries - Reuses loaded indices and models
    queries = [
        "Efficacy of metformin in elderly patients",
        "Side effects of insulin therapy",
        "Latest diabetes management guidelines"
    ]
    results = queryome.research_multiple(queries)
    for r in results:
        print(f"Query: {r['query']}")
        print(f"Result: {r['result']}")
        print(f"Log directory: {r['log_directory']}")
        print("---")
    
    # Specify individual log directories for each query
    custom_log_dirs = ["./metformin_logs", "./insulin_logs", "./guidelines_logs"]
    results = queryome.research_multiple(queries, log_dirs=custom_log_dirs)
    
    # Convenience function for batch processing
    results = batch_research(queries, log_dirs=custom_log_dirs)
"""

import os
import tempfile
from datetime import datetime
from typing import Optional, Dict, Any, List
from agent import PIAgent, FileLogger, initialize_search_engine


class Queryome:
    """
    Main interface for the Queryome medical research system.
    
    This class provides a simple Python API for conducting medical literature research
    using the multi-agent system.
    """
    
    def __init__(self, 
                 log_dir: Optional[str] = None,
                 enable_search_engine: bool = True,
                 openai_api_key: Optional[str] = None,
                 embedding_device: str = "cuda:0"):
        """m
        Initialize the Queryome research system.
        
        Args:
            log_dir: Directory for storing logs. If None, uses a temporary directory.
            enable_search_engine: Whether to initialize the search engine.
            openai_api_key: OpenAI API key. If None, uses OPENAI_API_KEY environment variable.
        """
        # Set up API key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key must be provided either as parameter or OPENAI_API_KEY environment variable")
        
        # Set up logging directory
        if log_dir:
            self.log_dir = log_dir
        else:
            self.log_dir = os.path.join(tempfile.gettempdir(), "queryome", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        
        self.logger = FileLogger(self.log_dir)
        self.logger.log(f"Queryome initialized - Log directory: {os.path.abspath(self.log_dir)}")
        
        # Initialize search engine if requested
        self.search_engine_enabled = False
        if enable_search_engine:
            self.search_engine_enabled = initialize_search_engine(device=embedding_device)
            if not self.search_engine_enabled:
                self.logger.log("WARNING: Failed to initialize search engine")
        
        # Initialize PI Agent
        self.pi_agent = PIAgent(logger=self.logger)
        self.logger.log("Queryome initialization complete")
    
    def research(self, query: str) -> str:
        """
        Conduct research on a medical query.
        
        Args:
            query: The research question to investigate
            
        Returns:
            Research results as a formatted string
            
        Raises:
            ValueError: If query is empty
            Exception: If research fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        query = query.strip()
        self.logger.log(f"[QUERYOME] Starting research for: '{query}'")
        
        try:
            result = self.pi_agent.research(query)
            self.logger.log(f"[QUERYOME] Research completed successfully")
            return result
        except Exception as e:
            self.logger.log(f"[QUERYOME] Research failed: {str(e)}")
            raise Exception(f"Research failed: {str(e)}") from e
    
    def research_multiple(self, queries: List[str], log_dirs: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Conduct research on multiple medical queries using the same loaded indices and models.
        
        Args:
            queries: List of research questions to investigate
            log_dirs: Optional list of log directories for each query. If None, uses subdirectories
                     of the main log directory. If provided, must match length of queries.
            
        Returns:
            List of dictionaries containing query, result, and metadata for each query
            
        Raises:
            ValueError: If queries list is empty, contains empty queries, or log_dirs length mismatch
            Exception: If research fails
        """
        if not queries:
            raise ValueError("Queries list cannot be empty")
        
        if log_dirs is not None and len(log_dirs) != len(queries):
            raise ValueError(f"log_dirs length ({len(log_dirs)}) must match queries length ({len(queries)})")
        
        results = []
        for i, query in enumerate(queries):
            if not query or not query.strip():
                self.logger.log(f"[QUERYOME] Skipping empty query at index {i}")
                results.append({
                    "query": query,
                    "result": None,
                    "error": "Query cannot be empty",
                    "query_index": i
                })
                continue
            
            query = query.strip()
            self.logger.log(f"[QUERYOME] Processing query {i+1}/{len(queries)}: '{query}'")
            
            # Create query-specific logger if custom log directory provided
            if log_dirs is not None:
                query_log_dir = log_dirs[i]
                query_logger = FileLogger(query_log_dir)
                query_logger.log(f"[QUERYOME] Starting research for query {i+1}: '{query}'")
                # Create a new PI agent with the query-specific logger
                from agent import PIAgent
                query_pi_agent = PIAgent(logger=query_logger)
            else:
                # Use subdirectory of main log directory
                safe_query_name = "".join(c for c in query if c.isalnum() or c in (' ', '_', '-')).rstrip()
                safe_query_name = safe_query_name.replace(' ', '_')[:50]
                query_logger = self.logger.get_sub_logger(f"query_{i+1}_{safe_query_name}")
                query_pi_agent = PIAgent(logger=query_logger)
            
            try:
                result = query_pi_agent.research(query)
                self.logger.log(f"[QUERYOME] Query {i+1} completed successfully")
                query_logger.log(f"[QUERYOME] Query completed successfully")
                results.append({
                    "query": query,
                    "result": result,
                    "error": None,
                    "query_index": i,
                    "log_directory": query_logger.base_dir
                })
            except Exception as e:
                error_msg = f"Research failed: {str(e)}"
                self.logger.log(f"[QUERYOME] Query {i+1} failed: {error_msg}")
                query_logger.log(f"[QUERYOME] Query failed: {error_msg}")
                results.append({
                    "query": query,
                    "result": None,
                    "error": error_msg,
                    "query_index": i,
                    "log_directory": query_logger.base_dir
                })
        
        self.logger.log(f"[QUERYOME] Completed processing {len(queries)} queries")
        return results
    
    def get_log_directory(self) -> str:
        """
        Get the current log directory path.
        
        Returns:
            Absolute path to the log directory
        """
        return os.path.abspath(self.log_dir)
    
    def get_research_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current research session.
        
        Returns:
            Dictionary containing research statistics
        """
        return {
            "log_directory": self.get_log_directory(),
            "search_engine_enabled": self.search_engine_enabled,
            "total_relevant_articles": len(self.pi_agent.all_relevant_information),
            "last_query": self.pi_agent.user_query if self.pi_agent.user_query else None
        }
    
    def close(self):
        """Clean up resources."""
        from agent.utils import search_engine
        if search_engine and hasattr(search_engine, 'close'):
            search_engine.close()
            self.logger.log("Search engine closed")
        self.logger.log("Queryome session closed")


class QueryomeSession:
    """
    Context manager for Queryome sessions.
    
    Example:
        with QueryomeSession() as queryome:
            result = queryome.research("What is the efficacy of aspirin?")
            print(result)
    """
    
    def __init__(self, **kwargs):
        """Initialize with same parameters as Queryome."""
        self.kwargs = kwargs
        self.queryome = None
    
    def __enter__(self):
        """Enter the context manager."""
        self.queryome = Queryome(**self.kwargs)
        return self.queryome
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        if self.queryome:
            self.queryome.close()


def quick_research(query: str, 
                  log_dir: Optional[str] = None,
                  enable_search_engine: bool = True) -> str:
    """
    Convenience function for quick research queries.
    
    Args:
        query: The research question
        log_dir: Directory for logs
        enable_search_engine: Whether to use search engine
        
    Returns:
        Research results as a string
    """
    with QueryomeSession(log_dir=log_dir, enable_search_engine=enable_search_engine) as queryome:
        return queryome.research(query)


def batch_research(queries: List[str],
                  log_dir: Optional[str] = None,
                  log_dirs: Optional[List[str]] = None,
                  enable_search_engine: bool = True) -> List[Dict[str, Any]]:
    """
    Convenience function for batch research queries.
    Reuses the same loaded indices and models for all queries.
    
    Args:
        queries: List of research questions
        log_dir: Main directory for logs (used when log_dirs is None)
        log_dirs: Optional list of individual log directories for each query
        enable_search_engine: Whether to use search engine
        
    Returns:
        List of dictionaries with query results and metadata
    """
    with QueryomeSession(log_dir=log_dir, enable_search_engine=enable_search_engine) as queryome:
        return queryome.research_multiple(queries, log_dirs=log_dirs)


# Example usage when run as script
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python queryome.py 'Your research question here'")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    
    try:
        print(f"Researching: {query}")
        result = quick_research(query)
        
        print("\n" + "="*80)
        print("RESEARCH RESULTS")
        print("="*80)
        print(result)
        
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)