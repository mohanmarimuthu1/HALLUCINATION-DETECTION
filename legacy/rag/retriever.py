"""
Retriever Module
Handles retrieving relevant documents from the knowledge base
"""
from typing import List, Dict, Any
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from knowledge_base.vector_store import get_vector_store


class Retriever:
    """
    Retrieves relevant documents from the vector store based on a query.
    """
    
    def __init__(self, top_k: int = None):
        """
        Initialize the retriever.
        
        Args:
            top_k: Number of documents to retrieve (default from config)
        """
        self.top_k = top_k or config.TOP_K_DOCUMENTS
        self.vector_store = get_vector_store()
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The search query
            top_k: Number of documents to retrieve (optional override)
            
        Returns:
            List of retrieved documents with metadata
        """
        k = top_k or self.top_k
        results = self.vector_store.search(query, top_k=k)
        return results
    
    def retrieve_as_context(self, query: str, top_k: int = None) -> str:
        """
        Retrieve documents and format them as a context string.
        
        Args:
            query: The search query
            top_k: Number of documents to retrieve (optional override)
            
        Returns:
            Formatted context string
        """
        results = self.retrieve(query, top_k)
        
        if not results:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Source {i}] (Relevance: {result['similarity']:.2f}):\n{result['document']}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def get_retrieval_info(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """
        Get detailed retrieval information including sources.
        
        Args:
            query: The search query
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with context and source information
        """
        results = self.retrieve(query, top_k)
        
        return {
            "query": query,
            "num_results": len(results),
            "results": results,
            "context": self.retrieve_as_context(query, top_k),
            "sources": [r.get("metadata", {}).get("source", "Unknown") for r in results]
        }


# Singleton instance
_retriever = None


def get_retriever() -> Retriever:
    """
    Get or create the retriever instance.
    
    Returns:
        Retriever instance
    """
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def main():
    """Test the retriever"""
    retriever = get_retriever()
    
    # Test queries
    test_queries = [
        "What is hallucination in LLMs?",
        "How does RAG work?",
        "What are word embeddings?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        info = retriever.get_retrieval_info(query)
        print(f"Found {info['num_results']} relevant documents")
        print(f"\nContext:\n{info['context'][:500]}...")


if __name__ == "__main__":
    main()
