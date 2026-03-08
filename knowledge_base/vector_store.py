"""
Vector Store Module
Manages ChromaDB for storing and retrieving document embeddings
"""
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from knowledge_base.embeddings import get_embedding_model


class VectorStore:
    """
    Manages the ChromaDB vector database for document storage and retrieval.
    """
    
    def __init__(self, persist_directory: str = None, collection_name: str = None):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the database (default from config)
            collection_name: Name of the collection (default from config)
        """
        self.persist_directory = persist_directory or config.CHROMA_PERSIST_DIRECTORY
        self.collection_name = collection_name or config.COLLECTION_NAME
        
        # Create persist directory if it doesn't exist
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        # Initialize embedding model
        self.embedding_model = get_embedding_model()
        
        print(f"Vector store initialized. Collection: {self.collection_name}")
        print(f"Current document count: {self.collection.count()}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
        """
        if not documents:
            print("No documents to add")
            return
        
        # Prepare data for ChromaDB
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [f"doc_{i}_{hash(doc.page_content) % 10000}" for i, doc in enumerate(documents)]
        
        # Create embeddings
        print(f"Creating embeddings for {len(texts)} documents...")
        embeddings = self.embedding_model.embed_texts(texts)
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Added {len(documents)} documents to vector store")
        print(f"Total documents: {self.collection.count()}")
    
    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return (default from config)
            
        Returns:
            List of results with document, metadata, and distance
        """
        top_k = top_k or config.TOP_K_DOCUMENTS
        
        # Create query embedding
        query_embedding = self.embedding_model.embed_text(query)
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else 0,
                    "similarity": 1 - results['distances'][0][i] if results['distances'] else 1
                })
        
        return formatted_results
    
    def clear(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print("Vector store cleared")
    
    def get_count(self) -> int:
        """Get the number of documents in the store."""
        return self.collection.count()


# Singleton instance
_vector_store = None


def get_vector_store() -> VectorStore:
    """
    Get or create the vector store instance.
    
    Returns:
        VectorStore instance
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def main():
    """Test the vector store"""
    from document_loader import DocumentLoader
    
    # Initialize
    store = get_vector_store()
    loader = DocumentLoader()
    
    # Load and add documents
    knowledge_base_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "knowledge_base.txt"
    )
    
    if os.path.exists(knowledge_base_path):
        # Clear existing data
        store.clear()
        
        # Load and chunk documents
        chunks = loader.load_and_chunk(file_path=knowledge_base_path)
        print(f"Loaded {len(chunks)} chunks")
        
        # Add to vector store
        store.add_documents(chunks)
        
        # Test search
        query = "What is hallucination in LLMs?"
        results = store.search(query)
        
        print(f"\nSearch results for: '{query}'")
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} (similarity: {result['similarity']:.3f}) ---")
            print(result['document'][:200] + "...")
    else:
        print(f"Knowledge base not found at: {knowledge_base_path}")


if __name__ == "__main__":
    main()
