"""
Embeddings Module
Handles creating vector embeddings for text using sentence-transformers
"""
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class EmbeddingModel:
    """
    Creates embeddings for text using sentence-transformers.
    Runs locally without API calls - completely free!
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformer model (default from config)
        """
        self.model_name = model_name or config.EMBEDDING_MODEL
        print(f"Loading embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        print("Embedding model loaded successfully!")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Create embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings.tolist()
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()


# Singleton instance for reuse
_embedding_model = None


def get_embedding_model() -> EmbeddingModel:
    """
    Get or create the embedding model instance.
    
    Returns:
        EmbeddingModel instance
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model


def main():
    """Test the embedding model"""
    model = get_embedding_model()
    
    # Test embedding
    test_text = "What is hallucination in large language models?"
    embedding = model.embed_text(test_text)
    
    print(f"Test text: {test_text}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 10 values: {embedding[:10]}")


if __name__ == "__main__":
    main()
