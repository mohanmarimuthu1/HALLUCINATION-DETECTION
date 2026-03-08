"""
Generator Module
Handles LLM-based response generation using Google Gemini with RAG context
"""
from typing import Dict, Any, Optional
import os
import sys
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from rag.retriever import get_retriever

# Import central LLM client
from detection.llm_client import get_llm_client


class Generator:
    """
    Generates responses using Google Gemini with RAG context.
    """
    
    def __init__(self, model_name: str = None, temperature: float = None):
        """
        Initialize the generator.
        
        Args:
            model_name: Model to use (default from config)
            temperature: Generation temperature (default from config)
        """
        self.model_name = model_name or config.LLM_MODEL
        self.temperature = temperature or config.LLM_TEMPERATURE
        
        # Initialize the LLM client
        self.client = get_llm_client()
        
        # Initialize retriever
        self.retriever = get_retriever()
        
        print(f"Generator initialized with model: {self.model_name}")
    
    def _call_gemini(self, prompt: str, max_retries: int = 3) -> str:
        """
        Call LLM API with the given prompt.
        
        Args:
            prompt: The prompt to send
            max_retries: Maximum retry attempts for rate limits
            
        Returns:
            Generated text response
        """
        try:
            return self.client.generate_content(prompt, max_retries=max_retries)
        except Exception as e:
            return f"Error calling LLM API: {str(e)}"
    
    def _create_rag_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt that includes retrieved context.
        
        Args:
            query: User's question
            context: Retrieved context from knowledge base
            
        Returns:
            Formatted prompt
        """
        prompt = f"""You are a highly capable AI assistant interacting with a user. Your goal is to answer their question helpfully and accurately.

You have been provided with supplementary context from a knowledge base. 
If the context is relevant to the question, use it to inform your answer. 
However, if the context DOES NOT contain the answer or the user asks a general practical question (like "How do I code in Python?" or "What's the capital of France?"), you MUST use your own vast general knowledge to answer the question to the best of your ability.

IMPORTANT INSTRUCTIONS:
1. Always try to be helpful and provide a complete answer.
2. If the supplementary context is highly relevant, rely on it primarily.
3. If the context is irrelevant or lacking, ignore it and answer using your own knowledge. 
4. DO NOT say "I don't have enough information" unless you genuinely don't know the answer even with your worldly knowledge.
5. Be concise but comprehensive.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
        return prompt
    
    def _create_direct_prompt(self, query: str) -> str:
        """
        Create a prompt without RAG context (for comparison).
        
        Args:
            query: User's question
            
        Returns:
            Formatted prompt
        """
        prompt = f"""Answer the following question to the best of your ability.

QUESTION: {query}

ANSWER:"""
        return prompt
    
    def generate_with_rag(self, query: str) -> Dict[str, Any]:
        """
        Generate a response using RAG (with retrieved context).
        
        Args:
            query: User's question
            
        Returns:
            Dictionary with response and retrieval info
        """
        # Retrieve relevant context
        retrieval_info = self.retriever.get_retrieval_info(query)
        context = retrieval_info["context"]
        
        # Create prompt with context
        prompt = self._create_rag_prompt(query, context)
        
        # Generate response
        answer = self._call_gemini(prompt)
        
        return {
            "query": query,
            "answer": answer,
            "context": context,
            "retrieval_results": retrieval_info["results"],
            "num_sources": retrieval_info["num_results"],
            "mode": "RAG"
        }
    
    def generate_without_rag(self, query: str) -> Dict[str, Any]:
        """
        Generate a response without RAG (direct LLM).
        
        Args:
            query: User's question
            
        Returns:
            Dictionary with response
        """
        # Create direct prompt
        prompt = self._create_direct_prompt(query)
        
        # Generate response
        answer = self._call_gemini(prompt)
        
        return {
            "query": query,
            "answer": answer,
            "context": None,
            "retrieval_results": [],
            "num_sources": 0,
            "mode": "Direct LLM"
        }
    
    def generate(self, query: str, use_rag: bool = True) -> Dict[str, Any]:
        """
        Generate a response with or without RAG.
        
        Args:
            query: User's question
            use_rag: Whether to use RAG (default True)
            
        Returns:
            Dictionary with response and metadata
        """
        if use_rag:
            return self.generate_with_rag(query)
        else:
            return self.generate_without_rag(query)


# Singleton instance
_generator = None


def get_generator() -> Generator:
    """
    Get or create the generator instance.
    
    Returns:
        Generator instance
    """
    global _generator
    if _generator is None:
        _generator = Generator()
    return _generator


def main():
    """Test the generator"""
    generator = get_generator()
    
    # Test query
    query = "What is hallucination in the context of LLMs?"
    
    print(f"Query: {query}\n")
    print("="*60)
    
    # Generate with RAG
    result = generator.generate_with_rag(query)
    print(f"[RAG Response]")
    print(f"Answer: {result['answer']}")
    print(f"\nNumber of sources used: {result['num_sources']}")


if __name__ == "__main__":
    main()
