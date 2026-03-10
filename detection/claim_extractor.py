"""
Claim Extractor Module
Extracts factual claims from LLM responses for verification
"""
from typing import List, Dict, Any
import google.generativeai as genai
import re
import os
import sys
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# Import central LLM client
from detection.llm_client import get_llm_client

class ClaimExtractor:
    """
    Extracts individual factual claims from text for verification.
    Uses LLM to parse text into atomic claims.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the claim extractor.
        
        Args:
            model_name: Gemini model to use
        """
        self.model_name = model_name or config.LLM_MODEL
        self.client = get_llm_client()
    
    def extract_claims(self, text: str, max_retries: int = 3) -> List[str]:
        """
        Extract factual claims from text.
        
        Args:
            text: The text to extract claims from
            max_retries: Maximum number of retries on rate limit
            
        Returns:
            List of extracted claims
        """
        prompt = f"""Extract the main factual claims from this text. Output ONLY a numbered list.

TEXT: {text}

CLAIMS (numbered list only):"""

        try:
            claims_text = self.client.generate_content(prompt)
            # Parse the claims
            claims = self._parse_claims(claims_text)

            # If the fallback failed completely due to exhausted keys
            if not claims and "NO_CLAIMS_POSSIBLE" in claims_text:
                return self._simple_extract(text)

            # Fallback for empty results
            if not claims:
               return self._simple_extract(text)
            
            return claims
            
        except Exception as e:
            print(f"Error extracting claims: {e}")
            # Try simple extraction as absolute fallback
            return self._simple_extract(text)
    
    def _simple_extract(self, text: str) -> List[str]:
        """
        Simple rule-based claim extraction as fallback.
        Splits text into sentences as claims.
        """
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        claims = []
        for s in sentences:
            s = s.strip()
            # Filter out short sentences and questions
            if len(s) > 20 and not s.endswith('?'):
                claims.append(s)
        return claims[:5]  # Return max 5 claims
    
    def _parse_claims(self, claims_text: str) -> List[str]:
        """
        Parse the LLM output into a list of claims.
        
        Args:
            claims_text: Raw LLM output
            
        Returns:
            List of claims
        """
        if "NO_CLAIMS" in claims_text.upper():
            return []
        
        claims = []
        lines = claims_text.strip().split('\n')
        
        for line in lines:
            # Remove numbering and clean up
            line = line.strip()
            if not line:
                continue
                
            # Remove common prefixes like "1.", "1)", "- ", etc.
            cleaned = re.sub(r'^[\d]+[\.\)]\s*', '', line)
            cleaned = re.sub(r'^[-\*]\s*', '', cleaned)
            cleaned = cleaned.strip()
            
            if cleaned and len(cleaned) > 10:  # Filter out very short strings
                claims.append(cleaned)
        
        return claims[:2]  # Limit to 2 claims to heavily reduce API calls


# Singleton instance
_claim_extractor = None


def get_claim_extractor() -> ClaimExtractor:
    """
    Get or create the claim extractor instance.
    
    Returns:
        ClaimExtractor instance
    """
    global _claim_extractor
    if _claim_extractor is None:
        _claim_extractor = ClaimExtractor()
    return _claim_extractor
