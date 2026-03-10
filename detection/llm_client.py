"""
LLM Client Module
Centralized module for handling LLM API calls with automatic fallback
from Gemini to OpenRouter (DeepSeek) when rate limits are hit.
"""
import google.generativeai as genai
import requests
import json
import time
import os
import sys
from typing import Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Configure Gemini API
genai.configure(api_key=config.GOOGLE_API_KEY)

class LLMClient:
    """
    Client for interacting with LLMs.
    Handles primary API calls to Gemini and falls back to OpenRouter
    if rate limits are encountered.
    """
    def __init__(self, model_name: str = None):
        """
        Initialize the LLM client.
        
        Args:
            model_name: Primary Gemini model to use
        """
        self.primary_model_name = model_name or config.LLM_MODEL
        self.gemini_model = genai.GenerativeModel(
            model_name=self.primary_model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,  # Low temperature
            )
        )
        self.openrouter_keys = getattr(config, 'OPENROUTER_API_KEYS', [])
        self.current_key_idx = 0
        
    def generate_content(self, prompt: str, max_retries: int = 3) -> str:
        """
        Generate content using Gemini, with fallback to OpenRouter.
        
        Args:
            prompt: Required text prompt
            max_retries: How many times to retry API errors
            
        Returns:
            Generated text
        """
        # Try Gemini first
        for attempt in range(max_retries):
            try:
                response = self.gemini_model.generate_content(prompt)
                return response.text
                
            except Exception as e:
                error_str = str(e).lower()
                # If quota is exhausted, fall back to OpenRouter immediately
                if "quota" in error_str:
                    print(f"\\n[LLM Client] Gemini quota exhausted. Falling back to OpenRouter/DeepSeek...")
                    return self._generate_with_openrouter(prompt)
                # If it's a rate limit (15 RPM), wait very long and retry instead of failing
                elif "429" in error_str or "too many requests" in error_str:
                    wait_time = 20 * (attempt + 1)  # 20s, 40s, 60s
                    print(f"\\n[LLM Client] Gemini rate limit hit. Waiting {wait_time}s before retry {attempt+1}/{max_retries}...")
                    time.sleep(wait_time)
                    if attempt == max_retries - 1:
                        return self._generate_with_openrouter(prompt)
                else:
                    print(f"\\n[LLM Client] Gemini error: {e}")
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5
                        time.sleep(wait_time)
                    else:
                        print(f"\\n[LLM Client] Max retries reached for Gemini. Falling back to OpenRouter...")
                        return self._generate_with_openrouter(prompt)
        
        return self._generate_with_openrouter(prompt)

    def _generate_with_openrouter(self, prompt: str) -> str:
        """
        Fallback using OpenRouter API, rotating through keys if needed.
        """
        if not self.openrouter_keys:
            print("[LLM Client] Extended fallback failed: No OpenRouter keys configured in config.py")
            return "NO_CLAIMS_POSSIBLE_DUE_TO_API_LIMITS" if "claim" in prompt.lower() else "Unable to verify due to API limits"

        max_attempts = len(self.openrouter_keys)
        
        for _ in range(max_attempts):
            api_key = self.openrouter_keys[self.current_key_idx]
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": config.DEEPSEEK_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1
            }
            
            try:
                response = requests.post(
                    config.OPENROUTER_BASE_URL + "/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    response_json = response.json()
                    return response_json['choices'][0]['message']['content']
                    
                elif response.status_code == 429 or response.status_code == 402:  # Rate limit or quota
                    print(f"[LLM Client] OpenRouter key at index {self.current_key_idx} exhausted/rate-limited. Rotating to next key.")
                    self._rotate_key()
                else:
                    print(f"[LLM Client] OpenRouter error ({response.status_code}): {response.text}")
                    # Might be a temporary glitch, try rotating anyway just to be safe
                    self._rotate_key()
                    
            except Exception as e:
                print(f"[LLM Client] Connection error to OpenRouter: {e}")
                self._rotate_key()
                
        # If we exhausted all keys
        print("[LLM Client] All configured backup keys failed.")
        return "NO_CLAIMS_POSSIBLE" if "claim" in prompt.lower() else "NOT_ENOUGH_INFO: Insufficient evidence in context to verify."
        
    def _rotate_key(self):
        """Rotate to the next OpenRouter key in the list."""
        self.current_key_idx = (self.current_key_idx + 1) % len(self.openrouter_keys)

# Singleton configuration
_current_client = None

def get_llm_client() -> LLMClient:
    """Return a singleton LLM client."""
    global _current_client
    if _current_client is None:
        _current_client = LLMClient()
    return _current_client
