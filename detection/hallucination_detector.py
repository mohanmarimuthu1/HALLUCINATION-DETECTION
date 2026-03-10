"""
Hallucination Detector Module
Main detection pipeline that combines claim extraction and fact verification
"""
from typing import Dict, Any, List
import os
import sys
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from detection.claim_extractor import get_claim_extractor
from detection.fact_verifier import get_fact_verifier


class HallucinationDetector:
    """
    Main hallucination detection pipeline.
    Extracts claims from LLM responses and verifies them against retrieved context.
    """
    
    def __init__(self):
        """Initialize the hallucination detector."""
        self.claim_extractor = get_claim_extractor()
        self.fact_verifier = get_fact_verifier()
        
        # Thresholds from config
        self.threshold_low = config.HALLUCINATION_THRESHOLD_LOW
        self.threshold_high = config.HALLUCINATION_THRESHOLD_HIGH
    
    def detect(self, response: str, context: str) -> Dict[str, Any]:
        """
        Detect hallucinations in an LLM response.
        
        Args:
            response: The LLM's generated response
            context: The retrieved context used for generation
            
        Returns:
            Detection results with scores and claim analysis
        """
        # Step 1: Extract claims from the response
        claims = self.claim_extractor.extract_claims(response)
        
        if not claims:
            return {
                "response": response,
                "context": context,
                "claims": [],
                "verification_results": [],
                "summary": {
                    "total_claims": 0,
                    "supported": 0,
                    "contradicted": 0,
                    "not_enough_info": 0,
                    "average_score": 0.0,
                    "hallucination_percentage": 0.0
                },
                "overall_score": 0.15,
                "overall_verdict": "LIKELY_FACTUAL",
                "risk_level": "LOW"
            }
        
        time.sleep(2) # Give API a breather between extraction and validation bursts
        
        # Step 2: Verify each claim against the context
        verification_results = []
        if claims:
            try:
                # Verifier is optimized to verify all claims in one API call
                verification_results = self.fact_verifier.verify_claims(claims, context)
            except Exception as e:
                print(f"Error verifying claims: {e}")
        
        # Step 3: Get summary statistics
        summary = self.fact_verifier.get_summary(verification_results)
        
        # Step 4: Calculate overall verdict
        overall_score = summary["average_score"]
        overall_verdict = self._get_verdict(overall_score)
        risk_level = self._get_risk_level(overall_score)
        
        return {
            "response": response,
            "context": context,
            "claims": claims,
            "verification_results": verification_results,
            "summary": summary,
            "overall_score": overall_score,
            "overall_verdict": overall_verdict,
            "risk_level": risk_level
        }
    
    def _get_verdict(self, score: float) -> str:
        """
        Get overall verdict based on score.
        
        Args:
            score: Hallucination score (0-1)
            
        Returns:
            Verdict string
        """
        if score <= self.threshold_low:
            return "LIKELY_FACTUAL"
        elif score >= self.threshold_high:
            return "LIKELY_HALLUCINATED"
        else:
            return "PARTIALLY_SUPPORTED"
    
    def _get_risk_level(self, score: float) -> str:
        """
        Get risk level based on score.
        
        Args:
            score: Hallucination score (0-1)
            
        Returns:
            Risk level string
        """
        if score <= 0.2:
            return "LOW"
        elif score <= 0.4:
            return "MEDIUM_LOW"
        elif score <= 0.6:
            return "MEDIUM"
        elif score <= 0.8:
            return "MEDIUM_HIGH"
        else:
            return "HIGH"
    
    def get_detailed_report(self, detection_result: Dict[str, Any]) -> str:
        """
        Generate a detailed human-readable report.
        
        Args:
            detection_result: Result from detect() method
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("HALLUCINATION DETECTION REPORT")
        report.append("=" * 60)
        
        # Overall assessment
        report.append(f"\n📊 OVERALL ASSESSMENT")
        report.append(f"   Hallucination Score: {detection_result['overall_score']:.2f} (0 = factual, 1 = hallucinated)")
        report.append(f"   Verdict: {detection_result['overall_verdict']}")
        report.append(f"   Risk Level: {detection_result['risk_level']}")
        
        # Summary
        summary = detection_result['summary']
        report.append(f"\n📈 CLAIM ANALYSIS")
        report.append(f"   Total Claims: {summary['total_claims']}")
        report.append(f"   ✅ Supported: {summary['supported']}")
        report.append(f"   ❌ Contradicted: {summary['contradicted']}")
        report.append(f"   ❓ Not Enough Info: {summary['not_enough_info']}")
        report.append(f"   Hallucination %: {summary['hallucination_percentage']:.1f}%")
        
        # Individual claims
        if detection_result['verification_results']:
            report.append(f"\n📝 CLAIM-BY-CLAIM VERIFICATION")
            for i, result in enumerate(detection_result['verification_results'], 1):
                icon = "✅" if result['verdict'] == "SUPPORTED" else ("❌" if result['verdict'] == "CONTRADICTED" else "❓")
                report.append(f"\n   {i}. {result['claim']}")
                report.append(f"      {icon} {result['verdict']} (Confidence: {result['confidence']})")
                report.append(f"      📌 {result['explanation']}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


# Singleton instance
_hallucination_detector = None


def get_hallucination_detector() -> HallucinationDetector:
    """
    Get or create the hallucination detector instance.
    
    Returns:
        HallucinationDetector instance
    """
    global _hallucination_detector
    if _hallucination_detector is None:
        _hallucination_detector = HallucinationDetector()
    return _hallucination_detector


def main():
    """Test the hallucination detector"""
    detector = get_hallucination_detector()
    
    # Test context
    context = """
    Large Language Models (LLMs) are neural networks trained on large text corpora.
    The transformer architecture was introduced in the paper "Attention is All You Need" in 2017.
    Hallucination in NLP refers to generating information not supported by data.
    RAG (Retrieval Augmented Generation) helps reduce hallucinations by retrieving documents.
    ChromaDB is a popular open-source vector database for RAG applications.
    """
    
    # Test response (mix of factual and hallucinated content)
    response = """
    Large Language Models are neural networks trained on text data.
    The transformer architecture was introduced in 2015 by researchers at Google.
    Hallucination is when models generate incorrect information.
    RAG uses vector databases like ChromaDB to retrieve relevant documents.
    GPT-4 has exactly 1.7 trillion parameters.
    """
    
    print("Testing Hallucination Detector")
    print("="*60)
    print(f"\nCONTEXT:\n{context}")
    print(f"\nRESPONSE TO ANALYZE:\n{response}")
    print("\n")
    
    # Detect hallucinations
    result = detector.detect(response, context)
    
    # Print detailed report
    report = detector.get_detailed_report(result)
    print(report)


if __name__ == "__main__":
    main()
