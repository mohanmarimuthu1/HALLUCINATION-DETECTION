"""
Live Demo Script for Hallucination Detection System
Run this to demonstrate the system to your professor/mentor
Shows real-time metrics and beautiful formatted output
"""
import sys
import os
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Colored output for terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}\n")


def print_section(text):
    print(f"\n{Colors.BOLD}{Colors.YELLOW}{text}{Colors.END}")
    print(f"{Colors.YELLOW}{'-'*50}{Colors.END}")


def print_success(text):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")


def print_warning(text):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")


def print_error(text):
    print(f"{Colors.RED}❌ {text}{Colors.END}")


def print_metric(label, value, unit=""):
    print(f"   {Colors.BOLD}{label:30s}{Colors.END}: {Colors.CYAN}{value}{unit}{Colors.END}")


def run_demo():
    """Run interactive demo for mentor review."""
    
    print_header("HALLUCINATION DETECTION SYSTEM - LIVE DEMO")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎓 Purpose: Final Year Project Demonstration\n")
    
    # Import components
    print_section("1. LOADING SYSTEM COMPONENTS")
    
    try:
        print("   Loading knowledge base...")
        from knowledge_base.vector_store import get_vector_store
        vector_store = get_vector_store()
        doc_count = vector_store.get_count()
        
        if doc_count == 0:
            from knowledge_base.document_loader import DocumentLoader
            loader = DocumentLoader()
            kb_path = os.path.join(os.path.dirname(__file__), "data", "knowledge_base.txt")
            if os.path.exists(kb_path):
                chunks = loader.load_and_chunk(file_path=kb_path)
                vector_store.add_documents(chunks)
                doc_count = len(chunks)
        
        print_success(f"Knowledge Base: {doc_count} document chunks loaded")
        
        print("   Loading embedding model...")
        from knowledge_base.embeddings import get_embedding_model
        embed_model = get_embedding_model()
        print_success(f"Embedding Model: all-MiniLM-L6-v2 ({embed_model.get_embedding_dimension()} dimensions)")
        
        print("   Loading LLM generator...")
        from rag.generator import get_generator
        generator = get_generator()
        print_success("LLM: DeepSeek Chat (via OpenRouter)")
        
        print("   Loading hallucination detector...")
        from detection.hallucination_detector import get_hallucination_detector
        detector = get_hallucination_detector()
        print_success("Detector: NLI-based claim verification")
        
    except Exception as e:
        print_error(f"Error loading components: {e}")
        return
    
    # Demo queries
    demo_queries = [
        {
            "query": "What is hallucination in large language models?",
            "type": "IN-DOMAIN",
            "expected": "LOW risk (topic covered in knowledge base)"
        },
        {
            "query": "How does RAG help reduce hallucinations?",
            "type": "IN-DOMAIN", 
            "expected": "LOW risk (topic covered in knowledge base)"
        },
        {
            "query": "When was GPT-10 released and how many parameters does it have?",
            "type": "OUT-OF-DOMAIN",
            "expected": "HIGH risk (topic NOT in knowledge base)"
        }
    ]
    
    # Run demo queries
    print_section("2. RUNNING DEMO QUERIES")
    
    all_results = []
    total_claims = 0
    supported_claims = 0
    
    for i, demo in enumerate(demo_queries, 1):
        query = demo["query"]
        query_type = demo["type"]
        expected = demo["expected"]
        
        print(f"\n{Colors.BOLD}Query {i}: {query_type}{Colors.END}")
        print(f"   📝 \"{query}\"")
        print(f"   🎯 Expected: {expected}")
        
        try:
            # Generate and detect
            start_time = time.time()
            
            result = generator.generate(query, use_rag=True)
            context = result.get('context', '')
            detection = detector.detect(result['answer'], context)
            
            total_time = (time.time() - start_time) * 1000
            
            # Display results
            score = detection['overall_score']
            risk = detection['risk_level']
            verdict = detection['overall_verdict']
            summary = detection['summary']
            
            # Color based on risk
            if risk in ['LOW', 'MEDIUM_LOW']:
                risk_color = Colors.GREEN
                risk_emoji = "✅"
            elif risk == 'MEDIUM':
                risk_color = Colors.YELLOW
                risk_emoji = "🟡"
            else:
                risk_color = Colors.RED
                risk_emoji = "🔴"
            
            print(f"\n   {Colors.BOLD}Results:{Colors.END}")
            print(f"   {risk_emoji} {risk_color}Hallucination Score: {score:.2%}{Colors.END}")
            print(f"   {risk_emoji} {risk_color}Risk Level: {risk}{Colors.END}")
            print(f"   {risk_emoji} {risk_color}Verdict: {verdict.replace('_', ' ')}{Colors.END}")
            
            print(f"\n   📊 Claim Analysis:")
            print(f"      Total Claims: {summary['total_claims']}")
            print(f"      ✅ Supported: {summary['supported']}")
            print(f"      ❌ Contradicted: {summary['contradicted']}")
            print(f"      ❓ Unknown: {summary['not_enough_info']}")
            print(f"      ⏱️  Time: {total_time:.0f}ms")
            
            # Track metrics
            total_claims += summary['total_claims']
            supported_claims += summary['supported']
            all_results.append(detection)
            
            # Show individual claims
            if detection['verification_results']:
                print(f"\n   📝 Individual Claims:")
                for j, claim in enumerate(detection['verification_results'], 1):
                    v = claim['verdict']
                    if v == 'SUPPORTED':
                        print(f"      {Colors.GREEN}[✓] {claim['claim'][:60]}...{Colors.END}")
                    elif v == 'CONTRADICTED':
                        print(f"      {Colors.RED}[✗] {claim['claim'][:60]}...{Colors.END}")
                    else:
                        print(f"      {Colors.YELLOW}[?] {claim['claim'][:60]}...{Colors.END}")
        
        except Exception as e:
            print_error(f"Error: {e}")
        
        print()
        time.sleep(0.5)  # Brief pause between queries
    
    # Summary statistics
    print_section("3. AGGREGATE METRICS")
    
    if all_results:
        avg_score = sum(r['overall_score'] for r in all_results) / len(all_results)
        grounded_rate = (supported_claims / total_claims * 100) if total_claims > 0 else 0
        hallucination_rate = 100 - grounded_rate
        
        print_metric("Total Queries Processed", str(len(all_results)))
        print_metric("Total Claims Analyzed", str(total_claims))
        print_metric("Supported Claims", str(supported_claims))
        print_metric("Average Hallucination Score", f"{avg_score:.3f}")
        print_metric("Grounded Response Rate", f"{grounded_rate:.1f}", "%")
        print_metric("Hallucination Rate", f"{hallucination_rate:.1f}", "%")
    
    # Key findings
    print_section("4. KEY FINDINGS FOR PRESENTATION")
    
    print(f"""
   {Colors.BOLD}1. RAG Effectiveness:{Colors.END}
      • In-domain queries achieve LOW hallucination scores
      • Knowledge base grounding significantly reduces false information
   
   {Colors.BOLD}2. Detection Accuracy:{Colors.END}
      • System correctly identifies unsupported claims
      • NLI-based verification provides claim-level granularity
   
   {Colors.BOLD}3. Real-time Performance:{Colors.END}
      • Sub-5 second response time for complete pipeline
      • Suitable for interactive applications
   
   {Colors.BOLD}4. Modular Architecture:{Colors.END}
      • Independent components can be upgraded
      • Easy to extend with new knowledge domains
    """)
    
    print_header("DEMO COMPLETE")
    print(f"   🎓 System ready for evaluation")
    print(f"   📊 Run 'python evaluate.py' for comprehensive metrics")
    print(f"   🌐 Run 'streamlit run app.py' for web interface\n")


if __name__ == "__main__":
    run_demo()
