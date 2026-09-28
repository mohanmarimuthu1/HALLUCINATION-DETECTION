"""
Setup and Initialization Script
Run this first to set up the project environment and knowledge base
"""
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def main():
    print("=" * 60)
    print("🚀 HALLUCINATION DETECTION SYSTEM - SETUP")
    print("=" * 60)
    
    # Step 1: Check config
    print("\n📋 Step 1: Checking configuration...")
    try:
        import config
        print(f"   ✅ Config loaded successfully")
        print(f"   📌 LLM Model: {config.LLM_MODEL}")
        print(f"   📌 Embedding Model: {config.EMBEDDING_MODEL}")
    except Exception as e:
        print(f"   ❌ Error loading config: {e}")
        return
    
    # Step 2: Initialize embedding model
    print("\n🧠 Step 2: Loading embedding model (this may take a moment first time)...")
    try:
        from knowledge_base.embeddings import get_embedding_model
        embedding_model = get_embedding_model()
        test_embedding = embedding_model.embed_text("test")
        print(f"   ✅ Embedding model loaded")
        print(f"   📌 Embedding dimension: {len(test_embedding)}")
    except Exception as e:
        print(f"   ❌ Error loading embedding model: {e}")
        return
    
    # Step 3: Load documents
    print("\n📚 Step 3: Loading knowledge base documents...")
    try:
        from knowledge_base.document_loader import DocumentLoader
        loader = DocumentLoader()
        
        knowledge_base_path = os.path.join(project_root, "data", "knowledge_base.txt")
        
        if os.path.exists(knowledge_base_path):
            chunks = loader.load_and_chunk(file_path=knowledge_base_path)
            print(f"   ✅ Loaded {len(chunks)} document chunks")
        else:
            print(f"   ⚠️ Knowledge base not found at: {knowledge_base_path}")
            return
    except Exception as e:
        print(f"   ❌ Error loading documents: {e}")
        return
    
    # Step 4: Create vector store
    print("\n🗄️ Step 4: Building vector store...")
    try:
        from knowledge_base.vector_store import get_vector_store
        vector_store = get_vector_store()
        
        # Clear and rebuild
        vector_store.clear()
        vector_store.add_documents(chunks)
        
        print(f"   ✅ Vector store created with {vector_store.get_count()} documents")
    except Exception as e:
        print(f"   ❌ Error creating vector store: {e}")
        return
    
    # Step 5: Test retrieval
    print("\n🔍 Step 5: Testing retrieval...")
    try:
        from rag.retriever import get_retriever
        retriever = get_retriever()
        
        test_query = "What is hallucination in LLMs?"
        results = retriever.retrieve(test_query)
        
        print(f"   ✅ Retrieval working")
        print(f"   📌 Query: '{test_query}'")
        print(f"   📌 Found {len(results)} relevant documents")
    except Exception as e:
        print(f"   ❌ Error testing retrieval: {e}")
        return
    
    # Step 6: Test LLM
    print("\n🤖 Step 6: Testing LLM connection...")
    try:
        from rag.generator import get_generator
        generator = get_generator()
        
        result = generator.generate_with_rag("What is RAG?")
        print(f"   ✅ LLM connection working")
        print(f"   📌 Generated response: {result['answer'][:100]}...")
    except Exception as e:
        print(f"   ❌ Error testing LLM: {e}")
        print(f"   💡 Make sure your API key is correct in config.py")
        return
    
    # Step 7: Test hallucination detection
    print("\n🔬 Step 7: Testing hallucination detection...")
    try:
        from detection.hallucination_detector import get_hallucination_detector
        detector = get_hallucination_detector()
        
        test_response = "LLMs are neural networks. The transformer was introduced in 2017."
        test_context = "Large Language Models are neural networks. The transformer architecture was introduced in 2017."
        
        detection = detector.detect(test_response, test_context)
        print(f"   ✅ Hallucination detection working")
        print(f"   📌 Score: {detection['overall_score']:.2f}")
        print(f"   📌 Verdict: {detection['overall_verdict']}")
    except Exception as e:
        print(f"   ❌ Error testing hallucination detection: {e}")
        return
    
    # Success!
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("=" * 60)
    print("""
To run the web application:
    
    streamlit run app.py

The app will open in your browser at http://localhost:8501
    """)


if __name__ == "__main__":
    main()
