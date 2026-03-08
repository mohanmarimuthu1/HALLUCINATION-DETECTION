"""
Document Loader Module
Handles loading and chunking documents for the knowledge base
"""
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class DocumentLoader:
    """
    Loads and processes documents for the knowledge base.
    Handles text files and creates chunks for embedding.
    """
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize the document loader.
        
        Args:
            chunk_size: Size of text chunks (default from config)
            chunk_overlap: Overlap between chunks (default from config)
        """
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    
    def load_text_file(self, file_path: str) -> str:
        """
        Load content from a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Content of the file as string
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return content
    
    def load_directory(self, directory_path: str, extensions: List[str] = None) -> List[Document]:
        """
        Load all text files from a directory.
        
        Args:
            directory_path: Path to the directory
            extensions: List of file extensions to include (default: ['.txt'])
            
        Returns:
            List of Document objects
        """
        if extensions is None:
            extensions = ['.txt']
        
        documents = []
        
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        for filename in os.listdir(directory_path):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in extensions:
                file_path = os.path.join(directory_path, filename)
                content = self.load_text_file(file_path)
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": file_path,
                        "filename": filename
                    }
                )
                documents.append(doc)
        
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        chunked_docs = []
        
        for doc in documents:
            chunks = self.text_splitter.split_text(doc.page_content)
            
            for i, chunk in enumerate(chunks):
                chunked_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                )
                chunked_docs.append(chunked_doc)
        
        return chunked_docs
    
    def load_and_chunk(self, file_path: str = None, directory_path: str = None) -> List[Document]:
        """
        Load documents and split them into chunks.
        
        Args:
            file_path: Path to a single file (optional)
            directory_path: Path to a directory (optional)
            
        Returns:
            List of chunked Document objects
        """
        documents = []
        
        if file_path:
            content = self.load_text_file(file_path)
            doc = Document(
                page_content=content,
                metadata={"source": file_path, "filename": os.path.basename(file_path)}
            )
            documents.append(doc)
        
        if directory_path:
            dir_docs = self.load_directory(directory_path)
            documents.extend(dir_docs)
        
        if not documents:
            raise ValueError("No file_path or directory_path provided")
        
        return self.chunk_documents(documents)


def main():
    """Test the document loader"""
    loader = DocumentLoader()
    
    # Test loading the knowledge base
    knowledge_base_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "knowledge_base.txt"
    )
    
    if os.path.exists(knowledge_base_path):
        chunks = loader.load_and_chunk(file_path=knowledge_base_path)
        print(f"Loaded {len(chunks)} chunks from knowledge base")
        print(f"\nSample chunk:\n{chunks[0].page_content[:200]}...")
    else:
        print(f"Knowledge base not found at: {knowledge_base_path}")


if __name__ == "__main__":
    main()
