"""
Vector database operations using ChromaDB
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import uuid
from datetime import datetime


class VectorDatabase:
    """Manage vector database operations"""
    
    def __init__(self, persist_directory: str, collection_name: str):
        """
        Initialize ChromaDB client
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection
        """
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: Optional[List[str]] = None):
        """
        Add documents to the vector database
        
        Args:
            documents: List of text documents
            metadatas: List of metadata dictionaries
            ids: Optional list of document IDs
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        # Add timestamp to metadata
        for metadata in metadatas:
            metadata['timestamp'] = datetime.now().isoformat()
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(self, query_text: str, n_results: int = 5, filter_dict: Optional[Dict] = None) -> Dict:
        """
        Query the vector database
        
        Args:
            query_text: Query string
            n_results: Number of results to return
            filter_dict: Optional filter dictionary
            
        Returns:
            Query results
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=filter_dict
        )
        return results
    
    def get_all_documents(self) -> Dict:
        """
        Get all documents from the collection
        
        Returns:
            All documents with metadata
        """
        return self.collection.get()
    
    def delete_collection(self):
        """Delete the entire collection"""
        self.client.delete_collection(name=self.collection.name)
    
    def count_documents(self) -> int:
        """
        Get count of documents in collection
        
        Returns:
            Number of documents
        """
        return self.collection.count()
    
    def delete_documents(self, ids: List[str]):
        """
        Delete specific documents by ID
        
        Args:
            ids: List of document IDs to delete
        """
        self.collection.delete(ids=ids)
