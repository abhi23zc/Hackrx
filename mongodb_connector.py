from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from typing import Optional, Dict, Any, List
import logging
from contextlib import contextmanager

class MongoDBConnector:
    """
    A class to handle MongoDB connections and operations.
    """
    
    def __init__(self, username: str, password: str, db_name: str, host: str = "localhost", port: int = 27017):
        """
        Initialize MongoDB connector.
        
        Args:
            username (str): MongoDB username
            password (str): MongoDB password
            db_name (str): Database name
            host (str): MongoDB host (default: localhost)
            port (int): MongoDB port (default: 27017)
        """
        self.username = username
        self.password = password
        self.db_name = db_name
        self.host = host
        self.port = port
        self.client: Optional[MongoClient] = None
        self.database: Optional[Database] = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def get_connection_string(self) -> str:
        """
        Generate the MongoDB connection string.
        
        Returns:
            str: MongoDB connection string
        """
        return f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/{self.db_name}?retryWrites=true"
    
    def connect(self) -> bool:
        """
        Establish connection to MongoDB.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            connection_string = self.get_connection_string()
            self.client = MongoClient(connection_string)
            
            # Test the connection
            self.client.admin.command('ping')
            
            # Get database
            self.database = self.client[self.db_name]
            
            self.logger.info(f"Successfully connected to MongoDB database: {self.db_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to MongoDB: {str(e)}")
            return False
    
    def disconnect(self):
        """
        Close the MongoDB connection.
        """
        if self.client:
            self.client.close()
            self.client = None
            self.database = None
            self.logger.info("MongoDB connection closed")
    
    def is_connected(self) -> bool:
        """
        Check if connected to MongoDB.
        
        Returns:
            bool: True if connected, False otherwise
        """
        try:
            if self.client:
                self.client.admin.command('ping')
                return True
            return False
        except:
            return False
    
    def get_collection(self, collection_name: str) -> Optional[Collection]:
        """
        Get a collection from the database.
        
        Args:
            collection_name (str): Name of the collection
            
        Returns:
            Collection: MongoDB collection object or None if not connected
        """
        if not self.is_connected():
            self.logger.warning("Not connected to MongoDB. Attempting to connect...")
            if not self.connect():
                return None
        
        return self.database[collection_name]
    
    def insert_one(self, collection_name: str, document: Dict[str, Any]) -> Optional[str]:
        """
        Insert a single document into a collection.
        
        Args:
            collection_name (str): Name of the collection
            document (Dict[str, Any]): Document to insert
            
        Returns:
            str: Inserted document ID or None if failed
        """
        collection = self.get_collection(collection_name)
        if not collection:
            return None
        
        try:
            result = collection.insert_one(document)
            self.logger.info(f"Inserted document with ID: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            self.logger.error(f"Failed to insert document: {str(e)}")
            return None
    
    def insert_many(self, collection_name: str, documents: List[Dict[str, Any]]) -> Optional[List[str]]:
        """
        Insert multiple documents into a collection.
        
        Args:
            collection_name (str): Name of the collection
            documents (List[Dict[str, Any]]): List of documents to insert
            
        Returns:
            List[str]: List of inserted document IDs or None if failed
        """
        collection = self.get_collection(collection_name)
        if not collection:
            return None
        
        try:
            result = collection.insert_many(documents)
            inserted_ids = [str(doc_id) for doc_id in result.inserted_ids]
            self.logger.info(f"Inserted {len(inserted_ids)} documents")
            return inserted_ids
        except Exception as e:
            self.logger.error(f"Failed to insert documents: {str(e)}")
            return None
    
    def find_one(self, collection_name: str, filter_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Find a single document in a collection.
        
        Args:
            collection_name (str): Name of the collection
            filter_dict (Dict[str, Any]): Filter criteria
            
        Returns:
            Dict[str, Any]: Found document or None if not found
        """
        collection = self.get_collection(collection_name)
        if not collection:
            return None
        
        try:
            return collection.find_one(filter_dict)
        except Exception as e:
            self.logger.error(f"Failed to find document: {str(e)}")
            return None
    
    def find_many(self, collection_name: str, filter_dict: Dict[str, Any] = None, limit: int = 0) -> Optional[List[Dict[str, Any]]]:
        """
        Find multiple documents in a collection.
        
        Args:
            collection_name (str): Name of the collection
            filter_dict (Dict[str, Any]): Filter criteria (default: None for all documents)
            limit (int): Maximum number of documents to return (default: 0 for no limit)
            
        Returns:
            List[Dict[str, Any]]: List of found documents or None if failed
        """
        collection = self.get_collection(collection_name)
        if not collection:
            return None
        
        try:
            cursor = collection.find(filter_dict or {})
            if limit > 0:
                cursor = cursor.limit(limit)
            return list(cursor)
        except Exception as e:
            self.logger.error(f"Failed to find documents: {str(e)}")
            return None
    
    def update_one(self, collection_name: str, filter_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> bool:
        """
        Update a single document in a collection.
        
        Args:
            collection_name (str): Name of the collection
            filter_dict (Dict[str, Any]): Filter criteria
            update_dict (Dict[str, Any]): Update operations
            
        Returns:
            bool: True if update successful, False otherwise
        """
        collection = self.get_collection(collection_name)
        if not collection:
            return False
        
        try:
            result = collection.update_one(filter_dict, update_dict)
            self.logger.info(f"Updated {result.modified_count} document(s)")
            return result.modified_count > 0
        except Exception as e:
            self.logger.error(f"Failed to update document: {str(e)}")
            return False
    
    def delete_one(self, collection_name: str, filter_dict: Dict[str, Any]) -> bool:
        """
        Delete a single document from a collection.
        
        Args:
            collection_name (str): Name of the collection
            filter_dict (Dict[str, Any]): Filter criteria
            
        Returns:
            bool: True if deletion successful, False otherwise
        """
        collection = self.get_collection(collection_name)
        if not collection:
            return False
        
        try:
            result = collection.delete_one(filter_dict)
            self.logger.info(f"Deleted {result.deleted_count} document(s)")
            return result.deleted_count > 0
        except Exception as e:
            self.logger.error(f"Failed to delete document: {str(e)}")
            return False
    
    def upsert_one(self, collection_name: str, filter_dict: Dict[str, Any], document: Dict[str, Any]) -> bool:
        """
        Insert or update a document in a collection.
        
        Args:
            collection_name (str): Name of the collection
            filter_dict (Dict[str, Any]): Filter criteria to find existing document
            document (Dict[str, Any]): Document to insert/update
            
        Returns:
            bool: True if operation successful, False otherwise
        """
        collection = self.get_collection(collection_name)
        if not collection:
            return False
        
        try:
            result = collection.replace_one(filter_dict, document, upsert=True)
            self.logger.info(f"Upserted document: {result.upserted_id if result.upserted_id else 'updated existing'}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to upsert document: {str(e)}")
            return False
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for MongoDB connection.
        
        Usage:
            with mongodb_connector.get_connection() as db:
                # Use db for operations
                pass
        """
        try:
            if not self.is_connected():
                self.connect()
            yield self.database
        finally:
            # Connection will be managed by the class
            pass
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect() 