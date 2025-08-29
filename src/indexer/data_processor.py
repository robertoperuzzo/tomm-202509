"""
Data processor for preparing documents for indexing.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class DataProcessor:
    """Processes document data for indexing."""
    
    def __init__(self, processed_data_path: str, chunks_data_path: str):
        """Initialize data processor."""
        self.processed_data_path = Path(processed_data_path)
        self.chunks_data_path = Path(chunks_data_path)
    
    def get_available_extraction_methods(self) -> List[str]:
        """Get list of available extraction methods."""
        methods = []
        for item in self.processed_data_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                methods.append(item.name)
        return sorted(methods)
    
    def get_processed_files(self, extraction_method: str,
                           max_documents: int = -1) -> List[Path]:
        """Get processed files for an extraction method."""
        method_path = self.processed_data_path / extraction_method
        
        if not method_path.exists():
            logger.warning(f"Path not found: {method_path}")
            return []
        
        # Get all JSON files
        json_files = list(method_path.glob("*.json"))
        
        if max_documents > 0:
            json_files = json_files[:max_documents]
            logger.info(f"Limited to {len(json_files)} documents for "
                       f"{extraction_method}")
        
        return sorted(json_files)
    
    def get_chunks_file(self, document_id: str) -> Optional[Path]:
        """Get chunks file for a document."""
        # Look for chunks file
        pattern = f"*{document_id}*.json"
        chunks_files = list(self.chunks_data_path.glob(pattern))
        
        if not chunks_files:
            logger.warning(f"No chunks file found for document: {document_id}")
            return None
        
        if len(chunks_files) > 1:
            logger.warning(f"Multiple chunks files found for {document_id}, "
                          f"using first: {chunks_files[0]}")
        
        return chunks_files[0]
    
    def load_processed_document(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load a processed document."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading processed document {file_path}: {e}")
            return None
    
    def load_chunks_data(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load chunks data."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading chunks data {file_path}: {e}")
            return None
    
    def get_available_strategies(self, chunks_data: Dict[str, Any]) -> List[str]:
        """Get available chunking strategies from chunks data."""
        if 'results' not in chunks_data:
            return []
        
        return list(chunks_data['results'].keys())
    
    def prepare_documents_for_indexing(self, extraction_method: str,
                                     chunking_strategy: str,
                                     max_documents: int = -1) -> List[Dict[str, Any]]:
        """Prepare documents for indexing."""
        documents = []
        
        # Get processed files
        processed_files = self.get_processed_files(extraction_method,
                                                 max_documents)
        
        logger.info(f"Processing {len(processed_files)} documents for "
                   f"{extraction_method}_{chunking_strategy}")
        
        for file_path in processed_files:
            # Load processed document
            processed_doc = self.load_processed_document(file_path)
            if not processed_doc:
                continue
            
            # Extract document info
            document_id = processed_doc.get('document_id')
            if not document_id:
                logger.warning(f"No document_id in {file_path}")
                continue
            
            # Load chunks data
            chunks_file = self.get_chunks_file(document_id)
            if not chunks_file:
                continue
            
            chunks_data = self.load_chunks_data(chunks_file)
            if not chunks_data:
                continue
            
            # Check if strategy exists
            if 'results' not in chunks_data:
                logger.warning(f"No results in chunks data for {document_id}")
                continue
            
            if chunking_strategy not in chunks_data['results']:
                logger.warning(f"Strategy {chunking_strategy} not found in "
                              f"chunks data for {document_id}")
                continue
            
            # Process chunks for this strategy
            strategy_data = chunks_data['results'][chunking_strategy]
            if 'chunks' not in strategy_data:
                logger.warning(f"No chunks in strategy data for {document_id}")
                continue
            
            # Create documents for indexing
            for chunk in strategy_data['chunks']:
                doc = self._create_index_document(
                    processed_doc, chunk, extraction_method, chunking_strategy
                )
                if doc:
                    documents.append(doc)
        
        logger.info(f"Prepared {len(documents)} documents for indexing")
        return documents
    
    def _create_index_document(self, processed_doc: Dict[str, Any],
                              chunk: Dict[str, Any], extraction_method: str,
                              chunking_strategy: str) -> Optional[Dict[str, Any]]:
        """Create a document for indexing."""
        try:
            return {
                "chunk_id": chunk.get('chunk_id', ''),
                "document_id": processed_doc.get('document_id', ''),
                "document_title": processed_doc.get('title', ''),
                "document_filename": processed_doc.get('file_name', ''),
                "extraction_method": extraction_method,
                "chunking_strategy": chunking_strategy,
                "content": chunk.get('content', ''),
                "token_count": chunk.get('token_count', 0),
                "chunk_index": chunk.get('metadata', {}).get('chunk_index', 0),
                "start_position": chunk.get('start_position', 0),
                "end_position": chunk.get('end_position', 0)
            }
        except Exception as e:
            logger.error(f"Error creating index document: {e}")
            return None
