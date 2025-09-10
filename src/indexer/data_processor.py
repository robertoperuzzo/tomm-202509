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

    def get_chunks_file(self, document_id: str, extraction_method: str = None,
                        chunking_strategy: str = None) -> Optional[Path]:
        """Get chunks file for a document with strategy-aware discovery."""
        # Try new format first if we have strategy and extraction method
        if extraction_method and chunking_strategy:
            pattern = (f"{document_id}_{extraction_method}_"
                       f"{chunking_strategy}.json")
            chunks_file = self.chunks_data_path / pattern

            if chunks_file.exists():
                return chunks_file

        # Fallback to old pattern for backward compatibility
        old_pattern = f"*{document_id}*.json"
        old_files = list(self.chunks_data_path.glob(old_pattern))

        if old_files:
            if extraction_method and chunking_strategy:
                logger.info("Using legacy chunks file format for %s",
                            document_id)
            if len(old_files) > 1:
                logger.warning("Multiple chunks files found for %s, "
                               "using first: %s", document_id, old_files[0])
            return old_files[0]

        suffix = ""
        if extraction_method and chunking_strategy:
            suffix = f"_{extraction_method}_{chunking_strategy}"
        logger.warning("No chunks file found for %s%s", document_id, suffix)
        return None

    def load_processed_document(self, file_path: Path) -> Optional[
            Dict[str, Any]]:
        """Load a processed document."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.error("Error loading processed document %s: %s",
                         file_path, e)
            return None

    def load_chunks_data(self, file_path: Path,
                         chunking_strategy: str = None) -> Optional[
                             Dict[str, Any]]:
        """Load chunks data with format auto-detection."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Detect format based on structure
            if 'results' in data and 'chunks' in data['results']:
                # New format: direct chunks access
                return data
            elif ('results' in data and chunking_strategy and
                  chunking_strategy in data['results']):
                # Old format: strategy-keyed access
                # Transform to new format for consistent processing
                strategy_data = data['results'][chunking_strategy]
                return {
                    'document_info': data.get('document_info', {}),
                    'strategy_config': strategy_data.get('config', {}),
                    'results': {
                        'chunks': strategy_data.get('chunks', [])
                    }
                }
            else:
                logger.error("Unrecognized chunks data format in %s",
                             file_path)
                return None

        except (IOError, json.JSONDecodeError, KeyError) as e:
            logger.error("Error loading chunks data %s: %s", file_path, e)
            return None

    def get_available_strategies(self, chunks_data: Dict[str, Any]) -> List[
            str]:
        """Get available chunking strategies from chunks data."""
        if 'results' not in chunks_data:
            return []

        return list(chunks_data['results'].keys())

    def prepare_documents_for_indexing(self, extraction_method: str,
                                       chunking_strategy: str,
                                       max_documents: int = -1) -> List[
                                           Dict[str, Any]]:
        """Prepare documents for indexing with new format support."""
        documents = []

        # Get processed files
        processed_files = self.get_processed_files(extraction_method,
                                                   max_documents)

        logger.info("Processing %d documents for %s_%s",
                    len(processed_files), extraction_method,
                    chunking_strategy)

        for file_path in processed_files:
            # Load processed document
            processed_doc = self.load_processed_document(file_path)
            if not processed_doc:
                continue

            # Extract document info
            document_id = processed_doc.get('document_id')
            if not document_id:
                logger.warning("No document_id in %s", file_path)
                continue

            # Use strategy-aware file discovery
            chunks_file = self.get_chunks_file(document_id, extraction_method,
                                               chunking_strategy)
            if not chunks_file:
                continue

            chunks_data = self.load_chunks_data(chunks_file, chunking_strategy)
            if not chunks_data:
                continue

            # Process chunks (same logic as before, but simpler access)
            if ('results' not in chunks_data or
                    'chunks' not in chunks_data['results']):
                logger.warning("No chunks in data for %s", document_id)
                continue

            # Create documents for indexing
            for chunk in chunks_data['results']['chunks']:
                doc = self._create_index_document(
                    processed_doc, chunk, extraction_method,
                    chunking_strategy, chunks_data
                )
                if doc:
                    documents.append(doc)

        logger.info("Prepared %d documents for indexing", len(documents))
        return documents

    def _create_index_document(self, processed_doc: Dict[str, Any],
                               chunk: Dict[str, Any], extraction_method: str,
                               chunking_strategy: str,
                               chunks_data: Dict[str, Any] = None) -> Optional[
                                   Dict[str, Any]]:
        """Create a document for indexing with enhanced metadata."""
        try:
            # Extract metadata
            metadata = chunk.get('metadata', {})
            strategy_config = metadata.get('strategy_config', {})

            # Get strategy config from new format if available
            if chunks_data:
                new_strategy_config = chunks_data.get('strategy_config', {})
                processing_metadata = chunks_data.get(
                    'processing_metadata', {})
            else:
                new_strategy_config = {}
                processing_metadata = {}

            # Create the document with all available fields
            doc = {
                "chunk_id": chunk.get('chunk_id', ''),
                "document_id": processed_doc.get('document_id', ''),
                "document_title": (processed_doc.get('title', '') or
                                   metadata.get('title', '')),
                "document_filename": processed_doc.get('file_name', ''),
                "extraction_method": extraction_method,
                "chunking_strategy": chunking_strategy,
                "strategy_name": chunk.get('strategy_name', chunking_strategy),
                "content": chunk.get('content', ''),
                "token_count": chunk.get('token_count', 0),
                "chunk_index": metadata.get('chunk_index', 0),
                "total_chunks": metadata.get('total_chunks', 0),
                "start_position": chunk.get('start_position', 0),
                "end_position": chunk.get('end_position', 0),
                "created_at": chunk.get('created_at', ''),
            }

            # Add new fields from enhanced format if available
            if chunks_data and 'document_info' in chunks_data:
                doc_info = chunks_data['document_info']
                doc.update({
                    "preprocessing_method": doc_info.get(
                        'preprocessing_method', extraction_method),
                    "content_length": doc_info.get('content_length', 0),
                })

            # Add processing metadata if available
            if processing_metadata:
                doc.update({
                    "processing_time": processing_metadata.get(
                        'processing_time', 0),
                    "memory_usage": processing_metadata.get('memory_usage', 0),
                    "cpu_usage_percent": processing_metadata.get(
                        'cpu_usage_percent', 0),
                    "gpu_usage_percent": processing_metadata.get(
                        'gpu_usage_percent', 0),
                })

            # Add authors if available
            authors = metadata.get('authors', [])
            if authors:
                doc["authors"] = authors

            # Add strategy configuration from old format
            if strategy_config:
                if 'chunk_size' in strategy_config:
                    doc["chunk_size"] = strategy_config['chunk_size']
                if 'chunk_overlap' in strategy_config:
                    doc["chunk_overlap"] = strategy_config['chunk_overlap']
                if 'encoding_name' in strategy_config:
                    doc["encoding_name"] = strategy_config['encoding_name']

            # Add strategy parameters from new format
            if new_strategy_config and 'parameters' in new_strategy_config:
                params = new_strategy_config['parameters']
                doc.update({
                    "chunk_size": params.get('chunk_size', 0),
                    "chunk_overlap": params.get('overlap', 0),
                    "encoding_name": params.get('encoding_name', ''),
                })

            return doc

        except (KeyError, TypeError, ValueError) as e:
            logger.error("Error creating index document: %s", e)
            return None
