"""Fixed-size chunking strategy implementation.

This strategy splits documents into chunks of a fixed token count,
providing a simple baseline for comparison with other strategies.
"""

from typing import Dict, List, Any
import logging

from ..base import ChunkingStrategy, TokenCounter, ChunkingError
from ..models import ProcessedDocument, DocumentChunk
from ..utils import clean_text


logger = logging.getLogger(__name__)


class FixedSizeChunker(ChunkingStrategy):
    """Fixed-size token-based chunking strategy.
    
    This strategy splits text into chunks with a fixed number of tokens,
    with optional overlap between chunks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the fixed-size chunker.
        
        Args:
            config: Configuration dictionary containing:
                - chunk_size: Number of tokens per chunk (default: 1024)
                - chunk_overlap: Number of tokens to overlap (default: 0)
                - encoding_name: Tokenizer encoding name (default: cl100k_base)
        """
        super().__init__(config)
        self.strategy_name = "fixed_size"  # Override base class inference
        
        self.chunk_size = config.get('chunk_size', 1024)
        self.chunk_overlap = config.get('chunk_overlap', 0)
        self.encoding_name = config.get('encoding_name', 'cl100k_base')
        
        # Initialize token counter
        self.token_counter = TokenCounter(self.encoding_name)
        
        # Validate configuration
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
            
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
    
    def chunk_document(
        self, document: ProcessedDocument
    ) -> List[DocumentChunk]:
        """Chunk document using fixed-size token-based splitting.
        
        Args:
            document: The document to chunk
            
        Returns:
            List of document chunks
            
        Raises:
            ChunkingError: If chunking fails
        """
        if not self.validate_document(document):
            raise ChunkingError(
                f"Document validation failed for {document.document_id}",
                strategy_name=self.strategy_name,
                document_id=document.document_id
            )
        
        try:
            # Clean the text
            text = clean_text(document.full_text)
            
            if not text:
                logger.warning(
                    f"No text content for document {document.document_id}"
                )
                return []
            
            # Split text using LangChain's TokenTextSplitter
            chunks = self._split_text_by_tokens(text)
            
            # Create DocumentChunk objects
            document_chunks = []
            for i, chunk_text in enumerate(chunks):
                chunk = self._create_document_chunk(
                    document, chunk_text, i
                )
                document_chunks.append(chunk)
            
            logger.info(
                f"Created {len(document_chunks)} chunks for document "
                f"{document.document_id} using fixed-size strategy"
            )
            
            return document_chunks
            
        except Exception as e:
            logger.error(f"Fixed-size chunking failed: {str(e)}")
            raise ChunkingError(
                f"Fixed-size chunking failed: {str(e)}",
                strategy_name=self.strategy_name,
                document_id=document.document_id
            )
    
    def _split_text_by_tokens(self, text: str) -> List[str]:
        """Split text into chunks by token count.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        try:
            # Try to use LangChain's TokenTextSplitter
            from langchain_text_splitters import TokenTextSplitter
            
            splitter = TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                encoding_name=self.encoding_name
            )
            
            return splitter.split_text(text)
            
        except ImportError:
            logger.warning(
                "LangChain TokenTextSplitter not available, "
                "using fallback implementation"
            )
            return self._fallback_split_text(text)
    
    def _fallback_split_text(self, text: str) -> List[str]:
        """Fallback text splitting when LangChain is not available.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        words = text.split()
        if not words:
            return []
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for word in words:
            # Estimate tokens for this word (rough approximation)
            word_tokens = max(1, len(word) // 4)
            
            if (current_tokens + word_tokens > self.chunk_size and
                    current_chunk):
                
                # Finalize current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    overlap_words = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_words + [word]
                    current_tokens = sum(
                        max(1, len(w) // 4) for w in current_chunk
                    )
                else:
                    current_chunk = [word]
                    current_tokens = word_tokens
            else:
                current_chunk.append(word)
                current_tokens += word_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks
    
    def _create_document_chunk(
        self, 
        document: ProcessedDocument, 
        chunk_text: str, 
        chunk_index: int
    ) -> DocumentChunk:
        """Create a DocumentChunk from text and metadata.
        
        Args:
            document: Source document
            chunk_text: Text content of the chunk
            chunk_index: Index of this chunk in the document
            
        Returns:
            DocumentChunk object
        """
        # Calculate positions (approximate for token-based splitting)
        start_pos = chunk_index * (self.chunk_size - self.chunk_overlap)
        end_pos = start_pos + len(chunk_text)
        
        # Count actual tokens in the chunk
        token_count = self.token_counter.count_tokens(chunk_text)
        
        return DocumentChunk(
            chunk_id=self.generate_chunk_id(document.document_id, chunk_index),
            document_id=document.document_id,
            strategy_name=self.strategy_name,
            content=chunk_text,
            start_position=max(0, start_pos),
            end_position=end_pos,
            token_count=token_count,
            metadata={
                'title': document.title,
                'authors': document.authors,
                'chunk_index': chunk_index,
                'total_chunks': 0,  # Will be updated by pipeline
                'strategy_config': self.get_strategy_config()
            }
        )
    
    def get_strategy_config(self) -> Dict[str, Any]:
        """Get the current configuration for this strategy.
        
        Returns:
            Dictionary containing strategy configuration
        """
        return {
            'strategy_name': self.strategy_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'encoding_name': self.encoding_name
        }
