"""Semantic chunking strategy implementation.

This strategy creates chunks based on semantic similarity, identifying
natural breakpoints in the document based on meaning rather than structure.
"""

from typing import Dict, List, Any
import logging

from ..base import ChunkingStrategy, TokenCounter, ChunkingError
from ..models import ProcessedDocument, DocumentChunk
from ..utils import clean_text


logger = logging.getLogger(__name__)


class SemanticChunker(ChunkingStrategy):
    """Semantic chunking strategy based on sentence similarity.

    This strategy analyzes semantic similarity between sentences to
    identify natural breakpoints for chunking.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the semantic chunker.

        Args:
            config: Configuration dictionary containing:
                - embedding_model: Model for embeddings
                  (default: 'sentence-transformers/all-MiniLM-L6-v2')
                - similarity_threshold: Threshold for similarity (default: 0.8)
                - min_chunk_size: Minimum chunk size in chars (default: 200)
                - max_chunk_size: Maximum chunk size in chars (default: 2000)
                - batch_size: Batch size for embeddings (default: 32)
        """
        super().__init__(config)
        # Override base class inference
        self.strategy_name = "semantic"

        self.embedding_model = config.get(
            'embedding_model',
            'sentence-transformers/all-MiniLM-L6-v2'
        )
        self.similarity_threshold = config.get('similarity_threshold', 0.8)
        self.min_chunk_size = config.get('min_chunk_size', 200)
        self.max_chunk_size = config.get('max_chunk_size', 2000)
        self.batch_size = config.get('batch_size', 32)

        # Initialize token counter
        self.token_counter = TokenCounter()

        # Initialize embedding model (lazy loading)
        self._embedder = None

        # Validate configuration
        if self.similarity_threshold < 0 or self.similarity_threshold > 1:
            raise ValueError("similarity_threshold must be between 0 and 1")

        if self.min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")

        if self.max_chunk_size <= self.min_chunk_size:
            raise ValueError(
                "max_chunk_size must be greater than min_chunk_size"
            )

    @property
    def embedder(self):
        """Lazy loading of the sentence transformer model."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.embedding_model)
                logger.info(f"Loaded embedding model: {self.embedding_model}")
            except ImportError:
                logger.warning(
                    "sentence-transformers not available, "
                    "falling back to text-based chunking"
                )
                self._embedder = "fallback"
        return self._embedder

    def chunk_document(
        self, document: ProcessedDocument
    ) -> List[DocumentChunk]:
        """Chunk document using semantic similarity analysis.

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

            # Check if we can use semantic analysis
            if self.embedder == "fallback":
                return self._fallback_chunking(document, text)

            # Try semantic chunking with LangChain's SemanticChunker
            try:
                chunks = self._semantic_split_text(text)
            except Exception as e:
                logger.warning(
                    f"Semantic chunking failed: {e}, using fallback"
                )
                return self._fallback_chunking(document, text)

            # Create DocumentChunk objects
            document_chunks = []
            for i, chunk_text in enumerate(chunks):
                chunk = self._create_document_chunk(
                    document, chunk_text, i, text
                )
                document_chunks.append(chunk)

            logger.info(
                f"Created {len(document_chunks)} chunks for document "
                f"{document.document_id} using semantic strategy"
            )

            return document_chunks

        except Exception as e:
            logger.error(f"Semantic chunking failed: {str(e)}")
            raise ChunkingError(
                f"Semantic chunking failed: {str(e)}",
                strategy_name=self.strategy_name,
                document_id=document.document_id
            )

    def _semantic_split_text(self, text: str) -> List[str]:
        """Split text using semantic analysis.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        try:
            # Try to use LangChain's SemanticChunker
            from langchain_experimental.text_splitter import SemanticChunker

            splitter = SemanticChunker(
                embeddings=self._get_langchain_embeddings(),
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=self.similarity_threshold * 100
            )

            chunks = splitter.split_text(text)

            # Filter chunks by size constraints
            filtered_chunks = []
            for chunk in chunks:
                if len(chunk) < self.min_chunk_size:
                    # Merge small chunks with previous if possible
                    if filtered_chunks:
                        filtered_chunks[-1] += "\n\n" + chunk
                    else:
                        filtered_chunks.append(chunk)
                elif len(chunk) > self.max_chunk_size:
                    # Split large chunks
                    sub_chunks = self._split_large_chunk(chunk)
                    filtered_chunks.extend(sub_chunks)
                else:
                    filtered_chunks.append(chunk)

            return [chunk for chunk in filtered_chunks if chunk.strip()]

        except ImportError:
            logger.warning(
                "LangChain experimental SemanticChunker not available"
            )
            return self._manual_semantic_split(text)

    def _get_langchain_embeddings(self):
        """Get LangChain-compatible embeddings wrapper."""
        try:
            # Try the new langchain-huggingface import first
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=self.embedding_model)
        except ImportError:
            try:
                # Fallback to community embeddings
                from langchain_community.embeddings import (
                    SentenceTransformerEmbeddings
                )
                return SentenceTransformerEmbeddings(
                    model_name=self.embedding_model
                )
            except ImportError:
                logger.warning("LangChain embeddings not available")
                raise

    def _manual_semantic_split(self, text: str) -> List[str]:
        """Manual semantic splitting using sentence similarity.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        # Split into sentences
        sentences = self._split_into_sentences(text)

        if len(sentences) <= 1:
            return [text]

        # Calculate sentence embeddings
        try:
            embeddings = self.embedder.encode(
                sentences, batch_size=self.batch_size
            )
        except Exception as e:
            logger.warning(f"Failed to compute embeddings: {e}")
            return self._fallback_sentence_split(sentences)

        # Find semantic breakpoints
        breakpoints = self._find_semantic_breakpoints(embeddings)

        # Create chunks from breakpoints
        chunks = []
        start_idx = 0

        for breakpoint in breakpoints:
            chunk_sentences = sentences[start_idx:breakpoint]
            chunk_text = ' '.join(chunk_sentences)

            if chunk_text.strip():
                chunks.append(chunk_text)

            start_idx = breakpoint

        # Add final chunk
        if start_idx < len(sentences):
            final_chunk = ' '.join(sentences[start_idx:])
            if final_chunk.strip():
                chunks.append(final_chunk)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        import re

        # Simple sentence splitting (could be improved with nltk/spacy)
        sentences = re.split(r'[.!?]+\s+', text)
        return [sent.strip() for sent in sentences if sent.strip()]

    def _find_semantic_breakpoints(self, embeddings) -> List[int]:
        """Find semantic breakpoints based on embedding similarity.

        Args:
            embeddings: Sentence embeddings array

        Returns:
            List of breakpoint indices
        """
        from sklearn.metrics.pairwise import cosine_similarity

        breakpoints = []

        for i in range(1, len(embeddings)):
            # Calculate similarity between consecutive sentences
            sim = cosine_similarity(
                [embeddings[i-1]], [embeddings[i]]
            )[0][0]

            # If similarity drops below threshold, it's a breakpoint
            if sim < self.similarity_threshold:
                breakpoints.append(i)

        return breakpoints

    def _fallback_sentence_split(self, sentences: List[str]) -> List[str]:
        """Fallback sentence-based splitting without embeddings.

        Args:
            sentences: List of sentences

        Returns:
            List of chunks
        """
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            if (current_size + sentence_size > self.max_chunk_size and
                    current_chunk):

                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size

        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)

        return chunks

    def _split_large_chunk(self, chunk: str) -> List[str]:
        """Split a chunk that's too large.

        Args:
            chunk: Large chunk to split

        Returns:
            List of smaller chunks
        """
        sentences = self._split_into_sentences(chunk)
        return self._fallback_sentence_split(sentences)

    def _fallback_chunking(
        self, document: ProcessedDocument, text: str
    ) -> List[DocumentChunk]:
        """Fallback to structure-based chunking.

        Args:
            document: The document to chunk
            text: Text content

        Returns:
            List of document chunks
        """
        from .sliding_langchain import SlidingLangChainChunker

        # Use LangChain sliding window as fallback
        fallback_config = {
            'chunk_size': min(self.max_chunk_size, 1000),
            'chunk_overlap': 200,
            'separators': ["\n\n", "\n", ". ", " ", ""]
        }

        fallback_chunker = SlidingLangChainChunker(fallback_config)
        chunks = fallback_chunker.chunk_document(document)

        # Update strategy name for consistency
        for chunk in chunks:
            chunk.strategy_name = self.strategy_name
            chunk.chunk_id = self.generate_chunk_id(
                document.document_id, chunks.index(chunk)
            )

        return chunks

    def _create_document_chunk(
        self,
        document: ProcessedDocument,
        chunk_text: str,
        chunk_index: int,
        original_text: str
    ) -> DocumentChunk:
        """Create a DocumentChunk from text and metadata.

        Args:
            document: Source document
            chunk_text: Text content of the chunk
            chunk_index: Index of this chunk in the document
            original_text: Original document text for position calculation

        Returns:
            DocumentChunk object
        """
        # Find approximate position in original text
        start_pos = original_text.find(chunk_text[:100])
        if start_pos == -1:
            start_pos = chunk_index * 1000  # Rough estimate

        end_pos = start_pos + len(chunk_text)

        # Count tokens in the chunk
        token_count = self.token_counter.count_tokens(chunk_text)

        return DocumentChunk(
            chunk_id=self.generate_chunk_id(document.document_id, chunk_index),
            document_id=document.document_id,
            strategy_name=self.strategy_name,
            content=chunk_text,
            start_position=start_pos,
            end_position=min(end_pos, len(original_text)),
            token_count=token_count,
            metadata={
                'title': document.title,
                'authors': document.authors,
                'chunk_index': chunk_index,
                'total_chunks': 0,  # Will be updated by pipeline
                'chunk_size': len(chunk_text),  # Actual size of this chunk
                'embedding_model': self.embedding_model,
                'similarity_threshold': self.similarity_threshold,
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
            'embedding_model': self.embedding_model,
            'similarity_threshold': self.similarity_threshold,
            'min_chunk_size': self.min_chunk_size,
            'max_chunk_size': self.max_chunk_size,
            'batch_size': self.batch_size
        }
