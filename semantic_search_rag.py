import os
import json
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
import asyncio
from dataclasses import dataclass
from datetime import datetime

# Core dependencies
import typesense
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
from cheshire_cat_api import CheshireCatAPI
from cheshire_cat_api.models import Message

# Configuration management
import yaml
from pydantic import BaseSettings, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document"""
    id: str
    text: str
    source: str
    page_number: int
    chunk_index: int
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class Config(BaseSettings):
    """Configuration settings for the semantic search system"""
    
    # Typesense settings
    typesense_host: str = Field(default="localhost", env="TYPESENSE_HOST")
    typesense_port: int = Field(default=8108, env="TYPESENSE_PORT")
    typesense_protocol: str = Field(default="http", env="TYPESENSE_PROTOCOL")
    typesense_api_key: str = Field(default="xyz", env="TYPESENSE_API_KEY")
    typesense_collection: str = Field(default="documents", env="TYPESENSE_COLLECTION")
    
    # Cheshire Cat settings
    cheshire_cat_host: str = Field(default="localhost", env="CHESHIRE_CAT_HOST")
    cheshire_cat_port: int = Field(default=1865, env="CHESHIRE_CAT_PORT")
    cheshire_cat_user_id: str = Field(default="user", env="CHESHIRE_CAT_USER_ID")
    
    # Embedding model settings
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    
    # Chunking strategy settings - Key for experimentation
    chunk_strategy: str = Field(default="token_based", env="CHUNK_STRATEGY")  # token_based, word_based, sentence_based
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")  # tokens for token_based, words for word_based
    chunk_overlap: int = Field(default=64, env="CHUNK_OVERLAP")  # overlap in same units as chunk_size
    min_chunk_size: int = Field(default=32, env="MIN_CHUNK_SIZE")  # minimum chunk size threshold
    
    # Pre-defined chunk configurations for easy testing
    chunk_config_name: Optional[str] = Field(default=None, env="CHUNK_CONFIG_NAME")
    
    # Search settings
    max_results: int = Field(default=5, env="MAX_SEARCH_RESULTS")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    
    class Config:
        env_file = ".env"

class PDFExtractor:
    """Handles PDF text extraction and chunking with multiple strategies"""
    
    # Predefined chunk configurations based on research
    CHUNK_CONFIGS = {
        "fact_based_small": {"strategy": "token_based", "size": 64, "overlap": 16, "description": "64 tokens - optimal for fact-based queries"},
        "fact_based_medium": {"strategy": "token_based", "size": 128, "overlap": 32, "description": "128 tokens - good balance for fact retrieval"},
        "context_small": {"strategy": "token_based", "size": 256, "overlap": 64, "description": "256 tokens - small context preservation"},
        "context_medium": {"strategy": "token_based", "size": 512, "overlap": 128, "description": "512 tokens - medium context for broader queries"},
        "context_large": {"strategy": "token_based", "size": 1024, "overlap": 256, "description": "1024 tokens - large context for complex topics"},
        "sentence_based": {"strategy": "sentence_based", "size": 5, "overlap": 1, "description": "5 sentences per chunk with 1 sentence overlap"},
        "paragraph_based": {"strategy": "sentence_based", "size": 10, "overlap": 2, "description": "~10 sentences for paragraph-like chunks"}
    }
    
    def __init__(self, chunk_strategy: str = "token_based", chunk_size: int = 512, 
                 chunk_overlap: int = 128, min_chunk_size: int = 32, config_name: str = None):
        
        # Use predefined config if specified
        if config_name and config_name in self.CHUNK_CONFIGS:
            config = self.CHUNK_CONFIGS[config_name]
            self.chunk_strategy = config["strategy"]
            self.chunk_size = config["size"]
            self.chunk_overlap = config["overlap"]
            logger.info(f"Using predefined chunk config '{config_name}': {config['description']}")
        else:
            self.chunk_strategy = chunk_strategy
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            
        self.min_chunk_size = min_chunk_size
        
        # Import tokenizer for token-based chunking
        if self.chunk_strategy == "token_based":
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                logger.info("Loaded tokenizer for token-based chunking")
            except ImportError:
                logger.warning("transformers not installed, falling back to word-based approximation")
                self.chunk_strategy = "word_based"
                # Rough approximation: 1 token ≈ 0.75 words
                self.chunk_size = int(self.chunk_size * 0.75)
                self.chunk_overlap = int(self.chunk_overlap * 0.75)
        
        logger.info(f"Chunking strategy: {self.chunk_strategy}, size: {self.chunk_size}, overlap: {self.chunk_overlap}")
    
    @classmethod
    def list_chunk_configs(cls):
        """List all available predefined chunk configurations"""
        print("Available chunk configurations:")
        for name, config in cls.CHUNK_CONFIGS.items():
            print(f"  {name}: {config['description']}")
            print(f"    Strategy: {config['strategy']}, Size: {config['size']}, Overlap: {config['overlap']}")
        return cls.CHUNK_CONFIGS
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[DocumentChunk]:
        """Extract text from PDF and create chunks"""
        chunks = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                pdf_filename = Path(pdf_path).name
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        page_chunks = self._create_chunks(
                            text, pdf_filename, page_num + 1
                        )
                        chunks.extend(page_chunks)
                        
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            
        return chunks
    
    def _create_chunks(self, text: str, source: str, page_number: int) -> List[DocumentChunk]:
        """Create chunks using the specified strategy"""
        if self.chunk_strategy == "token_based":
            return self._create_token_based_chunks(text, source, page_number)
        elif self.chunk_strategy == "word_based":
            return self._create_word_based_chunks(text, source, page_number)
        elif self.chunk_strategy == "sentence_based":
            return self._create_sentence_based_chunks(text, source, page_number)
        else:
            logger.warning(f"Unknown chunk strategy: {self.chunk_strategy}, falling back to word_based")
            return self._create_word_based_chunks(text, source, page_number)
    
    def _create_token_based_chunks(self, text: str, source: str, page_number: int) -> List[DocumentChunk]:
        """Create chunks based on token count"""
        chunks = []
        
        try:
            # Tokenize the entire text
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            chunk_idx = 0
            for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
                chunk_tokens = tokens[i:i + self.chunk_size]
                
                if len(chunk_tokens) < self.min_chunk_size:
                    break
                
                # Decode tokens back to text
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                
                if chunk_text.strip():
                    chunk_id = f"{source}_{page_number}_{chunk_idx}"
                    
                    chunk = DocumentChunk(
                        id=chunk_id,
                        text=chunk_text,
                        source=source,
                        page_number=page_number,
                        chunk_index=chunk_idx,
                        metadata={
                            'created_at': datetime.now().isoformat(),
                            'token_count': len(chunk_tokens),
                            'chunk_strategy': self.chunk_strategy,
                            'chunk_size_config': self.chunk_size
                        }
                    )
                    chunks.append(chunk)
                    chunk_idx += 1
                    
        except Exception as e:
            logger.error(f"Error in token-based chunking: {e}, falling back to word-based")
            return self._create_word_based_chunks(text, source, page_number)
            
        return chunks
    
    def _create_word_based_chunks(self, text: str, source: str, page_number: int) -> List[DocumentChunk]:
        """Create overlapping word-based chunks"""
        chunks = []
        words = text.split()
        
        chunk_idx = 0
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            
            if len(chunk_words) < self.min_chunk_size:
                break
                
            chunk_text = ' '.join(chunk_words)
            
            if chunk_text.strip():
                chunk_id = f"{source}_{page_number}_{chunk_idx}"
                
                chunk = DocumentChunk(
                    id=chunk_id,
                    text=chunk_text,
                    source=source,
                    page_number=page_number,
                    chunk_index=chunk_idx,
                    metadata={
                        'created_at': datetime.now().isoformat(),
                        'word_count': len(chunk_words),
                        'chunk_strategy': self.chunk_strategy,
                        'chunk_size_config': self.chunk_size
                    }
                )
                chunks.append(chunk)
                chunk_idx += 1
                
        return chunks
    
    def _create_sentence_based_chunks(self, text: str, source: str, page_number: int) -> List[DocumentChunk]:
        """Create chunks based on sentence boundaries"""
        chunks = []
        
        # Simple sentence splitting (can be enhanced with nltk or spacy)
        import re
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunk_idx = 0
        for i in range(0, len(sentences), self.chunk_size - self.chunk_overlap):
            chunk_sentences = sentences[i:i + self.chunk_size]
            
            if len(chunk_sentences) < max(1, self.min_chunk_size // 20):  # Adjust min for sentences
                break
                
            chunk_text = '. '.join(chunk_sentences)
            if not chunk_text.endswith('.'):
                chunk_text += '.'
            
            if chunk_text.strip():
                chunk_id = f"{source}_{page_number}_{chunk_idx}"
                
                chunk = DocumentChunk(
                    id=chunk_id,
                    text=chunk_text,
                    source=source,
                    page_number=page_number,
                    chunk_index=chunk_idx,
                    metadata={
                        'created_at': datetime.now().isoformat(),
                        'sentence_count': len(chunk_sentences),
                        'chunk_strategy': self.chunk_strategy,
                        'chunk_size_config': self.chunk_size
                    }
                )
                chunks.append(chunk)
                chunk_idx += 1
                
        return chunks

class EmbeddingManager:
    """Handles text embeddings using sentence transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Loaded embedding model: {model_name} (dimension: {self.dimension})")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Generate embeddings for multiple chunks"""
        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.encode(texts)
        
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i].tolist()
            
        return chunks

class TypesenseManager:
    """Handles Typesense vector database operations"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = typesense.Client({
            'nodes': [{
                'host': config.typesense_host,
                'port': config.typesense_port,
                'protocol': config.typesense_protocol
            }],
            'api_key': config.typesense_api_key,
            'connection_timeout_seconds': 60
        })
        self.collection_name = config.typesense_collection
    
    def create_collection(self, embedding_dimension: int):
        """Create Typesense collection for documents"""
        schema = {
            'name': self.collection_name,
            'fields': [
                {'name': 'id', 'type': 'string'},
                {'name': 'text', 'type': 'string'},
                {'name': 'source', 'type': 'string', 'facet': True},
                {'name': 'page_number', 'type': 'int32', 'facet': True},
                {'name': 'chunk_index', 'type': 'int32'},
                {'name': 'word_count', 'type': 'int32'},
                {'name': 'created_at', 'type': 'string'},
                {'name': 'embedding', 'type': 'float[]', 'num_dim': embedding_dimension}
            ]
        }
        
        try:
            # Delete existing collection if it exists
            self.client.collections[self.collection_name].delete()
            logger.info(f"Deleted existing collection: {self.collection_name}")
        except Exception:
            pass
        
        try:
            self.client.collections.create(schema)
            logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def index_chunks(self, chunks: List[DocumentChunk]):
        """Index document chunks in Typesense"""
        documents = []
        
        for chunk in chunks:
            doc = {
                'id': chunk.id,
                'text': chunk.text,
                'source': chunk.source,
                'page_number': chunk.page_number,
                'chunk_index': chunk.chunk_index,
                'word_count': chunk.metadata.get('word_count', 0),
                'created_at': chunk.metadata.get('created_at', ''),
                'embedding': chunk.embedding
            }
            documents.append(doc)
        
        try:
            # Batch import documents
            result = self.client.collections[self.collection_name].documents.import_(
                documents, {'action': 'upsert'}
            )
            logger.info(f"Indexed {len(documents)} chunks")
            return result
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            raise
    
    def semantic_search(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """Perform semantic search using vector similarity"""
        search_parameters = {
            'q': '*',
            'vector_query': f'embedding:([{",".join(map(str, query_embedding))}], k:{k})',
            'per_page': k
        }
        
        try:
            result = self.client.collections[self.collection_name].documents.search(search_parameters)
            return result['hits']
        except Exception as e:
            logger.error(f"Error performing semantic search: {e}")
            return []

class CheshireCatManager:
    """Handles Cheshire Cat AI integration for RAG"""
    
    def __init__(self, config: Config):
        self.config = config
        self.api = CheshireCatAPI(
            host=config.cheshire_cat_host,
            port=config.cheshire_cat_port
        )
        self.user_id = config.cheshire_cat_user_id
    
    async def generate_response(self, query: str, context: str) -> str:
        """Generate response using Cheshire Cat with RAG context"""
        prompt = f"""Based on the following context, please answer the question:

Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to answer the question, please say so."""

        try:
            message = Message(text=prompt)
            response = await self.api.send_message(message, self.user_id)
            return response.content
        except Exception as e:
            logger.error(f"Error generating response with Cheshire Cat: {e}")
            return "I'm sorry, I couldn't generate a response at this time."

class SemanticSearchRAG:
    """Main class orchestrating the semantic search and RAG system"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            self.config = Config(**config_data)
        else:
            self.config = Config()
        
        # Initialize components
        self.pdf_extractor = PDFExtractor(
            chunk_strategy=self.config.chunk_strategy,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            min_chunk_size=self.config.min_chunk_size,
            config_name=self.config.chunk_config_name
        )
        self.embedding_manager = EmbeddingManager(self.config.embedding_model)
        self.typesense_manager = TypesenseManager(self.config)
        self.cheshire_cat_manager = CheshireCatManager(self.config)
        
        # Create collection
        self.typesense_manager.create_collection(self.embedding_manager.dimension)
    
    def process_pdfs(self, pdf_directory: str):
        """Process all PDFs in a directory"""
        pdf_path = Path(pdf_directory)
        pdf_files = list(pdf_path.glob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        all_chunks = []
        for pdf_file in pdf_files:
            logger.info(f"Processing: {pdf_file.name}")
            chunks = self.pdf_extractor.extract_text_from_pdf(str(pdf_file))
            all_chunks.extend(chunks)
        
        logger.info(f"Extracted {len(all_chunks)} chunks from all PDFs")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        chunks_with_embeddings = self.embedding_manager.embed_chunks(all_chunks)
        
        # Index in Typesense
        logger.info("Indexing in Typesense...")
        self.typesense_manager.index_chunks(chunks_with_embeddings)
        
        logger.info("PDF processing complete!")
    
    async def search_and_generate(self, query: str) -> Dict[str, Any]:
        """Perform semantic search and generate response using RAG"""
        # Generate query embedding
        query_embedding = self.embedding_manager.embed_text(query)
        
        # Perform semantic search
        search_results = self.typesense_manager.semantic_search(
            query_embedding, k=self.config.max_results
        )
        
        # Prepare context from search results
        context_chunks = []
        for hit in search_results:
            document = hit['document']
            score = hit.get('vector_distance', 0)
            
            if score <= (2 - self.config.similarity_threshold):  # Typesense uses distance, not similarity
                context_chunks.append({
                    'text': document['text'],
                    'source': document['source'],
                    'page': document['page_number'],
                    'score': score
                })
        
        # Build context string
        context = "\n\n".join([
            f"[{chunk['source']}, page {chunk['page']}]: {chunk['text']}"
            for chunk in context_chunks
        ])
        
        # Generate response using Cheshire Cat
        response = await self.cheshire_cat_manager.generate_response(query, context)
        
        return {
            'query': query,
            'response': response,
            'context_chunks': context_chunks,
            'total_results': len(search_results)
        }
    
    def save_config_template(self, path: str = "config.yaml"):
        """Save a configuration template with chunk size options"""
        config_template = {
            'typesense_host': 'localhost',
            'typesense_port': 8108,
            'typesense_protocol': 'http',
            'typesense_api_key': 'your-api-key-here',
            'typesense_collection': 'documents',
            'cheshire_cat_host': 'localhost',
            'cheshire_cat_port': 1865,
            'cheshire_cat_user_id': 'user',
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            
            # Chunk configuration options
            'chunk_strategy': 'token_based',  # token_based, word_based, sentence_based
            'chunk_size': 512,
            'chunk_overlap': 128,
            'min_chunk_size': 32,
            
            # Alternative: use predefined configs (uncomment to use)
            # 'chunk_config_name': 'fact_based_small',  # See PDFExtractor.CHUNK_CONFIGS for options
            
            'max_results': 5,
            'similarity_threshold': 0.7
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_template, f, default_flow_style=False)
        
        logger.info(f"Configuration template saved to {path}")
        
        # Also save example configs for different use cases
        self._save_example_configs()
    
    def _save_example_configs(self):
        """Save example configurations for different chunk strategies"""
        configs = {
            'config_fact_based_64.yaml': {
                'chunk_config_name': 'fact_based_small',
                'max_results': 10,
                'similarity_threshold': 0.8,
                'typesense_collection': 'docs_fact_64'
            },
            'config_fact_based_128.yaml': {
                'chunk_config_name': 'fact_based_medium', 
                'max_results': 8,
                'similarity_threshold': 0.75,
                'typesense_collection': 'docs_fact_128'
            },
            'config_context_512.yaml': {
                'chunk_config_name': 'context_medium',
                'max_results': 5,
                'similarity_threshold': 0.7,
                'typesense_collection': 'docs_context_512'
            },
            'config_context_1024.yaml': {
                'chunk_config_name': 'context_large',
                'max_results': 3,
                'similarity_threshold': 0.65,
                'typesense_collection': 'docs_context_1024'
            }
        }
        
        base_config = {
            'typesense_host': 'localhost',
            'typesense_port': 8108,
            'typesense_protocol': 'http',
            'typesense_api_key': 'your-api-key-here',
            'cheshire_cat_host': 'localhost',
            'cheshire_cat_port': 1865,
            'cheshire_cat_user_id': 'user',
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'
        }
        
        for filename, specific_config in configs.items():
            full_config = {**base_config, **specific_config}
            with open(filename, 'w') as f:
                yaml.dump(full_config, f, default_flow_style=False)
        
        logger.info("Example configurations saved for different chunk strategies")
    
    def run_chunk_size_experiment(self, pdf_directory: str, queries: List[str], 
                                 output_file: str = "chunk_experiment_results.json"):
        """Run experiments with different chunk sizes and save results"""
        experiment_configs = [
            'fact_based_small', 'fact_based_medium', 'context_small', 
            'context_medium', 'context_large', 'sentence_based'
        ]
        
        results = {
            'experiment_date': datetime.now().isoformat(),
            'pdf_directory': pdf_directory,
            'queries': queries,
            'results': {}
        }
        
        for config_name in experiment_configs:
            logger.info(f"Running experiment with config: {config_name}")
            
            # Reinitialize with new chunk config
            temp_config = self.config.copy(deep=True)
            temp_config.chunk_config_name = config_name
            temp_config.typesense_collection = f"experiment_{config_name}"
            
            # Create new extractor with this config
            extractor = PDFExtractor(config_name=config_name)
            
            # Process PDFs
            all_chunks = []
            pdf_path = Path(pdf_directory)
            for pdf_file in pdf_path.glob("*.pdf"):
                chunks = extractor.extract_text_from_pdf(str(pdf_file))
                all_chunks.extend(chunks)
            
            # Generate embeddings and index
            chunks_with_embeddings = self.embedding_manager.embed_chunks(all_chunks)
            
            # Create temporary collection
            temp_typesense = TypesenseManager(temp_config)
            temp_typesense.create_collection(self.embedding_manager.dimension)
            temp_typesense.index_chunks(chunks_with_embeddings)
            
            # Test queries
            config_results = {
                'config': PDFExtractor.CHUNK_CONFIGS[config_name],
                'total_chunks': len(all_chunks),
                'avg_chunk_length': np.mean([len(chunk.text) for chunk in all_chunks]),
                'query_results': {}
            }
            
            for query in queries:
                query_embedding = self.embedding_manager.embed_text(query)
                search_results = temp_typesense.semantic_search(query_embedding, k=10)
                
                config_results['query_results'][query] = {
                    'num_results': len(search_results),
                    'avg_score': np.mean([hit.get('vector_distance', 0) for hit in search_results]) if search_results else 0,
                    'top_3_scores': [hit.get('vector_distance', 0) for hit in search_results[:3]]
                }
            
            results['results'][config_name] = config_results
            logger.info(f"Completed experiment for {config_name}")
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Experiment results saved to {output_file}")
        return results

# Example usage and CLI interface
async def main():
    """Main function demonstrating usage and chunk size experimentation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Semantic Search RAG Demo with Chunk Size Experimentation")
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--pdf-dir', type=str, help='Directory containing PDF files')
    parser.add_argument('--query', type=str, help='Search query')
    parser.add_argument('--create-config', action='store_true', help='Create configuration templates')
    parser.add_argument('--list-chunk-configs', action='store_true', help='List available chunk configurations')
    parser.add_argument('--chunk-config', type=str, help='Use predefined chunk configuration')
    parser.add_argument('--experiment', action='store_true', help='Run chunk size experiment')
    parser.add_argument('--experiment-queries', nargs='+', help='Queries for chunk experiment', 
                       default=["What are the main topics?", "List specific facts mentioned", "Explain the methodology used"])
    
    args = parser.parse_args()
    
    # List available chunk configurations
    if args.list_chunk_configs:
        PDFExtractor.list_chunk_configs()
        return
    
    # Create configuration templates
    if args.create_config:
        rag_system = SemanticSearchRAG()
        rag_system.save_config_template()
        return
    
    # Initialize system
    rag_system = SemanticSearchRAG(args.config)
    
    # Run chunk size experiment
    if args.experiment and args.pdf_dir:
        results = rag_system.run_chunk_size_experiment(args.pdf_dir, args.experiment_queries)
        
        print("\n=== CHUNK SIZE EXPERIMENT RESULTS ===")
        for config_name, config_results in results['results'].items():
            print(f"\n{config_name.upper()}:")
            print(f"  Strategy: {config_results['config']['description']}")
            print(f"  Total chunks: {config_results['total_chunks']}")
            print(f"  Avg chunk length: {config_results['avg_chunk_length']:.1f} chars")
            
            for query, query_results in config_results['query_results'].items():
                print(f"  Query '{query}': {query_results['num_results']} results, avg score: {query_results['avg_score']:.3f}")
        
        print(f"\nDetailed results saved to: chunk_experiment_results.json")
        return
    
    # Process PDFs
    if args.pdf_dir:
        print(f"Processing PDFs with chunk config: {args.chunk_config or 'default settings'}")
        if args.chunk_config:
            print(f"Using: {PDFExtractor.CHUNK_CONFIGS.get(args.chunk_config, {}).get('description', 'Unknown config')}")
        rag_system.process_pdfs(args.pdf_dir)
    
    # Perform search and generate response
    if args.query:
        result = await rag_system.search_and_generate(args.query)
        
        print(f"\nQuery: {result['query']}")
        print(f"\nResponse: {result['response']}")
        print(f"\nRelevant sources ({result['total_results']} found):")
        
        for i, chunk in enumerate(result['context_chunks'], 1):
            print(f"{i}. {chunk['source']} (page {chunk['page']}) - Score: {chunk['score']:.3f}")
            print(f"   Preview: {chunk['text'][:150]}...")

if __name__ == "__main__":
    asyncio.run(main())