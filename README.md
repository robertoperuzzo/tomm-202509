# Semantic Search RAG Demo Setup Guide

This implementation provides a complete semantic search system using Cheshire Cat AI and Typesense for RAG (Retrieval-Augmented Generation) with PDF processing capabilities.

## Requirements

### Python Dependencies
```
# Core dependencies
typesense==0.15.0
PyPDF2==3.0.1
sentence-transformers==2.2.2
numpy==1.24.3
cheshire-cat-api==1.3.1
pydantic==1.10.7
PyYAML==6.0

# For token-based chunking (recommended)
transformers==4.30.0
torch>=1.9.0

# Additional utilities
python-dotenv==1.0.0
asyncio-mqtt==0.13.0
```

### System Requirements
- Python 3.8+
- Docker and Docker Compose
- Typesense and Cheshire Cat services (provided via `compose.yml`)

## Installation Steps

### 1. Install Python Dependencies
```bash
pip install typesense PyPDF2 sentence-transformers numpy cheshire-cat-api pydantic PyYAML python-dotenv transformers torch
```

### 2. Start services (Typesense + Cheshire Cat) with Docker Compose
Using the provided `compose.yml` at the project root:
```bash
docker compose -f compose.yml up -d
```

Optional health checks:
```bash
curl -s http://localhost:8108/health
curl -s http://localhost:1865/ | head
```

Cheshire Cat is included in `compose.yml` and will be available at `http://localhost:1865`.

### 3. Configuration

Create a configuration file using the built-in template:
```bash
python semantic_search_rag.py --create-config
```

This creates `config.yaml`:
```yaml
typesense_host: localhost
typesense_port: 8108
typesense_protocol: http
typesense_api_key: your-api-key-here
typesense_collection: documents
cheshire_cat_host: localhost
cheshire_cat_port: 1865
cheshire_cat_user_id: user
embedding_model: sentence-transformers/all-MiniLM-L6-v2

# Chunking strategy configuration - KEY FOR EXPERIMENTATION
chunk_strategy: token_based  # token_based, word_based, sentence_based
chunk_size: 512
chunk_overlap: 128
min_chunk_size: 32

# Alternative: use research-based predefined configs
# chunk_config_name: fact_based_medium  # See available options below

max_results: 5
similarity_threshold: 0.7
```

### Predefined Chunk Configurations

The system includes research-based chunk configurations optimized for different use cases:

```bash
# List all available configurations
python semantic_search_rag.py --list-chunk-configs
```

Available configurations:
- `fact_based_small`: 64 tokens - optimal for fact-based queries
- `fact_based_medium`: 128 tokens - good balance for fact retrieval
- `context_small`: 256 tokens - small context preservation
- `context_medium`: 512 tokens - medium context for broader queries
- `context_large`: 1024 tokens - large context for complex topics
- `sentence_based`: 5 sentences per chunk with 1 sentence overlap
- `paragraph_based`: ~10 sentences for paragraph-like chunks

Alternatively, use environment variables in a `.env` file:
```env
TYPESENSE_HOST=localhost
TYPESENSE_PORT=8108
TYPESENSE_API_KEY=your-api-key-here
CHESHIRE_CAT_HOST=localhost
CHESHIRE_CAT_PORT=1865
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=500
MAX_SEARCH_RESULTS=5
```

## Usage

### Chunk Size Experimentation (NEW!)

**1. List available chunk configurations:**
```bash
python semantic_search_rag.py --list-chunk-configs
```

**2. Run comprehensive chunk size experiment:**
```bash
python semantic_search_rag.py --pdf-dir ./documents --experiment \
  --experiment-queries "What are the main topics?" "List specific facts" "Explain the methodology"
```

**3. Test specific chunk configuration:**
```bash
# Use fact-based small chunks (64 tokens)
python semantic_search_rag.py --config config.yaml --pdf-dir ./documents --chunk-config fact_based_small

# Use large context chunks (1024 tokens)
python semantic_search_rag.py --config config.yaml --pdf-dir ./documents --chunk-config context_large
```

### Command Line Interface

1. **Create configuration templates:**
```bash
python semantic_search_rag.py --create-config
# This creates: config.yaml, config_fact_based_64.yaml, config_context_512.yaml, etc.
```

2. **Process PDF documents with specific chunk size:**
```bash
# Using predefined config for fact-based queries
python semantic_search_rag.py --config config_fact_based_128.yaml --pdf-dir /path/to/pdf/directory

# Using custom token-based chunking
python semantic_search_rag.py --config config.yaml --pdf-dir /path/to/pdf/directory
```

3. **Perform semantic search:**
```bash
python semantic_search_rag.py --config config.yaml --query "What is machine learning?"
```

### Programmatic Usage with Chunk Experimentation

```python
import asyncio
from semantic_search_rag import SemanticSearchRAG, PDFExtractor

async def chunk_size_comparison():
    # Test different chunk configurations
    configs = ['fact_based_small', 'fact_based_medium', 'context_medium', 'context_large']

    results = {}
    for config_name in configs:
        # Initialize with specific chunk config
        rag_system = SemanticSearchRAG()
        rag_system.config.chunk_config_name = config_name
        rag_system.config.typesense_collection = f"test_{config_name}"

        # Reinitialize with new config
        rag_system.pdf_extractor = PDFExtractor(config_name=config_name)
        rag_system.typesense_manager.collection_name = f"test_{config_name}"
        rag_system.typesense_manager.create_collection(rag_system.embedding_manager.dimension)

        # Process documents
        rag_system.process_pdfs("./documents")

        # Test queries
        fact_query = "What specific numbers or statistics are mentioned?"
        context_query = "Explain the overall methodology and approach?"

        fact_result = await rag_system.search_and_generate(fact_query)
        context_result = await rag_system.search_and_generate(context_query)

        results[config_name] = {
            'config': PDFExtractor.CHUNK_CONFIGS[config_name],
            'fact_query_results': len(fact_result['context_chunks']),
            'context_query_results': len(context_result['context_chunks']),
            'fact_response_length': len(fact_result['response']),
            'context_response_length': len(context_result['response'])
        }

        print(f"\n=== {config_name.upper()} ===")
        print(f"Config: {PDFExtractor.CHUNK_CONFIGS[config_name]['description']}")
        print(f"Fact query - Found chunks: {len(fact_result['context_chunks'])}")
        print(f"Context query - Found chunks: {len(context_result['context_chunks'])}")

    return results

# Run automated experiment
async def run_experiment():
    rag_system = SemanticSearchRAG()

    # Run comprehensive experiment
    experiment_results = rag_system.run_chunk_size_experiment(
        "./documents",
        [
            "What are the specific facts and numbers mentioned?",
            "Explain the main concepts and their relationships?",
            "What methodology was used in this research?",
            "List all mentioned authors or sources?"
        ]
    )

    print("Experiment completed! Results saved to chunk_experiment_results.json")

    # Analyze results
    for config_name, config_results in experiment_results['results'].items():
        config_info = config_results['config']
        print(f"\n{config_name} ({config_info['description']}):")
        print(f"  Total chunks created: {config_results['total_chunks']}")
        print(f"  Average chunk length: {config_results['avg_chunk_length']:.0f} characters")

        for query, query_results in config_results['query_results'].items():
            print(f"  '{query[:30]}...': {query_results['num_results']} results (avg score: {query_results['avg_score']:.3f})")

if __name__ == "__main__":
    asyncio.run(run_experiment())
```

## Configuration Parameters

### Chunking Strategy Settings (NEW!)
- `chunk_strategy`: Chunking method (`token_based`, `word_based`, `sentence_based`)
- `chunk_size`: Size per chunk (tokens/words/sentences based on strategy)
- `chunk_overlap`: Overlap between consecutive chunks (same units)
- `min_chunk_size`: Minimum chunk size threshold
- `chunk_config_name`: Use predefined research-based configuration

**Research-Based Recommendations:**
- **64-128 tokens**: Best for fact-based queries, specific information retrieval
- **512-1024 tokens**: Better for context-heavy queries, conceptual understanding
- **Sentence-based**: Good for maintaining semantic coherence
- **Token-based**: Most precise, requires transformers library

### Typesense Settings
- `typesense_host`: Typesense server host
- `typesense_port`: Typesense server port
- `typesense_api_key`: API key for Typesense authentication
- `typesense_collection`: Collection name for documents

### Cheshire Cat Settings
- `cheshire_cat_host`: Cheshire Cat server host
- `cheshire_cat_port`: Cheshire Cat server port
- `cheshire_cat_user_id`: User ID for chat sessions

### Processing Settings
- `embedding_model`: Sentence transformer model name
- `max_results`: Maximum search results to return
- `similarity_threshold`: Minimum similarity score threshold

## Features

### Advanced Chunking Strategies
- **Token-based chunking**: Precise control using transformer tokenizers
- **Word-based chunking**: Simple word count-based splitting
- **Sentence-based chunking**: Maintains semantic boundaries
- **Research-based presets**: Optimized configurations for different query types
- **Overlap control**: Configurable overlap to preserve context across chunks

### Chunk Size Experimentation Framework
- **Automated experimentation**: Test multiple chunk sizes with same documents
- **Performance comparison**: Compare precision/recall across configurations
- **Query-specific optimization**: Different chunk sizes for different query types
- **Results logging**: Detailed JSON output for analysis
- **Predefined configurations**: Research-backed chunk size recommendations

### PDF Processing
- Extracts text from PDF files page by page
- Creates overlapping text chunks for better context
- Handles multiple PDFs in batch processing
- Preserves document metadata (source, page numbers)

### Semantic Search
- Uses sentence transformers for text embeddings
- Vector similarity search with Typesense
- Configurable similarity thresholds
- Returns ranked results with scores

### RAG Integration
- Integrates with Cheshire Cat AI for response generation
- Provides relevant context from search results
- Generates comprehensive answers based on retrieved documents
- Handles cases where context is insufficient

## Chunk Size Research Implementation

Based on the research notes you provided, this implementation includes:

### Precision vs. Recall Trade-offs
- **Small chunks (64-128 tokens)**: Higher precision for fact-based queries
  - Better for: "What is the exact number?", "Who said this quote?"
  - Use: `fact_based_small` or `fact_based_medium` configs

- **Large chunks (512-1024 tokens)**: Higher recall for context queries
  - Better for: "Explain the methodology", "What are the main concepts?"
  - Use: `context_medium` or `context_large` configs

### Experimental Validation
```bash
# Test the research hypothesis with your documents
python semantic_search_rag.py --pdf-dir ./docs --experiment \
  --experiment-queries \
    "What specific statistics are mentioned?" \
    "Explain the overall approach and methodology?" \
    "Who are the key authors cited?" \
    "What are the main theoretical frameworks discussed?"

# Results will show:
# - fact_based_small: High precision for specific facts
# - context_large: Better recall for conceptual queries
# - Quantitative metrics: chunk counts, similarity scores, response lengths
```

### Query-Adaptive Chunking
The system allows you to use different chunk sizes for different query types:

```python
# For fact-based queries
fact_rag = SemanticSearchRAG('config_fact_based_64.yaml')
fact_result = await fact_rag.search_and_generate("What is the exact methodology used?")

# For context-heavy queries
context_rag = SemanticSearchRAG('config_context_1024.yaml')
context_result = await context_rag.search_and_generate("Explain the theoretical framework and its implications?")
```

## Architecture Overview

```
PDFs → PDF Extractor → Text Chunks → Embedding Manager → Typesense (Vector DB)
                                                              ↓
User Query → Embedding Manager → Semantic Search ← Typesense
                    ↓
            Context + Query → Cheshire Cat AI → Generated Response
```

## Troubleshooting

### Common Issues

1. **Cheshire Cat Connection Error**
   - Ensure Cheshire Cat is running on the correct port
   - Check if the API endpoint is accessible

2. **Typesense Connection Error**
   - Verify Typesense server is running
   - Check API key configuration
   - Ensure proper network connectivity

3. **Embedding Model Download**
   - First run may take time to download the embedding model
   - Ensure internet connectivity for model download

4. **PDF Processing Errors**
   - Check if PDF files are not corrupted
   - Ensure PDFs contain extractable text (not just images)

### Performance Optimization

- Use GPU-enabled sentence transformers for faster embedding generation
- Adjust chunk size based on your document types
- Configure Typesense memory settings for large document collections
- Consider using lighter embedding models for faster inference

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

Please ensure compliance with the licenses of all dependencies:
- Typesense: Apache 2.0
- Cheshire Cat AI: MIT
- Sentence Transformers: Apache 2.0