#!/usr/bin/env python3
"""
Demo script for chunking strategies using real preprocessed data.

This script demonstrates all implemented chunking strategies using actual
preprocessed PDF data from the pipeline, showcasing:
- Fixed-size chunking
- Sliding windows (LangChain)
- Sliding windows (Unstructured)
- Semantic chunking

Usage:
    python scripts/demo_chunking.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chunker.pipeline import ChunkingPipeline
from chunker.models import ProcessedDocument
from chunker.config import get_default_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_preprocessed_document(file_path: Path) -> ProcessedDocument:
    """Load a preprocessed document from JSON file."""
    logger.info(f"Loading preprocessed document: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract document info
    doc_id = data.get('document_id', file_path.stem)
    title = data.get('title', 'Unknown Title')
    content = data.get('full_text', data.get('content', ''))
    
    # Handle different preprocessing formats
    if isinstance(content, list):
        # If content is list of elements (from unstructured)
        text_content = '\n'.join([
            elem.get('text', str(elem)) if isinstance(elem, dict) 
            else str(elem)
            for elem in content
        ])
    else:
        text_content = str(content)
    
    # Create ProcessedDocument
    doc = ProcessedDocument(
        document_id=doc_id,
        title=title,
        authors=[],  # Empty list as default
        abstract='',  # Empty string as default
        full_text=text_content,
        metadata={
            'source_file': str(file_path),
            'preprocessing_method': file_path.parent.name,
            'original_data_keys': list(data.keys())
        }
    )
    
    logger.info(f"Loaded document '{title}' with {len(text_content)} characters")
    return doc

def print_chunking_results(strategy_name: str, results) -> None:
    """Print formatted results for a chunking strategy."""
    print(f"\n{'='*60}")
    print(f"STRATEGY: {strategy_name.upper()}")
    print(f"{'='*60}")
    
    # Handle ChunkingResult object
    if hasattr(results, 'error') and results.error:
        print(f"❌ ERROR: {results.error}")
        return
    
    chunks = results.chunks if hasattr(results, 'chunks') else []
    stats = results.statistics if hasattr(results, 'statistics') else {}
    
    print("📊 STATISTICS:")
    print(f"   • Total chunks: {len(chunks)}")
    
    if hasattr(stats, 'processing_time_seconds'):
        print(f"   • Processing time: {stats.processing_time_seconds:.2f}s")
    if hasattr(stats, 'avg_chunk_size'):
        print(f"   • Avg chunk size: {stats.avg_chunk_size:.0f} chars")
    if hasattr(stats, 'min_chunk_size'):
        print(f"   • Min chunk size: {stats.min_chunk_size} chars")
    if hasattr(stats, 'max_chunk_size'):
        print(f"   • Max chunk size: {stats.max_chunk_size} chars")
    if hasattr(stats, 'memory_usage_mb'):
        print(f"   • Memory usage: {stats.memory_usage_mb:.2f} MB")
    
    print("\n📄 SAMPLE CHUNKS:")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        if hasattr(chunk, 'content'):
            content = chunk.content
        else:
            content = chunk['content']
        
        content_preview = (content[:200] + "..." 
                          if len(content) > 200 else content)
        
        chunk_size = (chunk.token_count if hasattr(chunk, 'token_count')
                     else len(content))
        
        print(f"   [{i+1}] Size: {chunk_size} chars")
        print(f"       Content: {repr(content_preview)}")
        
        if hasattr(chunk, 'metadata'):
            print(f"       Metadata: {chunk.metadata}")
        elif isinstance(chunk, dict) and 'metadata' in chunk:
            print(f"       Metadata: {chunk['metadata']}")
        print()

def demonstrate_chunking_strategies():
    """Main demo function showing all chunking strategies."""
    print("🚀 CHUNKING STRATEGIES DEMONSTRATION")
    print("Using real preprocessed data from the pipeline\n")
    
    # Find available preprocessed files
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    
    # Look for files in different preprocessing methods
    available_files = []
    for method_dir in ['pypdf', 'unstructured', 'marker']:
        method_path = data_dir / method_dir
        if method_path.exists():
            json_files = list(method_path.glob('*.json'))
            if json_files:
                available_files.extend(json_files)
    
    if not available_files:
        print("❌ No preprocessed files found! Please run preprocessing first.")
        return
    
    # Use the most recent file
    latest_file = max(available_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 Using file: {latest_file}")
    
    # Load the document
    try:
        document = load_preprocessed_document(latest_file)
    except Exception as e:
        logger.error(f"Failed to load document: {e}")
        return
    
    # Configure chunking strategies
    strategies_config = {
        'fixed_size': {
            'chunk_size': 512,
            'overlap': 50
        },
        'sliding_langchain': {
            'chunk_size': 1000,
            'chunk_overlap': 100,
            'separators': ['\n\n', '\n', '.', ' ']
        },
        'sliding_unstructured': {
            'max_characters': 800,
            'overlap': 80,
            'strategy': 'by_title'
        },
        'semantic': {
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'similarity_threshold': 0.8,
            'min_chunk_size': 200,
            'max_chunk_size': 1500
        }
    }
    
    # Get base configuration and update with our strategies
    base_config = get_default_config()
    
    # Update with our demo strategies
    base_config.strategy_configs.update(strategies_config)
    
    # Enable all our demo strategies
    base_config.enabled_strategies = list(strategies_config.keys())
    
    print(f"\n🔧 CONFIGURATION:")
    for strategy, config in strategies_config.items():
        print(f"   • {strategy}: {config}")
    
    # Initialize pipeline
    pipeline = ChunkingPipeline(base_config)
    
    # Process document with all strategies
    print(f"\n⚡ PROCESSING DOCUMENT...")
    print(f"Document: '{document.title}'")
    print(f"Content length: {len(document.full_text)} characters")
    
    try:
        results = pipeline.process_document(document)
        
        # Display results for each strategy
        for strategy_name, strategy_results in results.items():
            print_chunking_results(strategy_name, strategy_results)
        
        # Summary comparison
        print(f"\n{'='*60}")
        print("📊 STRATEGY COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        comparison_data = []
        for strategy_name, strategy_results in results.items():
            if not (hasattr(strategy_results, 'error') and 
                   strategy_results.error):
                chunks = (strategy_results.chunks 
                         if hasattr(strategy_results, 'chunks') else [])
                stats = (strategy_results.statistics 
                        if hasattr(strategy_results, 'statistics') else {})
                comparison_data.append({
                    'Strategy': strategy_name,
                    'Chunks': len(chunks),
                    'Time (s)': f"{getattr(stats, 'processing_time_seconds', 0):.2f}",
                    'Avg Size': f"{getattr(stats, 'avg_chunk_size', 0):.0f}",
                    'Memory (MB)': f"{getattr(stats, 'memory_usage_mb', 0):.2f}"
                })
        
        if comparison_data:
            # Print table header
            print(f"{'Strategy':<20} {'Chunks':<8} {'Time (s)':<10} {'Avg Size':<10} {'Memory (MB)':<12}")
            print("-" * 70)
            
            # Print table rows
            for row in comparison_data:
                print(f"{row['Strategy']:<20} {row['Chunks']:<8} {row['Time (s)']:<10} {row['Avg Size']:<10} {row['Memory (MB)']:<12}")
        
        # Save results
        output_dir = Path(__file__).parent.parent / "data" / "chunks"
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"chunking_demo_results_{document.document_id}.json"
        
        # Prepare results for JSON serialization
        serializable_results = {}
        for strategy_name, strategy_results in results.items():
            if hasattr(strategy_results, 'error') and strategy_results.error:
                serializable_results[strategy_name] = {
                    'chunks': [],
                    'statistics': {},
                    'error': strategy_results.error
                }
            else:
                # Convert chunks to dictionaries
                chunks_data = []
                for chunk in strategy_results.chunks:
                    if hasattr(chunk, 'to_dict'):
                        chunks_data.append(chunk.to_dict())
                    else:
                        chunks_data.append({
                            'chunk_id': chunk.chunk_id,
                            'content': chunk.content,
                            'token_count': chunk.token_count,
                            'metadata': chunk.metadata
                        })
                
                # Convert statistics to dictionary
                stats_data = {}
                if hasattr(strategy_results.statistics, '__dict__'):
                    stats_data = strategy_results.statistics.__dict__
                
                serializable_results[strategy_name] = {
                    'chunks': chunks_data,
                    'statistics': stats_data,
                    'error': None
                }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'document_info': {
                    'document_id': document.document_id,
                    'title': document.title,
                    'content_length': len(document.full_text),
                    'source_file': str(latest_file)
                },
                'strategies_config': strategies_config,
                'results': serializable_results
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point."""
    try:
        demonstrate_chunking_strategies()
        print(f"\n✅ Demo completed successfully!")
    except KeyboardInterrupt:
        print(f"\n⏹️  Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
