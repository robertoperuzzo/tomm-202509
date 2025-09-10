#!/usr/bin/env python3
"""
Demo script for chunking strategies using real preprocessed data.

This script processes all available preprocessed documents with all chunking
strategies, generating separate output files for each strategy-parser-document
combination and displaying comprehensive performance metrics.

Features:
- Individual files per strategy-parser-document combination
- Performance monitoring (CPU, Memory, GPU)
- Comprehensive metrics display
- CLI filtering options

Usage:
    python scripts/demo_chunking.py
    python scripts/demo_chunking.py --strategies=semantic,fixed_size
    python scripts/demo_chunking.py --methods=pypdf,unstructured
    python scripts/demo_chunking.py --no-metrics
"""

from src.chunker.config import get_default_config
from src.chunker.models import ProcessedDocument
from src.chunker.pipeline import ChunkingPipeline
import argparse
import json
import logging
import sys
import time
import traceback
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now we can import chunker modules

# Try to import GPU monitoring libraries
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor system performance metrics during processing."""

    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        self.start_cpu_percent = None
        self.start_gpu_util = None
        self.start_gpu_memory = None

    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.start_cpu_percent = self.process.cpu_percent()

        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    self.start_gpu_util = gpu.load * 100
                    self.start_gpu_memory = gpu.memoryUsed
                else:
                    self.start_gpu_util = 0.0
                    self.start_gpu_memory = 0.0
            except Exception:
                self.start_gpu_util = 0.0
                self.start_gpu_memory = 0.0
        else:
            self.start_gpu_util = 0.0
            self.start_gpu_memory = 0.0

    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        cpu_percent = self.process.cpu_percent()

        gpu_util = 0.0
        gpu_memory = 0.0

        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_util = gpu.load * 100
                    gpu_memory = gpu.memoryUsed
            except Exception:
                pass

        return {
            'processing_time': end_time - self.start_time if self.start_time else 0,
            'memory_usage': max(end_memory, self.start_memory or 0),
            'cpu_usage_percent': cpu_percent,
            'gpu_usage_percent': gpu_util,
            'gpu_memory_usage': gpu_memory
        }


def validate_json(file_path: Path) -> Dict[str, Any]:
    """Validate JSON file and load its content without modifying it."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully validated JSON file: {file_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
        logger.error(f"Error at line {e.lineno}, column {e.colno}")
        raise ValueError(f"Invalid JSON format in {file_path}: {e}")
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        raise


def load_preprocessed_document(file_path: Path) -> ProcessedDocument:
    """Load a preprocessed document from JSON file with error recovery."""
    logger.info(f"Loading preprocessed document: {file_path}")

    try:
        data = validate_json(file_path)

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

        # Validate we have meaningful content
        if len(text_content.strip()) < 10:
            raise ValueError(
                f"Document content too short: {len(text_content)} characters")

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

        logger.info(
            f"Successfully loaded document '{title}' "
            f"with {len(text_content)} characters")
        return doc

    except Exception as e:
        logger.error(f"Failed to load document from {file_path}: {e}")
        raise


def find_available_files(data_dir: Path,
                         methods_filter: Optional[List[str]] = None) -> List[Path]:
    """Find all available preprocessed files."""
    available_files = []

    # Look for files in different preprocessing methods
    for method_dir in ['pypdf', 'unstructured', 'marker']:
        if methods_filter and method_dir not in methods_filter:
            continue

        method_path = data_dir / method_dir
        if method_path.exists():
            for file_path in method_path.glob("*.json"):
                available_files.append(file_path)

    return sorted(available_files, key=lambda x: x.stat().st_mtime, reverse=True)


def process_single_combination(document: ProcessedDocument,
                               strategy_name: str,
                               strategy_config: Dict[str, Any],
                               output_dir: Path) -> Dict[str, Any]:
    """Process a single strategy-parser-document combination."""

    # Create pipeline for single strategy
    config = get_default_config()
    config.strategy_configs[strategy_name] = strategy_config
    config.enabled_strategies = [strategy_name]

    pipeline = ChunkingPipeline(config)

    # Start performance monitoring
    monitor = PerformanceMonitor()
    monitor.start_monitoring()

    try:
        # Process document
        results = pipeline.process_document(document)
        result = results.get(strategy_name)

        if not result:
            raise ValueError(f"No results for strategy {strategy_name}")

        # Get performance metrics
        metrics = monitor.get_metrics()

        # Calculate chunk statistics
        chunks = result.chunks if hasattr(result, 'chunks') else []
        chunk_sizes = [len(chunk.content) for chunk in chunks]

        statistics = {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
            'min_chunk_size': min(chunk_sizes) if chunk_sizes else 0,
            'max_chunk_size': max(chunk_sizes) if chunk_sizes else 0
        }

        # Prepare output data
        output_data = {
            'document_info': {
                'document_id': document.document_id,
                'title': document.title,
                'content_length': len(document.full_text),
                'source_file': document.metadata.get('source_file', ''),
                'preprocessing_method': document.metadata.get('preprocessing_method', '')
            },
            'strategy_config': {
                'strategy_name': strategy_name,
                'parameters': strategy_config
            },
            'results': {
                'chunks': [
                    {
                        'content': chunk.content,
                        'metadata': chunk.metadata,
                        'start_char': getattr(chunk, 'start_char', None),
                        'end_char': getattr(chunk, 'end_char', None)
                    }
                    for chunk in chunks
                ],
                'statistics': statistics,
                'error': result.error if hasattr(result, 'error') else None
            },
            'processing_metadata': {
                'timestamp': datetime.now().isoformat(),
                'processing_time': metrics['processing_time'],
                'memory_usage': metrics['memory_usage'],
                'cpu_usage_percent': metrics['cpu_usage_percent'],
                'gpu_usage_percent': metrics['gpu_usage_percent'],
                'gpu_memory_usage': metrics['gpu_memory_usage']
            }
        }

        # Save individual file
        preprocessing_method = document.metadata.get(
            'preprocessing_method', 'unknown')
        filename = f"{document.document_id}_{preprocessing_method}_{strategy_name}.json"
        output_file = output_dir / filename

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved results to {output_file}")

        # Return performance data for summary
        return {
            'document_id': document.document_id,
            'preprocessing_method': preprocessing_method,
            'strategy_name': strategy_name,
            'filename': filename,
            'metrics': metrics,
            'statistics': statistics,
            'error': result.error if hasattr(result, 'error') else None
        }

    except Exception as e:
        logger.error(
            f"Error processing {strategy_name} on {document.document_id}: {e}")
        error_metrics = monitor.get_metrics()
        return {
            'document_id': document.document_id,
            'preprocessing_method': document.metadata.get('preprocessing_method', 'unknown'),
            'strategy_name': strategy_name,
            'filename': None,
            'metrics': error_metrics,
            'statistics': {'total_chunks': 0, 'avg_chunk_size': 0, 'min_chunk_size': 0, 'max_chunk_size': 0},
            'error': str(e)
        }


def display_performance_summary(results: List[Dict[str, Any]], show_metrics: bool = True):
    """Display comprehensive performance summary."""
    if not show_metrics or not results:
        return

    print("\n" + "="*100)
    print("                           PERFORMANCE SUMMARY")
    print("="*100)

    # Group by document
    documents = {}
    for result in results:
        doc_id = result['document_id']
        if doc_id not in documents:
            documents[doc_id] = []
        documents[doc_id].append(result)

    total_chunks = 0
    total_time = 0
    peak_ram = 0
    peak_gpu = 0
    total_combinations = 0

    for doc_id, doc_results in documents.items():
        print(f"\nDocument: {doc_id}")
        print("="*100)

        # Header
        header = (
            f"{'Strategy/Parser':<15} {'RAM':<8} {'CPU %':<8} {'GPU %':<8} "
            f"{'GPU Mem':<10} {'Time (s)':<10} {'Chunks':<8} {'Avg Size':<10} {'Min/Max':<15}"
        )
        print(header)
        print("-" * len(header))

        # Results for this document
        for result in doc_results:
            if result['error']:
                strategy_parser = f"{result['preprocessing_method']}/{result['strategy_name']}"
                print(f"{strategy_parser:<15} {'ERROR':<8} {'ERROR':<8} {'ERROR':<8} "
                      f"{'ERROR':<10} {'ERROR':<10} {'ERROR':<8} {'ERROR':<10} {'ERROR':<15}")
                continue

            metrics = result['metrics']
            stats = result['statistics']

            strategy_parser = f"{result['preprocessing_method']}/{result['strategy_name']}"
            ram_mb = f"{metrics['memory_usage']:.1f}"
            cpu_pct = f"{metrics['cpu_usage_percent']:.1f}"
            gpu_pct = f"{metrics['gpu_usage_percent']:.1f}"
            gpu_mem = f"{metrics['gpu_memory_usage']:.0f}" if metrics['gpu_memory_usage'] > 0 else "0"
            time_s = f"{metrics['processing_time']:.2f}"
            chunks = f"{stats['total_chunks']}"
            avg_size = f"{stats['avg_chunk_size']:.0f}" if stats['avg_chunk_size'] > 0 else "0"
            min_max = f"{stats['min_chunk_size']}/{stats['max_chunk_size']}" if stats['min_chunk_size'] > 0 else "0/0"

            print(f"{strategy_parser:<15} {ram_mb:<8} {cpu_pct:<8} {gpu_pct:<8} "
                  f"{gpu_mem:<10} {time_s:<10} {chunks:<8} {avg_size:<10} {min_max:<15}")

            # Update totals
            total_chunks += stats['total_chunks']
            total_time += metrics['processing_time']
            peak_ram = max(peak_ram, metrics['memory_usage'])
            peak_gpu = max(peak_gpu, metrics['gpu_memory_usage'])
            total_combinations += 1

    # Overall summary
    avg_chunk_size = total_chunks / total_combinations if total_combinations > 0 else 0
    print("\n" + "="*100)
    print(f"TOTALS: {total_chunks} chunks | {total_time:.2f}s total | "
          f"{peak_ram:.1f}MB peak RAM | {peak_gpu:.0f}MB peak GPU | "
          f"{avg_chunk_size:.0f} avg chars/chunk")
    print("="*100)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Demo script for chunking strategies with performance monitoring"
    )

    parser.add_argument(
        '--strategies',
        type=str,
        help='Comma-separated list of strategies to run (e.g., semantic,fixed_size)'
    )

    parser.add_argument(
        '--methods',
        type=str,
        help='Comma-separated list of preprocessing methods to include (e.g., pypdf,unstructured)'
    )

    parser.add_argument(
        '--no-metrics',
        action='store_true',
        help='Skip performance metrics display'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    try:
        args = parse_arguments()

        print("🚀 CHUNKING STRATEGIES DEMONSTRATION")
        print("Processing all available documents with individual output files\n")

        # Parse CLI filters
        strategies_filter = None
        if args.strategies:
            strategies_filter = [s.strip() for s in args.strategies.split(',')]

        methods_filter = None
        if args.methods:
            methods_filter = [m.strip() for m in args.methods.split(',')]

        # Find available preprocessed files
        data_dir = Path(__file__).parent.parent / "data" / "processed"
        available_files = find_available_files(data_dir, methods_filter)

        if not available_files:
            logger.error("No preprocessed files found!")
            logger.error(
                "Please run preprocessing first or check data directory.")
            return

        print(f"Found {len(available_files)} preprocessed files")

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

        # Filter strategies if specified
        if strategies_filter:
            strategies_config = {
                k: v for k, v in strategies_config.items()
                if k in strategies_filter
            }

        # Ensure output directory exists
        output_dir = Path(__file__).parent.parent / "data" / "chunks"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Output directory: {output_dir}")
        print(f"Strategies to process: {list(strategies_config.keys())}")

        # Process all combinations
        all_results = []
        total_combinations = 0
        successful_combinations = 0

        for file_path in available_files:
            try:
                # Load document
                document = load_preprocessed_document(file_path)
                print(f"\nProcessing document: {document.title}")

                # Process each strategy
                for strategy_name, strategy_config in strategies_config.items():
                    print(f"  - Strategy: {strategy_name}")
                    total_combinations += 1

                    result = process_single_combination(
                        document, strategy_name, strategy_config, output_dir
                    )
                    all_results.append(result)

                    if not result['error']:
                        successful_combinations += 1

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue

        # Display summary
        print(f"\n✅ PROCESSING COMPLETE")
        print(f"Total combinations processed: {total_combinations}")
        print(f"Successful: {successful_combinations}")
        print(f"Failed: {total_combinations - successful_combinations}")

        # Display performance summary
        display_performance_summary(all_results, not args.no_metrics)

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
