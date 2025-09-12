#!/usr/bin/env python3
"""Process markitdown files to create chunk files for indexing."""

from chunker.pipeline import ChunkingPipeline
from chunker.config import get_default_config
from chunker.models import ProcessedDocument
import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def process_markitdown_files():
    """Process all markitdown files to create chunk files."""

    markitdown_dir = Path('data/processed/markitdown')
    output_dir = Path('data/chunks')
    output_dir.mkdir(exist_ok=True)

    strategies = ['fixed_size', 'sliding_langchain',
                  'sliding_unstructured', 'semantic']

    print("🚀 Processing markitdown files for chunking...")

    processed_count = 0
    error_count = 0

    for json_file in markitdown_dir.glob('*.json'):
        print(f"\nProcessing: {json_file.name}")

        try:
            # Load document
            with open(json_file, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)

            # Create ProcessedDocument object
            document = ProcessedDocument(
                document_id=doc_data['document_id'],
                title=doc_data.get('title', doc_data['document_id']),
                authors=doc_data.get('authors', []),
                abstract=doc_data.get('abstract', ''),
                full_text=doc_data.get('full_text', ''),
                metadata=doc_data.get('metadata', {}),
                elements=doc_data.get('elements', []),
                processing_method='markitdown'
            )

            # Process each strategy
            for strategy in strategies:
                try:
                    # Create config for single strategy
                    config = get_default_config()
                    config.enabled_strategies = [strategy]

                    # Create pipeline
                    pipeline = ChunkingPipeline(config)

                    # Process document
                    results = pipeline.process_document(document)

                    if strategy in results and results[strategy].success and results[strategy].chunks:
                        # Create output filename
                        result = results[strategy]
                        output_filename = f"{document.document_id}_markitdown_{strategy}.json"
                        output_path = output_dir / output_filename

                        # Save chunks
                        chunk_data = {
                            'metadata': {
                                'document_id': document.document_id,
                                'extraction_method': 'markitdown',
                                'chunking_strategy': strategy,
                                'chunk_count': len(result.chunks),
                                'processed_at': result.created_at.isoformat(),
                                'performance_metrics': result.statistics.to_dict()
                            },
                            'chunks': [chunk.to_dict() for chunk in result.chunks]
                        }

                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(chunk_data, f, indent=2,
                                      ensure_ascii=False)

                        print(
                            f"  ✅ {strategy}: {len(result.chunks)} chunks -> {output_filename}")

                    else:
                        print(f"  ❌ {strategy}: No chunks produced")
                        error_count += 1

                except Exception as e:
                    print(f"  ❌ {strategy}: Error - {e}")
                    error_count += 1

            processed_count += 1

        except Exception as e:
            print(f"  ❌ Failed to process {json_file.name}: {e}")
            error_count += 1

    print(f"\n✅ Processing complete!")
    print(f"Documents processed: {processed_count}")
    print(f"Errors encountered: {error_count}")

    # List created files
    markitdown_chunks = list(output_dir.glob("*_markitdown_*.json"))
    print(f"Created {len(markitdown_chunks)} markitdown chunk files:")
    for chunk_file in sorted(markitdown_chunks):
        print(f"  - {chunk_file.name}")


if __name__ == "__main__":
    process_markitdown_files()
