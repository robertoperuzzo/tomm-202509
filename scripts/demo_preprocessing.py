#!/usr/bin/env python3
"""
Simple PDF Preprocessing Script

This script demonstrates the ADR-007 implementation by processing PDFs
in the data/raw/ directory using the two standardized extraction methods.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessor.document_preprocessor import DocumentPreprocessor


def main():
    """Process PDFs with both available methods."""
    print("PDF Preprocessing Demo - ADR-007 Implementation")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = DocumentPreprocessor()
    
    # Discover documents
    documents = preprocessor.discover_documents()
    if not documents:
        print("No PDF documents found in data/raw/ directory")
        return
    
    print(f"\nFound {len(documents)} PDF files:")
    for doc in documents:
        print(f"  - {doc.name}")
    
    # Process with both available methods
    methods = ['pypdf', 'marker', 'unstructured']
    all_results = {}
    
    for method in methods:
        print(f"\n{'-' * 30}")
        print(f"Processing with {method.upper()} method...")
        print(f"{'-' * 30}")
        
        try:
            processed_docs = preprocessor.process_documents(
                extraction_method=method,
                track_performance=True,
                save_individual=True
            )
            
            all_results[method] = processed_docs
            
            if processed_docs:
                print(f"✓ Successfully processed {len(processed_docs)} documents")
                
                # Show stats for first document
                first_doc = processed_docs[0]
                print(f"  Sample: {first_doc['file_name']}")
                print(f"  Text length: {first_doc['text_length']:,} characters")
                
                if 'performance_metrics' in first_doc:
                    perf = first_doc['performance_metrics']
                    time_taken = perf.get('processing_time_seconds', 0)
                    rate = perf.get('extraction_rate', 0)
                    print(f"  Processing time: {time_taken:.2f} seconds")
                    print(f"  Extraction rate: {rate:.0f} chars/second")
            else:
                print(f"❌ No documents processed with {method}")
                
        except Exception as e:
            print(f"❌ Error with {method}: {e}")
            all_results[method] = []
    
    # Summary
    print(f"\n{'=' * 50}")
    print("PROCESSING SUMMARY")
    print(f"{'=' * 50}")
    
    total_processed = 0
    for method, results in all_results.items():
        count = len(results)
        total_processed += count
        status = "✓ Success" if count > 0 else "❌ Failed"
        print(f"{method.upper():<12}: {count:>3} documents {status}")
    
    print(f"\nTotal processed: {total_processed} documents")
    print(f"\nOutput directories:")
    print(f"  data/processed/pypdf/")
    print(f"  data/processed/unstructured/")
    
    # Compare methods on first document
    if documents:
        print(f"\n{'-' * 30}")
        print("METHOD COMPARISON")
        print(f"{'-' * 30}")
        
        first_doc = documents[0]
        print(f"Comparing extraction methods for: {first_doc.name}")
        
        try:
            comparison = preprocessor.compare_extraction_methods(first_doc)
            
            # Show results
            results = comparison.get('results', {})
            for method, result in results.items():
                if result.get('success'):
                    perf = result.get('performance_metrics', {})
                    quality = result.get('quality_metrics', {})
                    
                    time_taken = perf.get('processing_time_seconds', 0)
                    text_len = quality.get('text_length', 0)
                    readability = quality.get('readability_score', 0)
                    
                    print(f"\n{method.upper()}:")
                    print(f"  Time: {time_taken:.2f}s")
                    print(f"  Text: {text_len:,} chars")
                    print(f"  Readability: {readability:.1f}/100")
                else:
                    print(f"\n{method.upper()}: ❌ Failed")
            
            # Show recommendation
            recommendation = comparison.get('summary', {}).get('recommendation')
            if recommendation:
                print(f"\nRecommendation: {recommendation}")
                
        except Exception as e:
            print(f"Comparison failed: {e}")
    
    print(f"\n{'=' * 50}")
    print("Demo complete! Check the data/processed/ directories.")


if __name__ == "__main__":
    main()
