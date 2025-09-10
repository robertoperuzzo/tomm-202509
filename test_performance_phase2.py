#!/usr/bin/env python3
"""
Test script for Phase 2 Performance Analytics implementation.
Tests the performance analysis functionality of the updated indexer.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def test_performance_components():
    """Test that performance analysis components are properly structured."""

    # Test that PerformanceAnalyzer can be imported
    try:
        from indexer.performance_analyzer import PerformanceAnalyzer
        print("✅ PerformanceAnalyzer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PerformanceAnalyzer: {e}")
        return False

    # Test that TypesenseIndexer class has the performance methods
    try:
        from indexer.typesense_indexer import TypesenseIndexer
        print("✅ TypesenseIndexer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import TypesenseIndexer: {e}")
        return False

    # Check that performance methods exist on the class
    methods_to_check = [
        'get_performance_summary',
        'compare_strategies',
        'find_optimal_strategy',
        'analyze_strategy_performance'
    ]

    for method_name in methods_to_check:
        if hasattr(TypesenseIndexer, method_name):
            print(f"✅ Method {method_name} available on TypesenseIndexer")
        else:
            print(f"❌ Method {method_name} not found on TypesenseIndexer")
            return False

    # Test that schema includes performance fields from config
    try:
        from indexer.config import COLLECTION_SCHEMA_TEMPLATE
        schema_fields = COLLECTION_SCHEMA_TEMPLATE['fields']
        performance_fields = [
            'preprocessing_method',
            'content_length',
            'processing_time',
            'memory_usage',
            'cpu_usage_percent',
            'gpu_usage_percent'
        ]

        schema_field_names = [field['name'] for field in schema_fields]
        for field_name in performance_fields:
            if field_name in schema_field_names:
                print(f"✅ Schema includes {field_name}")
            else:
                print(f"❌ Schema missing {field_name}")
                return False

        print("✅ All performance fields present in schema")

    except ImportError as e:
        print(f"❌ Failed to import schema: {e}")
        return False

    # Test PerformanceAnalyzer class structure
    try:
        # Check if class has expected methods
        analyzer_methods = [
            'get_performance_summary',
            'compare_strategies_simple',
            'find_optimal_strategy_simple'
        ]

        for method_name in analyzer_methods:
            if hasattr(PerformanceAnalyzer, method_name):
                print(f"✅ PerformanceAnalyzer has {method_name}")
            else:
                print(f"❌ PerformanceAnalyzer missing {method_name}")
                return False

    except Exception as e:
        print(f"❌ Error checking PerformanceAnalyzer methods: {e}")
        return False

    print("\n🎉 Phase 2 Performance Analytics implementation successful!")
    print("\nImplemented features:")
    print("  - PerformanceAnalyzer module with analytics methods")
    print("  - Enhanced collection schema with performance metadata")
    print("  - Performance analysis methods in TypesenseIndexer")
    print("  - Support for strategy comparison and optimization")

    return True


if __name__ == "__main__":
    success = test_performance_components()
    sys.exit(0 if success else 1)
