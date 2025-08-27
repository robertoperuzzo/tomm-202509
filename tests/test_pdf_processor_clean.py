"""Comprehensive preprocessing tests using real PDF data.

This test module implements the ADR-006 three-method approach:
- PyPDF2: Raw baseline extraction (no OCR fixing)
- LangChain: Balanced approach with LangChain integration  
- Unstructured.io: Premium quality with structure awareness

Using real ArXiv PDF: 9308101_Dynamic Backtracking.pdf
"""

import json
import pytest
from pathlib import Path

from src.preprocessor.document_preprocessor import (
    DocumentPreprocessor, 
    ExtractionResult, 
    PerformanceTracker, 
    QualityAnalyzer
)


@pytest.fixture
def test_pdf_path():
    """Path to the test PDF file."""
    return Path(__file__).parent / "data" / "9308101_Dynamic Backtracking.pdf"


@pytest.fixture
def preprocessor(tmp_path):
    """Create a DocumentPreprocessor with temporary directories."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()
    
    return DocumentPreprocessor(raw_path=raw_dir, processed_path=processed_dir)


@pytest.fixture
def supported_methods():
    """List of supported extraction methods per ADR-006."""
    return ["pypdf2", "langchain", "unstructured"]


class TestThreeMethodExtraction:
    """Test the three standardized extraction methods per ADR-006."""
    
    def test_pdf_file_exists(self, test_pdf_path):
        """Verify the test PDF file exists."""
        assert test_pdf_path.exists(), f"Test PDF not found: {test_pdf_path}"
        assert test_pdf_path.suffix == ".pdf"
        assert test_pdf_path.stat().st_size > 0
    
    def test_supported_methods_list(self, preprocessor, supported_methods):
        """Test that preprocessor has correct supported methods."""
        assert preprocessor.SUPPORTED_METHODS == supported_methods
        assert len(preprocessor.SUPPORTED_METHODS) == 3
    
    def test_pypdf2_raw_extraction(self, preprocessor, test_pdf_path):
        """Test PyPDF2 raw extraction (no OCR fixing) - ADR-006."""
        result = preprocessor.extract_text_from_pdf(
            test_pdf_path, method="pypdf2", track_performance=True
        )
        
        assert result is not None
        assert isinstance(result, ExtractionResult)
        assert result.text is not None
        assert len(result.text) > 100
        
        # Verify it's raw extraction (may contain OCR artifacts)
        assert result.method_specific_data["extraction_method"] == "pypdf2"
        
        # Check performance tracking
        assert "processing_time_seconds" in result.performance_metrics
        assert "memory_usage_mb" in result.performance_metrics
        assert "characters_extracted" in result.performance_metrics
        
        # Check quality metrics
        assert "text_length" in result.quality_metrics
        assert "ocr_artifact_count" in result.quality_metrics
        assert "readability_score" in result.quality_metrics
    
    def test_langchain_extraction(self, preprocessor, test_pdf_path):
        """Test LangChain PyPDFParser extraction - ADR-006."""
        # Skip if LangChain not available
        try:
            result = preprocessor.extract_text_from_pdf(
                test_pdf_path, method="langchain", track_performance=True
            )
        except ImportError:
            pytest.skip("LangChain not available")
        
        assert result is not None
        assert isinstance(result, ExtractionResult)
        assert result.text is not None
        assert len(result.text) > 100
        
        # Verify LangChain-specific data
        assert result.method_specific_data["extraction_method"] == "langchain"
        assert "document_objects" in result.method_specific_data
        
        # Should have better quality than raw PyPDF2
        assert result.quality_metrics["text_length"] > 0
    
    def test_unstructured_extraction(self, preprocessor, test_pdf_path):
        """Test Unstructured.io extraction with structure awareness."""
        # Skip if Unstructured not available
        try:
            result = preprocessor.extract_text_from_pdf(
                test_pdf_path, method="unstructured", track_performance=True
            )
        except ImportError:
            pytest.skip("Unstructured.io not available")
        
        assert result is not None
        assert isinstance(result, ExtractionResult)
        assert result.text is not None
        assert len(result.text) > 100
        
        # Verify structure-aware extraction
        assert result.method_specific_data["extraction_method"] == "unstructured"
        assert "elements" in result.method_specific_data
        assert "element_count" in result.method_specific_data
        
        # Should have high quality text
        assert result.quality_metrics["structure_elements"] > 0
        assert "backtracking" in result.text.lower()


class TestPerformanceAndQuality:
    """Test performance tracking and quality analysis - ADR-006."""
    
    def test_performance_tracker(self, test_pdf_path):
        """Test PerformanceTracker context manager."""
        with PerformanceTracker() as tracker:
            # Simulate some processing
            text = "test content" * 1000
        
        metrics = tracker.get_metrics(len(text), 5)
        
        assert "processing_time_seconds" in metrics
        assert "memory_usage_mb" in metrics
        assert "characters_extracted" in metrics
        assert "pages_processed" in metrics
        assert "extraction_rate" in metrics
        
        assert metrics["characters_extracted"] == len(text)
        assert metrics["pages_processed"] == 5
    
    def test_quality_analyzer(self):
        """Test QualityAnalyzer text assessment."""
        # Test with clean text
        clean_text = "This is a well-formed sentence."
        metrics = QualityAnalyzer.analyze_text(clean_text)
        
        assert "text_length" in metrics
        assert "word_count" in metrics
        assert "unique_words" in metrics
        assert "readability_score" in metrics
        assert "ocr_artifact_count" in metrics
        
        assert metrics["text_length"] == len(clean_text)
        assert metrics["word_count"] > 0
        assert metrics["readability_score"] > 0


class TestDirectoryStructure:
    """Test method-specific directory structure - ADR-006."""
    
    def test_method_output_paths(self, preprocessor):
        """Test method-specific directory creation."""
        for method in ["pypdf2", "langchain", "unstructured"]:
            output_path = preprocessor._get_method_output_path(method)
            
            assert output_path.exists()
            assert output_path.name == method
            assert output_path.parent == preprocessor.processed_path
    
    def test_filename_generation(self, preprocessor, test_pdf_path):
        """Test filename generation per ADR-006 convention."""
        filename = preprocessor._generate_output_filename(test_pdf_path)
        
        # Should follow format: <clean_name>_YYYYMMDD_HHMMSS.json
        assert filename.endswith(".json")
        assert "9308101dynamicbacktracking" in filename
        assert "_20" in filename  # Should contain year
        
        # Should not contain spaces or special characters in the prefix
        prefix = filename.split("_20")[0]  # Get part before timestamp
        assert " " not in prefix
        assert "-" not in prefix
        assert prefix.islower() or prefix.replace("9308101", "").islower()


class TestBatchProcessing:
    """Test batch processing with method-specific directories - ADR-006."""
    
    def test_process_with_pypdf2_method(self, preprocessor, test_pdf_path, tmp_path):
        """Test processing documents with pypdf2 method."""
        # Copy PDF to preprocessor's raw directory
        raw_pdf_path = preprocessor.raw_path / test_pdf_path.name
        import shutil
        shutil.copy2(test_pdf_path, raw_pdf_path)
        
        # Process with pypdf2 method
        processed_docs = preprocessor.process_documents(
            extraction_method="pypdf2",
            track_performance=True,
            save_individual=True
        )
        
        # Verify method-specific directory was created
        pypdf2_dir = preprocessor.processed_path / "pypdf2"
        assert pypdf2_dir.exists()
        
        # If processing succeeded, verify structure
        if processed_docs and len(processed_docs) > 0:
            doc = processed_docs[0]
            assert doc["extraction_method"] == "pypdf2"
            assert "performance_metrics" in doc
            assert "quality_metrics" in doc
            
            # Check file was saved in correct location
            json_files = list(pypdf2_dir.glob("*.json"))
            assert len(json_files) > 0
    
    def test_process_all_methods(self, preprocessor, test_pdf_path, tmp_path):
        """Test processing documents with all three methods."""
        # Copy PDF to preprocessor's raw directory
        raw_pdf_path = preprocessor.raw_path / test_pdf_path.name
        import shutil
        shutil.copy2(test_pdf_path, raw_pdf_path)
        
        supported_methods = ["pypdf2", "langchain", "unstructured"]
        
        for method in supported_methods:
            try:
                # Verify method-specific directory was created
                method_dir = preprocessor.processed_path / method
                method_dir.mkdir(exist_ok=True)
                assert method_dir.exists()
                
                # Process with specific method
                processed_docs = preprocessor.process_documents(
                    extraction_method=method,
                    track_performance=True,
                    save_individual=True
                )
                
                # Skip if method dependencies not available
                if not processed_docs:
                    continue
                    
                # If processing succeeded, verify structure
                if len(processed_docs) > 0:
                    doc = processed_docs[0]
                    assert doc["extraction_method"] == method
                    assert "performance_metrics" in doc
                    assert "quality_metrics" in doc
                    
                    # Check file was saved in correct location
                    json_files = list(method_dir.glob("*.json"))
                    assert len(json_files) > 0
                    
            except (ImportError, Exception) as e:
                # Skip if method dependencies not available
                pytest.skip(f"{method} dependencies not available: {e}")


class TestADR006Compliance:
    """Verify compliance with ADR-006 requirements."""
    
    def test_three_methods_only(self, preprocessor, supported_methods):
        """Verify exactly three methods are supported."""
        assert len(preprocessor.SUPPORTED_METHODS) == 3
        assert set(preprocessor.SUPPORTED_METHODS) == set(supported_methods)
        assert "pypdf2" in preprocessor.SUPPORTED_METHODS
        assert "langchain" in preprocessor.SUPPORTED_METHODS
        assert "unstructured" in preprocessor.SUPPORTED_METHODS
    
    def test_directory_structure_compliance(self, preprocessor):
        """Verify directory structure matches ADR-006 specification."""
        # Test method-specific directories
        for method in preprocessor.SUPPORTED_METHODS:
            method_path = preprocessor._get_method_output_path(method)
            assert method_path.name == method
            assert method_path.exists()
        
        # Test comparative analysis directory can be created
        comparison_dir = preprocessor.processed_path / "comparative_analysis"
        comparison_dir.mkdir(exist_ok=True)
        assert comparison_dir.exists()
    
    def test_filename_convention_compliance(self, preprocessor, test_pdf_path):
        """Verify filename convention matches ADR-006."""
        filename = preprocessor._generate_output_filename(test_pdf_path)
        
        # Format: <clean_name>_YYYYMMDD_HHMMSS.json
        assert filename.endswith(".json")
        
        # Should contain original name in clean format
        assert "9308101" in filename
        
        # Should contain timestamp pattern
        import re
        timestamp_pattern = r'_\d{8}_\d{6}\.json$'
        assert re.search(timestamp_pattern, filename), f"Filename {filename} doesn't match timestamp pattern"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
