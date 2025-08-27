"""Comprehensive preprocessing tests using real PDF data.

This test module implements the preprocessing requirements from the Demo Implementation Plan:
- Basic text extraction from PDFs
- Minimal cleaning (remove extra whitespace)  
- Extract basic metadata (title, authors, abstract)

Using real ArXiv PDF: 9308101_Dynamic Backtracking.pdf
"""

import json
import pytest
from pathlib import Path

from src.preprocessor.document_preprocessor import DocumentPreprocessor


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


class TestRealPDFProcessing:
    """Test preprocessing using real ArXiv PDF data."""
    
    def test_pdf_file_exists(self, test_pdf_path):
        """Verify the test PDF file exists."""
        assert test_pdf_path.exists(), f"Test PDF not found: {test_pdf_path}"
        assert test_pdf_path.suffix == ".pdf"
        assert test_pdf_path.stat().st_size > 0
    
    def test_text_extraction_from_real_pdf(self, preprocessor, test_pdf_path):
        """Test basic text extraction from real PDF - Demo Plan requirement."""
        # Test with different extraction methods
        for method in ["pdfplumber", "pymupdf", "pypdf2"]:
            text = preprocessor.extract_text_from_pdf(test_pdf_path, method=method)
            
            assert text is not None, f"Failed to extract text with {method}"
            assert len(text) > 100, f"Extracted text too short with {method}"
            # Handle OCR artifacts - now fixed in extraction
        assert "backtracking" in text.lower(), "Expected content not found"
    
    def test_minimal_text_cleaning(self, preprocessor, test_pdf_path):
        """Test minimal cleaning as specified in Demo Plan."""
        # Extract raw text
        raw_text = preprocessor.extract_text_from_pdf(test_pdf_path)
        assert raw_text is not None
        
        # Apply minimal cleaning
        cleaned_text = preprocessor.clean_text(raw_text)
        
        # Verify cleaning occurred
        cleaning_msg = "Cleaning should not increase length"
        assert len(cleaned_text) <= len(raw_text), cleaning_msg
        strip_msg = "Should remove leading/trailing whitespace"
        assert cleaned_text.strip() == cleaned_text, strip_msg
        
        # Check that excessive whitespace was removed
        space_check = ("  " not in cleaned_text or
                       cleaned_text.count("  ") < raw_text.count("  "))
        assert space_check
        
        # Verify content is preserved - now properly extracted
        assert "backtracking" in cleaned_text.lower()
    
    def test_metadata_extraction_from_filename(self, preprocessor, test_pdf_path):
        """Test basic metadata extraction as specified in Demo Plan."""
        metadata = preprocessor.extract_metadata_from_filename(test_pdf_path)
        
        # Check required metadata fields
        assert "document_id" in metadata
        assert "file_name" in metadata
        assert "title" in metadata
        assert "source" in metadata
        
        # Verify ArXiv-specific metadata
        assert metadata["source"] == "arxiv"
        assert "arxiv_id" in metadata
        assert metadata["arxiv_id"] == "9308101"
        assert "Dynamic Backtracking" in metadata["title"]
    
    def test_complete_document_processing(self, preprocessor, test_pdf_path, tmp_path):
        """Test complete document processing pipeline."""
        # Copy PDF to preprocessor's raw directory
        raw_pdf_path = preprocessor.raw_path / test_pdf_path.name
        import shutil
        shutil.copy2(test_pdf_path, raw_pdf_path)
        
        # Process the document
        processed_doc = preprocessor.process_document(raw_pdf_path)
        
        assert processed_doc is not None
        
        # Verify Demo Plan requirements are met
        required_fields = [
            "document_id", "file_name", "title", "source",
            "full_text", "text_length", "word_count",
            "extraction_method", "processed_at"
        ]
        
        for field in required_fields:
            assert field in processed_doc, f"Missing required field: {field}"
        
        # Verify text extraction worked
        assert processed_doc["text_length"] > 1000
        assert processed_doc["word_count"] > 100
        # Content should now be properly extracted
        assert "backtracking" in processed_doc["full_text"].lower()
        
        # Verify metadata
        assert processed_doc["source"] == "arxiv"
        assert processed_doc["arxiv_id"] == "9308101"
    
    def test_json_serialization_with_real_data(self, preprocessor, test_pdf_path, tmp_path):
        """Test saving processed data as JSON - Demo Plan requirement."""
        # Copy PDF to preprocessor's raw directory
        raw_pdf_path = preprocessor.raw_path / test_pdf_path.name
        import shutil
        shutil.copy2(test_pdf_path, raw_pdf_path)
        
        # Process document
        processed_doc = preprocessor.process_document(raw_pdf_path)
        assert processed_doc is not None
        
        # Save as JSON
        output_file = preprocessor.save_processed_documents([processed_doc])
        assert Path(output_file).exists()
        
        # Verify JSON is valid and complete
        with open(output_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        assert len(loaded_data) == 1
        doc = loaded_data[0]
        assert doc["arxiv_id"] == "9308101"
        assert "Dynamic Backtracking" in doc["title"]
        assert doc["source"] == "arxiv"
        assert "Dynamic Backtracking" in loaded_data[0]["title"]
        assert len(loaded_data[0]["full_text"]) > 1000


class TestProcessingPipeline:
    """Test the complete preprocessing pipeline with real data."""
    
    def test_pipeline_with_multiple_extraction_methods(self, preprocessor, test_pdf_path, tmp_path):
        """Test pipeline robustness with different PDF extraction methods."""
        # Copy PDF to preprocessor's raw directory
        raw_pdf_path = preprocessor.raw_path / test_pdf_path.name
        import shutil
        shutil.copy2(test_pdf_path, raw_pdf_path)
        
        methods = ["pdfplumber", "pymupdf", "pypdf2"]
        
        for method in methods:
            processed_doc = preprocessor.process_document(
                raw_pdf_path, 
                extraction_method=method
            )
            
            assert processed_doc is not None, f"Processing failed with {method}"
            assert processed_doc["extraction_method"] == method
            assert processed_doc["text_length"] > 500, f"Too little text extracted with {method}"
    
    def test_batch_processing_with_real_data(self, preprocessor, test_pdf_path, tmp_path):
        """Test batch processing capability."""
        # Copy PDF to preprocessor's raw directory  
        raw_pdf_path = preprocessor.raw_path / test_pdf_path.name
        import shutil
        shutil.copy2(test_pdf_path, raw_pdf_path)
        
        # Process all documents in the directory
        processed_docs = preprocessor.process_documents()
        
        assert len(processed_docs) == 1
        doc = processed_docs[0]
        assert doc["arxiv_id"] == "9308101"
        assert doc["source"] == "arxiv"
        
        # Test statistics generation
        stats = preprocessor.get_processing_stats(processed_docs)
        assert stats["total_documents"] == 1
        assert stats["total_characters"] > 1000
        assert stats["sources"]["arxiv"] == 1


class TestDemoPlanCompliance:
    """Verify compliance with Demo Implementation Plan requirements."""
    
    def test_simple_poc_requirements(self, preprocessor, test_pdf_path, tmp_path):
        """Verify we meet the 'Keep it simple for the first PoC' requirements."""
        # Copy PDF to preprocessor's raw directory
        raw_pdf_path = preprocessor.raw_path / test_pdf_path.name
        import shutil
        shutil.copy2(test_pdf_path, raw_pdf_path)
        
        # Process document
        processed_doc = preprocessor.process_document(raw_pdf_path)
        
        # Demo Plan requirement: "basic text extraction from PDFs"
        assert "full_text" in processed_doc
        assert len(processed_doc["full_text"]) > 1000
        
        # Demo Plan requirement: "minimal cleaning (remove extra whitespace)"
        full_text = processed_doc["full_text"]
        assert not full_text.startswith(" ")
        assert not full_text.endswith(" ")
        # Check for reduced excessive whitespace
        assert full_text.count("   ") < 10  # Should be minimal triple spaces
        
        # Demo Plan requirement: "extract basic metadata (title, authors, abstract)"
        # Note: We extract what we can from filename, more metadata would come from ArXiv API
        assert "title" in processed_doc
        assert "Dynamic Backtracking" in processed_doc["title"]
        assert processed_doc["source"] == "arxiv"
        assert processed_doc["arxiv_id"] == "9308101"
    
    def test_json_output_for_chunking_step(self, preprocessor, test_pdf_path, tmp_path):
        """Verify JSON output is ready for chunking strategies step."""
        # Copy PDF to preprocessor's raw directory
        raw_pdf_path = preprocessor.raw_path / test_pdf_path.name
        import shutil
        shutil.copy2(test_pdf_path, raw_pdf_path)
        
        # Process and save
        processed_docs = preprocessor.process_documents()
        output_file = preprocessor.save_processed_documents(processed_docs)
        
        # Load and verify structure matches what chunking strategies need
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert len(data) == 1
        doc = data[0]
        
        # Fields needed for chunking strategies
        chunking_required_fields = [
            "document_id",     # To identify chunks back to source
            "title",           # For metadata in chunks
            "full_text",       # The text to chunk
            "source"           # To track document origin
        ]
        
        for field in chunking_required_fields:
            assert field in doc, f"Missing field required for chunking: {field}"
            
        # Verify text is substantial enough for chunking
        assert len(doc["full_text"]) > 5000, "Document too short for meaningful chunking"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
