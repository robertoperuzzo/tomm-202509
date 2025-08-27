"""Generic document preprocessing module for the chunking strategies demo.

This module handles processing and cleaning documents from any source.
It extracts text content and metadata from PDF files and stores them as JSON.
Documents should be downloaded by specialized downloaders into data/raw folder.

Supports three PDF extraction methods as per ADR-006:
1. PyPDF2 - Raw baseline extraction (no OCR fixing)
2. LangChain PyPDFParser - Balanced approach with LangChain integration
3. Unstructured.io - Premium quality with structure awareness
"""

import json
import logging
import re
import time
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn
)

# PDF processing libraries
from PyPDF2 import PdfReader

# LangChain for PDF parsing
try:
    from langchain_community.document_loaders.parsers.pdf import PyPDFParser
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning(
        "LangChain not available. PDF parsing will be limited."
    )
    PyPDFParser = None
    Document = None
    LANGCHAIN_AVAILABLE = False

# Unstructured.io for enhanced document processing
try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.documents.elements import Element
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning(
        "Unstructured.io not available. Falling back to traditional "
        "PDF processing."
    )
    partition_pdf = None
    Element = None
    UNSTRUCTURED_AVAILABLE = False

from ..config import DATA_RAW_PATH, DATA_PROCESSED_PATH, LOGS_PATH

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


@dataclass
class ExtractionResult:
    """Result of PDF text extraction with performance and quality metrics."""
    text: str
    performance_metrics: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    method_specific_data: Dict[str, Any]


class PerformanceTracker:
    """Context manager for tracking performance metrics during PDF extraction."""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.process = psutil.Process()
    
    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def get_metrics(self, text_length: int, pages_processed: int) -> Dict[str, Any]:
        """Calculate performance metrics."""
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        processing_time = end_time - self.start_time
        memory_usage = max(end_memory - self.start_memory, 0)
        
        return {
            "processing_time_seconds": round(processing_time, 3),
            "memory_usage_mb": round(memory_usage, 2),
            "characters_extracted": text_length,
            "pages_processed": pages_processed,
            "extraction_rate": round(text_length / processing_time, 2) if processing_time > 0 else 0
        }


class QualityAnalyzer:
    """Analyzer for assessing text extraction quality."""
    
    @staticmethod
    def analyze_text(text: str) -> Dict[str, Any]:
        """Analyze text quality metrics."""
        if not text:
            return {
                "text_length": 0,
                "word_count": 0,
                "unique_words": 0,
                "readability_score": 0.0,
                "ocr_artifact_count": 0,
                "structure_elements": 0
            }
        
        words = text.split()
        unique_words = set(word.lower().strip('.,!?;:"()[]{}') for word in words)
        
        # Estimate OCR artifacts (character spacing, broken words)
        ocr_artifacts = 0
        # Look for spaced out characters like "b a c k t r a c k i n g"
        ocr_artifacts += len(re.findall(r'\b[a-zA-Z]\s+[a-zA-Z]\s+[a-zA-Z]', text))
        # Look for broken words with numbers/special chars
        ocr_artifacts += len(re.findall(r'[a-zA-Z]+\d+[a-zA-Z]+', text))
        # Look for excessive consecutive spaces
        ocr_artifacts += len(re.findall(r'\s{3,}', text))
        
        # Simple readability based on sentence structure
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        readability_score = min(100, max(0, 100 - abs(avg_sentence_length - 20) * 2))
        
        return {
            "text_length": len(text),
            "word_count": len(words),
            "unique_words": len(unique_words),
            "readability_score": round(readability_score, 2),
            "ocr_artifact_count": ocr_artifacts,
            "structure_elements": 0  # Will be overridden by methods that detect structure
        }


class DocumentPreprocessor:
    """Generic document preprocessor for PDF text extraction and cleaning.
    
    This class handles processing documents from any source that have been
    downloaded to the raw data directory. It extracts text, cleans it,
    and saves processed documents as JSON.
    
    Supports three extraction methods as per ADR-006:
    - pypdf2: Raw baseline extraction (no OCR fixing)
    - langchain: Balanced approach with LangChain integration  
    - unstructured: Premium quality with structure awareness
    """

    # Supported extraction methods
    SUPPORTED_METHODS = ['pypdf2', 'langchain', 'unstructured']

    def __init__(self, raw_path: Optional[Path] = None, 
                 processed_path: Optional[Path] = None):
        """Initialize the preprocessor with necessary paths.
        
        Args:
            raw_path: Path to raw documents (defaults to DATA_RAW_PATH)
            processed_path: Path for processed outputs (defaults to DATA_PROCESSED_PATH)
        """
        self.raw_path = raw_path or DATA_RAW_PATH
        self.processed_path = processed_path or DATA_PROCESSED_PATH
        self.logs_path = LOGS_PATH

        # Create directories if they don't exist
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)

    def _get_method_output_path(self, method: str) -> Path:
        """Get output directory for specific extraction method."""
        method_dir = self.processed_path / method
        method_dir.mkdir(parents=True, exist_ok=True)
        return method_dir

    def _generate_output_filename(self, pdf_path: Path) -> str:
        """Generate filename based on original PDF name and timestamp."""
        # Convert original filename to lowercase, remove spaces and special chars
        original_name = pdf_path.stem.lower()
        clean_name = re.sub(r'[^a-z0-9]', '', original_name)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{clean_name}_{timestamp}.json"

    def discover_documents(self, file_pattern: str = "*.pdf") -> List[Path]:
        """Discover documents in the raw data directory.
        
        Args:
            file_pattern: Glob pattern for document files
            
        Returns:
            List of document file paths
        """
        documents = list(self.raw_path.glob(file_pattern))
        console.print(f"[blue]Found {len(documents)} documents in {self.raw_path}[/blue]")
        return documents

    def extract_metadata_from_filename(self, file_path: Path) -> Dict:
        """Extract metadata from filename.
        
        This is a basic implementation that can be extended for specific
        filename patterns from different sources.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with extracted metadata
        """
        filename = file_path.stem
        
        # Basic metadata extraction
        metadata = {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size if file_path.exists() else 0,
            'document_id': filename,
        }
        
        # Try to extract ArXiv ID if it looks like an ArXiv paper
        # Handle both old format (YYMMNNN) and new format (YYYY.NNNNN)
        arxiv_match = (re.match(r'^(\d{4}\.\d{4,5})', filename) or
                       re.match(r'^(\d{7})', filename))
        if arxiv_match:
            metadata['source'] = 'arxiv'
            metadata['arxiv_id'] = arxiv_match.group(1)
            # Extract title from filename (after arxiv_id_)
            title_part = filename.split('_', 1)
            if len(title_part) > 1:
                metadata['title'] = title_part[1].replace('_', ' ')
        else:
            metadata['source'] = 'unknown'
            metadata['title'] = filename.replace('_', ' ')
        
        return metadata

    def extract_text_from_pdf(self, pdf_path: Union[str, Path],
                               method: str = "unstructured",
                               track_performance: bool = True) -> Optional[ExtractionResult]:
        """Extract text from PDF file using specified method.
        
        Args:
            pdf_path: Path to PDF file
            method: Extraction method ("pypdf2", "langchain", "unstructured")
            track_performance: Whether to track performance metrics
            
        Returns:
            ExtractionResult or None if extraction fails
        """
        if method not in self.SUPPORTED_METHODS:
            # Handle deprecated method names with warnings
            method_mapping = {
                "pdfplumber": "langchain",
                "pymupdf": "langchain"
            }
            if method in method_mapping:
                logger.warning(
                    f"Method '{method}' is deprecated. Using '{method_mapping[method]}' instead."
                )
                method = method_mapping[method]
            else:
                logger.error(f"Unsupported extraction method: {method}")
                return None

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.error("PDF file not found: %s", pdf_path)
            return None
        
        try:
            if method == "unstructured":
                return self._extract_with_unstructured(pdf_path, track_performance)
            elif method == "langchain":
                return self._extract_with_langchain(pdf_path, track_performance)
            elif method == "pypdf2":
                return self._extract_with_pypdf2(pdf_path, track_performance)
                
        except Exception as e:
            logger.error(
                "Error extracting text from %s with %s: %s",
                pdf_path, method, e
            )
            # Simple fallback chain: unstructured -> langchain -> pypdf2
            fallback_methods = ["unstructured", "langchain", "pypdf2"]
            if method == "unstructured" and not UNSTRUCTURED_AVAILABLE:
                fallback_methods = ["langchain", "pypdf2"]
            if method == "langchain" and not LANGCHAIN_AVAILABLE:
                fallback_methods = ["unstructured", "pypdf2"]
            
            for fallback in fallback_methods:
                if fallback != method:
                    try:
                        logger.info("Trying fallback method: %s", fallback)
                        return self.extract_text_from_pdf(pdf_path, fallback, track_performance)
                    except Exception as fallback_e:
                        logger.warning(
                            "Fallback method %s also failed: %s",
                            fallback, fallback_e
                        )
                        continue
            return None

    def _extract_with_pypdf2(self, pdf_path: Path, 
                            track_performance: bool = True) -> ExtractionResult:
        """Raw PyPDF2 extraction with NO OCR fixing - demonstrates baseline quality."""
        with PerformanceTracker() as tracker:
            text_parts = []
            pages_processed = 0
            
            with open(pdf_path, 'rb') as file:
                reader = PdfReader(file)
                pages_processed = len(reader.pages)
                
                for page in reader.pages:
                    text = page.extract_text()
                    if text.strip():
                        text_parts.append(text)
            
            # Join text without any OCR fixing (raw extraction)
            raw_text = "\n".join(text_parts)
            
            # Calculate metrics
            performance_metrics = tracker.get_metrics(len(raw_text), pages_processed) if track_performance else {}
            quality_metrics = QualityAnalyzer.analyze_text(raw_text)
            
            return ExtractionResult(
                text=raw_text,
                performance_metrics=performance_metrics,
                quality_metrics=quality_metrics,
                method_specific_data={"extraction_method": "pypdf2", "pages_processed": pages_processed}
            )

    def _extract_with_langchain(self, pdf_path: Path,
                               track_performance: bool = True) -> ExtractionResult:
        """LangChain PyPDFParser extraction with Document objects."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is not available")
        
        with PerformanceTracker() as tracker:
            parser = PyPDFParser()
            
            # Parse PDF into LangChain Documents
            with open(pdf_path, 'rb') as file:
                documents = parser.parse(file, {"source": str(pdf_path)})
            
            # Extract text from Document objects
            text_parts = [doc.page_content for doc in documents]
            full_text = "\n".join(text_parts)
            pages_processed = len(documents)
            
            # Calculate metrics
            performance_metrics = tracker.get_metrics(len(full_text), pages_processed) if track_performance else {}
            quality_metrics = QualityAnalyzer.analyze_text(full_text)
            
            return ExtractionResult(
                text=full_text,
                performance_metrics=performance_metrics,
                quality_metrics=quality_metrics,
                method_specific_data={
                    "extraction_method": "langchain",
                    "pages_processed": pages_processed,
                    "document_objects": len(documents)
                }
            )

    def _extract_with_unstructured(self, pdf_path: Path,
                                  track_performance: bool = True,
                                  strategy: str = "auto",
                                  extract_images: bool = False) -> ExtractionResult:
        """Extract text and structure using Unstructured.io with performance tracking.
        
        Args:
            pdf_path: Path to PDF file
            track_performance: Whether to track performance metrics
            strategy: Processing strategy ("auto", "fast", "ocr_only", "hi_res")
            extract_images: Whether to extract and process images
            
        Returns:
            ExtractionResult with elements, metadata, and flat text
        """
        if not UNSTRUCTURED_AVAILABLE:
            raise ImportError("Unstructured.io is not available")
        
        with PerformanceTracker() as tracker:
            # Use unstructured to partition the PDF
            elements = partition_pdf(
                str(pdf_path),
                strategy=strategy,
                extract_images=extract_images,
                infer_table_structure=True,
                chunking_strategy=None  # We handle chunking separately
            )
            
            # Process elements to extract text and metadata
            processed_elements = []
            full_text_parts = []
            
            for i, element in enumerate(elements):
                element_dict = {
                    'type': element.category,
                    'text': str(element),
                    'element_id': f"elem_{i}",
                    'metadata': element.metadata.to_dict() if hasattr(element, 'metadata') else {}
                }
                processed_elements.append(element_dict)
                full_text_parts.append(str(element))
            
            # Combine all text
            full_text = "\n".join(full_text_parts)
            
            # Calculate metrics
            pages_processed = len(set(elem.metadata.page_number for elem in elements if hasattr(elem, 'metadata') and hasattr(elem.metadata, 'page_number'))) or 1
            performance_metrics = tracker.get_metrics(len(full_text), pages_processed) if track_performance else {}
            quality_metrics = QualityAnalyzer.analyze_text(full_text)
            quality_metrics["structure_elements"] = len(processed_elements)
            
            return ExtractionResult(
                text=full_text,
                performance_metrics=performance_metrics,
                quality_metrics=quality_metrics,
                method_specific_data={
                    'extraction_method': 'unstructured',
                    'processing_strategy': strategy,
                    'elements': processed_elements,
                    'element_count': len(processed_elements)
                }
            )
    def clean_text(self, text: str) -> str:
        """Clean extracted text by removing extra whitespace and formatting issues.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (basic patterns)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text

    def process_document(self, file_path: Path,
                        extraction_method: str = "unstructured",
                        track_performance: bool = True,
                        additional_metadata: Optional[Dict] = None) -> Optional[Dict]:
        """Process a single document with the new three-method approach.
        
        Args:
            file_path: Path to the document file
            extraction_method: PDF text extraction method ("pypdf2", "langchain", "unstructured")
            track_performance: Whether to track performance metrics
            additional_metadata: Optional additional metadata to include
            
        Returns:
            Processed document dictionary or None if processing fails
        """
        try:
            # Extract basic metadata from filename
            metadata = self.extract_metadata_from_filename(file_path)
            
            # Add any additional metadata provided
            if additional_metadata:
                metadata.update(additional_metadata)
            
            # Extract text using the new standardized approach
            extraction_result = self.extract_text_from_pdf(
                file_path, extraction_method, track_performance
            )
            
            if not extraction_result or not extraction_result.text:
                logger.warning("No text extracted from %s", file_path)
                return None
            
            # Build processed document with all metrics
            processed_document = {
                **metadata,  # Include all metadata
                'full_text': extraction_result.text,
                'text_length': len(extraction_result.text),
                'word_count': len(extraction_result.text.split()),
                'extraction_method': extraction_method,
                'processed_at': datetime.utcnow().isoformat(),
                'performance_metrics': extraction_result.performance_metrics,
                'quality_metrics': extraction_result.quality_metrics,
                **extraction_result.method_specific_data
            }
            
            logger.info("Processed %s (%d chars) with %s", 
                       file_path.name, len(extraction_result.text), extraction_method)
            return processed_document
            
        except Exception as e:
            logger.error("Error processing document %s: %s", file_path, e)
            return None

    def save_processed_document(self, processed_document: Dict, 
                               pdf_path: Path,
                               extraction_method: str) -> str:
        """Save a single processed document to method-specific directory.
        
        Args:
            processed_document: Processed document dictionary
            pdf_path: Original PDF path (for filename generation)
            extraction_method: Extraction method used
            
        Returns:
            Path to saved file
        """
        # Get method-specific output directory
        output_dir = self._get_method_output_path(extraction_method)
        
        # Generate filename based on original PDF
        filename = self._generate_output_filename(pdf_path)
        output_path = output_dir / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_document, f, indent=2, ensure_ascii=False)
            
            console.print(
                f"[green]Saved processed document to {output_path}[/green]"
            )
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error saving processed document: {e}")
            raise

    def process_documents(self, file_paths: Optional[List[Path]] = None,
                          file_pattern: str = "*.pdf",
                          extraction_method: str = "unstructured",
                          track_performance: bool = True,
                          metadata_mapping: Optional[Dict[str, Dict]] = None,
                          save_individual: bool = True
                          ) -> List[Dict]:
        """Process multiple documents with the new three-method approach.
        
        Args:
            file_paths: Specific file paths to process (if None, discovers all)
            file_pattern: Glob pattern for document discovery
            extraction_method: PDF text extraction method ("pypdf2", "langchain", "unstructured")
            track_performance: Whether to track performance metrics
            metadata_mapping: Optional mapping of filename -> additional metadata
            save_individual: Whether to save each document individually
            
        Returns:
            List of processed document dictionaries
        """
        if file_paths is None:
            file_paths = self.discover_documents(file_pattern)
        
        if not file_paths:
            console.print("[yellow]No documents found to process[/yellow]")
            return []
        
        console.print(f"[blue]Processing {len(file_paths)} documents with {extraction_method}...[/blue]")
        
        processed_documents = []
        saved_files = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            
            task = progress.add_task("Processing documents...", total=len(file_paths))
            
            for file_path in file_paths:
                # Get additional metadata if provided
                additional_metadata = None
                if metadata_mapping and file_path.name in metadata_mapping:
                    additional_metadata = metadata_mapping[file_path.name]
                
                processed_doc = self.process_document(
                    file_path, extraction_method, track_performance, additional_metadata
                )
                
                if processed_doc:
                    processed_documents.append(processed_doc)
                    
                    # Save individual document to method-specific directory
                    if save_individual:
                        try:
                            output_path = self.save_processed_document(
                                processed_doc, file_path, extraction_method
                            )
                            saved_files.append(output_path)
                        except Exception as e:
                            logger.error(f"Failed to save {file_path}: {e}")
                
                progress.advance(task)
        
        success_rate = len(processed_documents) / len(file_paths) * 100
        console.print(
            f"[green]Successfully processed {len(processed_documents)}/"
            f"{len(file_paths)} documents ({success_rate:.1f}%)[/green]"
        )
        
        if save_individual:
            console.print(f"[blue]Saved {len(saved_files)} files to {extraction_method}/ directory[/blue]")
        
        return processed_documents

    def compare_extraction_methods(self, file_path: Path, 
                                  methods: Optional[List[str]] = None,
                                  track_performance: bool = True) -> Dict[str, Any]:
        """Compare all three extraction methods on a single document.
        
        Args:
            file_path: PDF file to process with all methods
            methods: List of methods to compare (defaults to all three)
            track_performance: Whether to track performance metrics
            
        Returns:
            Comparison results with metrics for each method
        """
        if methods is None:
            methods = self.SUPPORTED_METHODS.copy()
        
        results = {}
        
        console.print(f"[blue]Comparing extraction methods for {file_path.name}[/blue]")
        
        for method in methods:
            console.print(f"  Processing with {method}...")
            try:
                extraction_result = self.extract_text_from_pdf(file_path, method, track_performance)
                if extraction_result:
                    results[method] = {
                        'success': True,
                        'text_preview': extraction_result.text[:500] + "..." if len(extraction_result.text) > 500 else extraction_result.text,
                        'performance_metrics': extraction_result.performance_metrics,
                        'quality_metrics': extraction_result.quality_metrics,
                        'method_specific_data': extraction_result.method_specific_data
                    }
                else:
                    results[method] = {'success': False, 'error': 'No extraction result'}
            except Exception as e:
                results[method] = {'success': False, 'error': str(e)}
        
        # Create comparison report
        comparison_report = {
            'document': str(file_path),
            'comparison_timestamp': datetime.utcnow().isoformat(),
            'methods_compared': methods,
            'results': results,
            'summary': self._generate_comparison_summary(results)
        }
        
        return comparison_report
    
    def _generate_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from method comparison results."""
        successful_methods = [method for method, result in results.items() if result.get('success')]
        
        if not successful_methods:
            return {'error': 'No methods succeeded'}
        
        # Performance comparison
        performance_summary = {}
        quality_summary = {}
        
        for method in successful_methods:
            perf = results[method].get('performance_metrics', {})
            quality = results[method].get('quality_metrics', {})
            
            if perf:
                performance_summary[method] = {
                    'processing_time': perf.get('processing_time_seconds', 0),
                    'extraction_rate': perf.get('extraction_rate', 0),
                    'memory_usage': perf.get('memory_usage_mb', 0)
                }
            
            if quality:
                quality_summary[method] = {
                    'text_length': quality.get('text_length', 0),
                    'ocr_artifacts': quality.get('ocr_artifact_count', 0),
                    'readability_score': quality.get('readability_score', 0)
                }
        
        # Find best performing method
        fastest_method = min(performance_summary.keys(), 
                           key=lambda x: performance_summary[x]['processing_time']) if performance_summary else None
        highest_quality = max(quality_summary.keys(),
                            key=lambda x: quality_summary[x]['readability_score']) if quality_summary else None
        
        return {
            'successful_methods': successful_methods,
            'performance_comparison': performance_summary,
            'quality_comparison': quality_summary,
            'fastest_method': fastest_method,
            'highest_quality_method': highest_quality,
            'recommendation': self._recommend_method(performance_summary, quality_summary)
        }
    
    def _recommend_method(self, performance_summary: Dict, quality_summary: Dict) -> str:
        """Recommend best method based on performance and quality trade-offs."""
        if not performance_summary or not quality_summary:
            return "unstructured"  # Default fallback
        
        # Simple recommendation logic
        # If unstructured has good performance, recommend it for quality
        # If pypdf2 is much faster and quality is acceptable, recommend it
        # Otherwise recommend langchain as balanced option
        
        methods = set(performance_summary.keys()) & set(quality_summary.keys())
        if not methods:
            return "unstructured"
        
        if "unstructured" in methods:
            unstructured_time = performance_summary["unstructured"]["processing_time"]
            if unstructured_time < 10:  # Less than 10 seconds
                return "unstructured - Best quality with reasonable performance"
        
        if "pypdf2" in methods and "unstructured" in methods:
            pypdf2_time = performance_summary["pypdf2"]["processing_time"]
            unstructured_time = performance_summary["unstructured"]["processing_time"]
            if pypdf2_time < unstructured_time / 3:  # 3x faster
                return "pypdf2 - Much faster, acceptable for speed-critical applications"
        
        if "langchain" in methods:
            return "langchain - Balanced performance and quality"
        
        return "unstructured - Default recommendation"

    def save_processed_documents(self, processed_documents: List[Dict], 
                               filename: Optional[str] = None) -> str:
        """Save processed documents to JSON file.
        
        Args:
            processed_documents: List of processed document dictionaries
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"processed_documents_{timestamp}.json"
        
        output_path = self.processed_path / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_documents, f, indent=2, ensure_ascii=False)
            
            console.print(
                f"[green]Saved {len(processed_documents)} processed documents "
                f"to {output_path}[/green]"
            )
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error saving processed documents: {e}")
            raise

    def get_processing_stats(self, processed_documents: List[Dict]) -> Dict:
        """Get statistics about processed documents.
        
        Args:
            processed_documents: List of processed documents
            
        Returns:
            Statistics dictionary
        """
        if not processed_documents:
            return {}
        
        total_docs = len(processed_documents)
        total_chars = sum(doc['text_length'] for doc in processed_documents)
        total_words = sum(doc['word_count'] for doc in processed_documents)
        
        # Group by source
        sources = {}
        extraction_methods = {}
        
        for doc in processed_documents:
            source = doc.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
            
            method = doc.get('extraction_method', 'unknown')
            extraction_methods[method] = extraction_methods.get(method, 0) + 1
        
        return {
            'total_documents': total_docs,
            'total_characters': total_chars,
            'total_words': total_words,
            'avg_characters_per_doc': total_chars / total_docs if total_docs > 0 else 0,
            'avg_words_per_doc': total_words / total_docs if total_docs > 0 else 0,
            'sources': sources,
            'extraction_methods': extraction_methods,
        }


def load_processed_documents(file_path: str) -> List[Dict]:
    """Load processed documents from JSON file.
    
    Args:
        file_path: Path to processed documents JSON file
        
    Returns:
        List of processed document dictionaries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        logger.info(f"Loaded {len(documents)} processed documents from {file_path}")
        return documents
        
    except Exception as e:
        logger.error(f"Error loading processed documents: {e}")
        return []


if __name__ == "__main__":
    def main():
        """Demo function showcasing the new three-method approach per ADR-006."""
        preprocessor = DocumentPreprocessor()
        
        # Discover documents
        documents = preprocessor.discover_documents()
        if not documents:
            print("No PDF documents found in raw data directory")
            return
        
        console.print("\n[bold blue]ADR-006: Three-Method PDF Extraction Demo[/bold blue]")
        
        # Process documents with all three methods
        all_results = {}
        for method in preprocessor.SUPPORTED_METHODS:
            console.print(f"\n[yellow]Processing with {method} method...[/yellow]")
            try:
                processed_docs = preprocessor.process_documents(
                    extraction_method=method,
                    track_performance=True,
                    save_individual=True  # Save to method-specific directories
                )
                all_results[method] = processed_docs
                
                # Show method-specific statistics
                if processed_docs:
                    stats = preprocessor.get_processing_stats(processed_docs)
                    console.print(f"[green]{method} completed: {len(processed_docs)} documents[/green]")
                    
                    # Show performance summary for first document
                    if processed_docs[0].get('performance_metrics'):
                        perf = processed_docs[0]['performance_metrics']
                        console.print(f"  Sample performance: {perf.get('processing_time_seconds', 0):.2f}s, "
                                    f"{perf.get('extraction_rate', 0):.0f} chars/sec")
                        
            except Exception as e:
                console.print(f"[red]Error with {method}: {e}[/red]")
                all_results[method] = []
        
        # Compare methods on first document if available
        if documents and len(documents) > 0:
            console.print(f"\n[yellow]Comparing all methods on {documents[0].name}...[/yellow]")
            try:
                comparison = preprocessor.compare_extraction_methods(documents[0])
                
                # Save comparison report
                comparison_dir = preprocessor.processed_path / "comparative_analysis"
                comparison_dir.mkdir(exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                comparison_file = comparison_dir / f"comparison_report_{timestamp}.json"
                
                with open(comparison_file, 'w', encoding='utf-8') as f:
                    json.dump(comparison, f, indent=2, ensure_ascii=False)
                
                console.print(f"[green]Comparison report saved to: {comparison_file}[/green]")
                
                # Show recommendation
                recommendation = comparison['summary'].get('recommendation', 'N/A')
                console.print(f"[bold]Method recommendation: {recommendation}[/bold]")
                
            except Exception as e:
                console.print(f"[red]Error in method comparison: {e}[/red]")
        
        # Final summary
        console.print("\n[bold]Processing Summary:[/bold]")
        total_processed = sum(len(results) for results in all_results.values())
        console.print(f"  Total documents processed: {total_processed}")
        
        for method, results in all_results.items():
            if results:
                console.print(f"  {method}: {len(results)} documents → data/processed/{method}/")
            else:
                console.print(f"  {method}: Failed")
        
        console.print("\n[bold green]ADR-006 implementation complete![/bold green]")
        console.print("Check data/processed/ subdirectories for method-specific outputs.")
    
    main()
