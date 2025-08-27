"""Generic document preprocessing module for the chunking strategies demo.

This module handles processing and cleaning documents from any source.
It extracts text content and metadata from PDF files and stores them as JSON.
Documents should be downloaded by specialized downloaders into data/raw folder.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn
)

# PDF processing libraries
import pdfplumber
import pymupdf  # fitz
from PyPDF2 import PdfReader

from ..config import DATA_RAW_PATH, DATA_PROCESSED_PATH, LOGS_PATH

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


class DocumentPreprocessor:
    """Generic document preprocessor for PDF text extraction and cleaning.
    
    This class handles processing documents from any source that have been
    downloaded to the raw data directory. It extracts text, cleans it,
    and saves processed documents as JSON.
    """

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
                             method: str = "pdfplumber") -> Optional[str]:
        """Extract text from PDF file using specified method.
        
        Args:
            pdf_path: Path to PDF file
            method: Extraction method ("pdfplumber", "pymupdf", "pypdf2")
            
        Returns:
            Extracted text or None if extraction fails
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return None
        
        try:
            if method == "pdfplumber":
                return self._extract_with_pdfplumber(pdf_path)
            elif method == "pymupdf":
                return self._extract_with_pymupdf(pdf_path)
            elif method == "pypdf2":
                return self._extract_with_pypdf2(pdf_path)
            else:
                logger.error(f"Unknown extraction method: {method}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path} with {method}: {e}")
            # Try fallback methods
            for fallback in ["pdfplumber", "pymupdf", "pypdf2"]:
                if fallback != method:
                    try:
                        logger.info(f"Trying fallback method: {fallback}")
                        if fallback == "pdfplumber":
                            return self._extract_with_pdfplumber(pdf_path)
                        elif fallback == "pymupdf":
                            return self._extract_with_pymupdf(pdf_path)
                        elif fallback == "pypdf2":
                            return self._extract_with_pypdf2(pdf_path)
                    except Exception as fallback_e:
                        logger.warning(f"Fallback method {fallback} also failed: {fallback_e}")
                        continue
            return None

    def _extract_with_pdfplumber(self, pdf_path: Path) -> str:
        """Extract text using pdfplumber."""
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        raw_text = "\n".join(text_parts)
        
        # Fix common OCR artifacts: character-spaced text
        return self._fix_ocr_artifacts(raw_text)
    
    def _fix_ocr_artifacts(self, text: str) -> str:
        """Fix common OCR artifacts like character-spaced words."""
        import re
        
        # Fix patterns like "b a c k t r a c k i n g" -> "backtracking"
        # This handles words that are completely character-spaced
        def fix_spaced_word(match):
            spaced_word = match.group()
            # Remove spaces between single letters
            fixed = re.sub(r'\b([a-zA-Z])\s+(?=[a-zA-Z])', r'\1', spaced_word)
            return fixed
        
        # Find sequences of single letters separated by spaces
        pattern = r'\b[a-zA-Z](?:\s+[a-zA-Z]){2,}\b'
        fixed_text = re.sub(pattern, fix_spaced_word, text)
        
        # Fix broken words like "Bac ktrac king" -> "Backtracking"
        # Pattern: word fragment + space + word fragment
        fixed_text = re.sub(r'\b([A-Za-z]{2,})\s+([a-z]{2,})\b',
                            lambda m: m.group(1) + m.group(2), fixed_text)
        
        # Fix specific OCR patterns like "ktrac king" -> "ktracking"
        fixed_text = re.sub(r'([a-z]+)\s+([a-z]+ing)\b',
                            lambda m: m.group(1) + m.group(2), fixed_text)
        
        # Fix line breaks within words (common in pymupdf)
        # Pattern: letter + newline + letter
        fixed_text = re.sub(r'([a-zA-Z])\n([a-z])', r'\1\2', fixed_text)
        
        return fixed_text

    def _extract_with_pymupdf(self, pdf_path: Path) -> str:
        """Extract text using PyMuPDF (fitz)."""
        doc = pymupdf.open(pdf_path)
        text_parts = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                text_parts.append(text)
        doc.close()
        raw_text = "\n".join(text_parts)
        
        # Fix OCR artifacts for pymupdf
        return self._fix_ocr_artifacts(raw_text)

    def _extract_with_pypdf2(self, pdf_path: Path) -> str:
        """Extract text using PyPDF2."""
        text_parts = []
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text = page.extract_text()
                if text.strip():
                    text_parts.append(text)
        raw_text = "\n".join(text_parts)
        
        # Fix OCR artifacts for pypdf2
        return self._fix_ocr_artifacts(raw_text)

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
                        extraction_method: str = "pdfplumber",
                        additional_metadata: Optional[Dict] = None) -> Optional[Dict]:
        """Process a single document.
        
        Args:
            file_path: Path to the document file
            extraction_method: PDF text extraction method to use
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
            
            # Extract text from PDF
            raw_text = self.extract_text_from_pdf(file_path, extraction_method)
            
            if not raw_text:
                logger.warning(f"No text extracted from {file_path}")
                return None
            
            # Clean the text
            cleaned_text = self.clean_text(raw_text)
            
            processed_document = {
                **metadata,  # Include all metadata
                'full_text': cleaned_text,
                'text_length': len(cleaned_text),
                'word_count': len(cleaned_text.split()),
                'extraction_method': extraction_method,
                'processed_at': datetime.utcnow().isoformat(),
            }
            
            logger.info(f"Processed {file_path.name} ({len(cleaned_text)} chars)")
            return processed_document
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return None

    def process_documents(self, file_paths: Optional[List[Path]] = None,
                         file_pattern: str = "*.pdf",
                         extraction_method: str = "pdfplumber",
                         metadata_mapping: Optional[Dict[str, Dict]] = None
                         ) -> List[Dict]:
        """Process multiple documents.
        
        Args:
            file_paths: Specific file paths to process (if None, discovers all)
            file_pattern: Glob pattern for document discovery
            extraction_method: PDF text extraction method to use
            metadata_mapping: Optional mapping of filename -> additional metadata
            
        Returns:
            List of processed document dictionaries
        """
        if file_paths is None:
            file_paths = self.discover_documents(file_pattern)
        
        if not file_paths:
            console.print("[yellow]No documents found to process[/yellow]")
            return []
        
        console.print(f"[blue]Processing {len(file_paths)} documents...[/blue]")
        
        processed_documents = []
        
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
                    file_path, extraction_method, additional_metadata
                )
                
                if processed_doc:
                    processed_documents.append(processed_doc)
                
                progress.advance(task)
        
        success_rate = len(processed_documents) / len(file_paths) * 100
        console.print(
            f"[green]Successfully processed {len(processed_documents)}/"
            f"{len(file_paths)} documents ({success_rate:.1f}%)[/green]"
        )
        
        return processed_documents

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
        """Demo function to show how to use the DocumentPreprocessor."""
        preprocessor = DocumentPreprocessor()
        
        # Process all PDFs in the raw data directory
        processed_docs = preprocessor.process_documents(
            extraction_method="pdfplumber"
        )
        
        if processed_docs:
            # Save processed documents
            output_file = preprocessor.save_processed_documents(processed_docs)
            
            # Show statistics
            stats = preprocessor.get_processing_stats(processed_docs)
            console.print("\n[bold]Processing Statistics:[/bold]")
            for key, value in stats.items():
                if isinstance(value, dict):
                    console.print(f"  {key}:")
                    for subkey, subvalue in value.items():
                        console.print(f"    {subkey}: {subvalue}")
                else:
                    console.print(f"  {key}: {value}")
            
            print(f"\nProcessed documents saved to: {output_file}")
        else:
            print("No documents were processed successfully")
    
    main()
