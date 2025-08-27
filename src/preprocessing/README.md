# ArXiv Preprocessing Pipeline

This directory contains the implementation of the data preprocessing pipeline for ArXiv papers as described in the Demo Implementation Plan.

## Overview

The preprocessing pipeline performs the following steps:

1. **Data Collection**: Download ArXiv papers based on search queries (AI, CL, LG categories)
2. **Text Extraction**: Extract text content from PDF files using multiple extraction methods
3. **Text Cleaning**: Remove extra whitespace, page numbers, and formatting artifacts
4. **Metadata Extraction**: Extract title, authors, abstract, and publication information
5. **Data Storage**: Save processed papers as JSON files for use in chunking strategies

## Files

- `simple_processor.py`: Simplified processor for demo purposes with sample data
- `arxiv_preprocessor.py`: Full ArXiv API integration and PDF processing
- `__init__.py`: Module initialization

## Usage

### Quick Test with Sample Data

```bash
python3 test_preprocessing.py
```

This will:

- Create 3 sample papers with realistic content
- Process them through the preprocessing pipeline
- Save results to `data/processed/arxiv_processed_YYYYMMDD_HHMMSS.json`

### Sample Output Structure

Each processed paper has the following structure:

```json
{
  "paper_id": "sample_2023_001",
  "title": "Attention Is All You Need",
  "abstract": "The dominant sequence transduction models...",
  "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
  "full_text": "Abstract The dominant sequence transduction models...",
  "text_length": 874,
  "processed_at": "2025-08-27T09:07:35.983879"
}
```

## Key Features

### Multiple PDF Extraction Methods

- **pdfplumber**: Best for structured documents with tables
- **PyMuPDF**: Fast and robust for general text extraction
- **PyPDF2**: Fallback for compatibility

### Text Cleaning

- Remove excessive whitespace and newlines
- Clean page numbers and headers/footers
- Preserve paragraph structure
- Handle mathematical notation and special characters

### Error Handling

- Graceful fallback between extraction methods
- Detailed logging of processing errors
- Skip problematic files without stopping pipeline

## Integration with Chunking Strategies

The processed papers are saved in JSON format that can be directly consumed by the chunking strategies:

- **Fixed-size blocks**: Use `TokenTextSplitter` on `full_text`
- **Sliding windows (LangChain)**: Use `RecursiveCharacterTextSplitter`
- **Sliding windows (Unstructured)**: Use `partition_pdf()` with overlap
- **Semantic chunking**: Use `SemanticChunker` for natural breakpoints

## Data Flow

```
ArXiv API → PDF Download → Text Extraction → Text Cleaning → JSON Storage
     ↓
Chunking Strategies → Vector Indexing → Search Collections
```

## Requirements

The preprocessing pipeline requires these packages (already in requirements.txt):

- `aiohttp`: Async HTTP requests for ArXiv API
- `pdfplumber`: PDF text extraction
- `PyMuPDF`: Alternative PDF processing
- `PyPDF2`: Fallback PDF reader
- `rich`: Progress bars and console output
- `aiofiles`: Async file operations

## Next Steps

1. **Real ArXiv Integration**: Uncomment the full ArXiv API processor for production use
2. **Batch Processing**: Add support for processing large paper collections
3. **Metadata Enhancement**: Extract more paper metadata (references, figures, etc.)
4. **Quality Metrics**: Add text quality assessment scores
5. **Caching**: Implement smart caching to avoid re-processing papers
