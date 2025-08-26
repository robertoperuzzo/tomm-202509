#!/bin/bash

# Post-create script for the development container
echo "Running post-create setup..."

# Install Python dependencies if not already installed
pip install --user --upgrade pip
pip install --user -r requirements.txt

# Create project directory structure
mkdir -p data/{raw,processed,chunks}
mkdir -p src/{preprocessing,chunking,indexing,evaluation}
mkdir -p notebooks
mkdir -p tests
mkdir -p frontend
mkdir -p backend
mkdir -p logs

# Create .env file for environment variables
cat > .env << EOF
# Typesense configuration
TYPESENSE_HOST=typesense
TYPESENSE_PORT=8108
TYPESENSE_PROTOCOL=http
TYPESENSE_API_KEY=xyz

# Application configuration
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
FRONTEND_PORT=3000

# Data paths
DATA_RAW_PATH=./data/raw
DATA_PROCESSED_PATH=./data/processed
DATA_CHUNKS_PATH=./data/chunks

# Chunking configuration
CHUNK_SIZE_TOKENS=512
CHUNK_OVERLAP=50
SEMANTIC_THRESHOLD=0.7
EOF

# Create initial Python package structure
touch src/__init__.py
touch src/preprocessing/__init__.py
touch src/chunking/__init__.py
touch src/indexing/__init__.py
touch src/evaluation/__init__.py

# Create initial configuration files
cat > src/config.py << 'EOF'
"""Configuration settings for the chunking strategies demo."""

import os
from pathlib import Path
from typing import Dict, Any

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
DATA_CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks"
LOGS_PATH = PROJECT_ROOT / "logs"

# Typesense configuration
TYPESENSE_CONFIG = {
    "host": os.getenv("TYPESENSE_HOST", "localhost"),
    "port": int(os.getenv("TYPESENSE_PORT", "8108")),
    "protocol": os.getenv("TYPESENSE_PROTOCOL", "http"),
    "api_key": os.getenv("TYPESENSE_API_KEY", "xyz"),
}

# Chunking strategies configuration
CHUNKING_STRATEGIES = {
    "fixed_size": {
        "chunk_size": int(os.getenv("CHUNK_SIZE_TOKENS", "512")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "50")),
        "collection_name": "chunks_fixed_size"
    },
    "sliding_langchain": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "collection_name": "chunks_sliding_langchain"
    },
    "sliding_unstructured": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "collection_name": "chunks_sliding_unstructured"
    },
    "semantic": {
        "threshold": float(os.getenv("SEMANTIC_THRESHOLD", "0.7")),
        "collection_name": "chunks_semantic"
    }
}

# Collection schema for Typesense
COLLECTION_SCHEMA = {
    "name": "",  # Will be set dynamically
    "fields": [
        {"name": "id", "type": "string"},
        {"name": "paper_id", "type": "string", "facet": True},
        {"name": "title", "type": "string", "facet": True},
        {"name": "authors", "type": "string[]", "facet": True},
        {"name": "abstract", "type": "string"},
        {"name": "content", "type": "string"},
        {"name": "chunk_index", "type": "int32", "facet": True},
        {"name": "chunk_strategy", "type": "string", "facet": True},
        {"name": "chunk_size", "type": "int32"},
        {"name": "created_at", "type": "int64"},
    ]
}
EOF

echo "✅ Post-create setup completed successfully!"
echo ""
echo "📁 Created directory structure:"
echo "   - data/{raw,processed,chunks}"
echo "   - src/{preprocessing,chunking,indexing,evaluation}"
echo "   - notebooks, tests, frontend, backend, logs"
echo ""
echo "⚙️  Created configuration files:"
echo "   - .env (environment variables)"
echo "   - src/config.py (application configuration)"
echo ""
echo "🐍 Python dependencies installed"
echo ""
echo "🚀 Development environment is ready!"
