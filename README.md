# Chunking Strategies Demo - Development Environment

This project demonstrates various chunking strategies for improving vector indexing and retrieval quality in AI-powered search systems.

## 🚀 Getting Started

### Prerequisites

- [VS Code](https://code.visualstudio.com/) with [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- [Docker](https://www.docker.com/get-started) and [Docker Compose](https://docs.docker.com/compose/install/)

### Setup Development Environment

1. **Open in Dev Container**:

   ```bash
   # Clone the repository (if not already done)
   git clone <repository-url>
   cd tomm-202509

   # Open in VS Code
   code .
   ```

2. **Launch Dev Container**:

   - VS Code will detect the `.devcontainer` configuration
   - Click "Reopen in Container" when prompted
   - Or use Command Palette: `Dev Containers: Reopen in Container`

3. **Start Services**:

   ````bash
   # Start Typesense service
   docker compose up -d typesense

   # Check if Typesense is running
   curl http://localhost:8108/health
   ```### Project Structure
   ````

```
├── .devcontainer/          # VS Code Dev Container configuration
│   ├── devcontainer.json   # Container configuration
│   ├── Dockerfile          # Development container image
│   └── post-create.sh      # Setup script
├── data/                   # Data storage
│   ├── raw/               # Raw ArXiv papers (PDFs)
│   ├── processed/         # Preprocessed papers (JSON)
│   └── chunks/            # Generated chunks
├── src/                   # Source code
│   ├── preprocessor/      # Data preprocessing modules
│   ├── chunker/          # Chunking strategy implementations
│   ├── indexer/          # Typesense indexing utilities
│   ├── evaluation/       # Evaluation and metrics
│   └── config.py         # Configuration settings
├── tests/               # Unit tests
├── frontend/            # HTML/CSS/JavaScript with InstantSearch.js
├── backend/            # FastAPI backend (to be created)
└── docker-compose.yml   # Service orchestration
```

## 🧪 Development Workflow

### 1. Data Preparation

- Download ArXiv papers to `data/raw/`
- Run preprocessor scripts to extract text and metadata using two methods:
  - **pypdf**: Fast extraction using LangChain's PyPDFParser
  - **unstructured**: Premium quality with structure awareness
- Store processed data in `data/processed/`

### 2. Chunker Strategies

Implement and test different chunker approaches:

- **Fixed-size blocks**: Token-aware splitting using LangChain's `TokenTextSplitter`
- **Sliding windows (LangChain)**: Document-structure aware splitting with `RecursiveCharacterTextSplitter`
- **Sliding windows (Unstructured)**: Element-based chunker with custom overlap
- **Semantic chunker**: Natural breakpoint identification using `SemanticChunker`

### 3. Indexer and Search

- Create separate Typesense collections for each strategy
- Index chunks with metadata and search capabilities
- Implement comparison queries across strategies

### 4. Evaluation

- Define test query sets for different search scenarios
- Measure relevance scores, search speed, and chunk quality
- Generate comparative analysis reports

## 🛠️ Available Tools and Libraries

### Python Dependencies

- **Unstructured**: Premium PDF processing and element-based chunking
- **LangChain**: Text splitters, semantic chunking, and PDF parsing (PyPDFParser)
- **Typesense**: Vector indexing and search engine
- **FastAPI**: Backend API framework

### Development Tools

- **Black**: Code formatting
- **Flake8/Pylint**: Code linting
- **pytest**: Testing framework

## 🎯 Next Implementation Steps

1. ✅ **Set up VS Code Dev Container** (COMPLETED)
2. 🔄 **Create Docker Compose configuration** (COMPLETED)
3. ⏳ **Implement data preprocessing pipeline**
4. ⏳ **Prototype chunking strategies**
5. ⏳ **Develop strategy comparison UI**
6. ⏳ **Integrate evaluation metrics**

## 📊 Services

### Typesense Search Engine

- **URL**: http://localhost:8108
- **API Key**: `xyz` (development only)
- **Health Check**: `curl http://localhost:8108/health`

### Backend API (Future)

- **URL**: http://localhost:8000
- **Framework**: FastAPI with automatic OpenAPI docs

### Frontend (Future)

- **URL**: http://localhost:3000
- **Framework**: Plain HTML/CSS/JavaScript with InstantSearch.js

## 🐛 Troubleshooting

### Container Issues

```bash
# Rebuild dev container
# Command Palette: "Dev Containers: Rebuild Container"

# Check container logs
docker compose logs dev

# Reset everything
docker compose down -v
docker system prune -f
```

### Typesense Issues

```bash
# Check Typesense status
docker compose logs typesense

# Reset Typesense data
docker compose down
docker volume rm tomm-202509_typesense-data
docker compose up -d typesense
```

### Python Dependencies

```bash
# Reinstall dependencies
pip install -r requirements.txt

# Update packages
pip install --upgrade -r requirements.txt
```

## 📚 Documentation

- **[Architecture Decision Records (ADR)](docs/adr/README.md)**: Key architectural decisions and their rationale
- **[Implementation Progress](docs/adr/)**: Detailed documentation of system components and design choices

## 📚 Resources

- [Unstructured Documentation](https://unstructured-io.github.io/unstructured/)
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Typesense Documentation](https://typesense.org/docs/)
- [InstantSearch.js Guide](https://www.algolia.com/doc/guides/building-search-ui/what-is-instantsearch/js/)

---

**Ready to start developing!** 🎉
