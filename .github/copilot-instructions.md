# Imtithal Project: AI Coding Guidelines

## Project Overview

**Imtithal** is a Retrieval-Augmented Generation (RAG) system designed to provide expert guidance on Saudi regulatory compliance, specifically covering SDAIA (Saudi Data and AI Authority) and NCA (National Cybersecurity Authority) policies. The system combines vector-based semantic search with generative AI to deliver accurate, context-grounded responses in Arabic.

## Architecture

### Core Components

1. **Vector Database Layer** (`src/utils.py`): ChromaDB persistent storage with multilingual embeddings
   - Model: `BAAI/bge-m3` (BGE-M3) - optimized for Arabic + multilingual support (~2.2GB)
   - Database: `./chroma_db/` with persistent client configuration
   - Collection: `imtithal_docs` - stores regulatory text chunks with source metadata

2. **ETL Pipeline** (`src/utils.py::index_files_with_llama`): Recursive document processing
   - **Parser**: LlamaParse (calls external API) - preserves table structures in Arabic docs
   - **Chunking**: 800-character chunks with 200-character overlap (preserves legal context across splits)
   - **Smart Indexing**: Skips previously indexed files by checking collection metadata

3. **RAG Query Engine** (`src/main.py::ask_imtithal`): Interactive compliance advisor
   - Encodes user query to embedding → semantic search (top 5 results) → prompt engineering → Gemini 3 Flash generation
   - Enforces strict context-grounding: model refuses to answer outside indexed knowledge base

4. **Data Sources**: Two regulatory domains
   - `./data/SDAIA/`: Data protection & AI governance regulations
   - `./data/NCA/`: Cybersecurity requirements & compliance frameworks

## Critical Workflows

### Initial Setup
```bash
# 1. Virtual environment activation
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables (create .env file)
GOOGLE_API_KEY=<your-gemini-api-key>
LLAMA_CLOUD_API_KEY=<your-llamaparse-api-key>

# 4. First run: index regulatory documents
python src/main.py  # Responds with prompt to index data/SDAIA and data/NCA
```

### Adding New Regulatory Documents
- Place PDFs/TXT files in `data/SDAIA/` or `data/NCA/`
- Run `python src/main.py` and select "y" when prompted to re-index
- System automatically detects new files; previously indexed files are skipped (by filename comparison)

### Debugging Vector Search Issues
- If answers seem off-topic: check chunk_size/overlap in `split_text()` - may be splitting legal clauses
- If Arabic text retrieval fails: verify `LlamaParse(language="ar")` setting and embedding model loaded
- Inspect ChromaDB directly: `collection.get()` shows all indexed documents with metadata

## Project-Specific Patterns

### Multilingual Handling
- **Arabic-first design**: All prompts, comments, and user-facing text use Arabic
- Use `BAAI/bge-m3` model for all embeddings (NOT standard English models like `all-MiniLM`)
- LlamaParse configured with `result_type="markdown"` + `language="ar"` to preserve Arabic table structures

### Metadata-Driven Context Grounding
- Every chunk stored in ChromaDB includes `{"source": filename, "category": folder_name}`
- System retrieves top 5 results and concatenates their raw text into context
- Model receives explicit instruction: *"Answer only from provided context; explicitly state if information unavailable"*

### Error Handling Philosophy
- All exceptions caught in `ask_imtithal()` return user-friendly Arabic error messages
- File indexing logs progress but doesn't halt on individual PDF parse failures (non-blocking)

## Integration Points & External Dependencies

| Service | Purpose | Configuration |
|---------|---------|----------------|
| **Google Gemini 3 Flash** | Response generation | API key in `.env`; model hardcoded as `"gemini-3-flash"` |
| **LlamaParse API** | PDF extraction with table preservation | API key in `.env`; Arabic language mode enabled |
| **ChromaDB** | Persistent vector storage | Path `./chroma_db/`; uses PersistentClient for durability |
| **Sentence Transformers** | Embeddings | BGE-M3 model (~2.2GB download on first run) |

## Code Conventions

- **No classes**: Functional style with utility functions; ChromaDB collection passed as parameter
- **Arabic comments throughout**: Explain "why" for regulatory context (e.g., why chunk_overlap=200)
- **Retry logic absent**: Relies on external API reliability; consider adding exponential backoff if timeouts occur
- **Single collection assumption**: Code assumes one `imtithal_docs` collection; multi-collection support not implemented

## Common Modification Points

1. **Change embedding model**: Update `SentenceTransformer()` model name in `src/utils.py`
2. **Adjust retrieval count**: Modify `n_results=5` in `ask_imtithal()` to retrieve more/fewer chunks
3. **Tune chunking strategy**: Edit `chunk_size=800, overlap=200` parameters
4. **Add new data sources**: Create subfolders in `data/` and update folder name expectations
5. **Switch LLM**: Replace `client.models.generate_content()` call; ensure API compatibility

---

**Last Updated**: January 2026  
**Created For**: GitHub Copilot / AI Agent Guidance
