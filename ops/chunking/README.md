# Chunker System

A clean, extensible architecture for implementing different text chunking strategies using the Strategy pattern.

## Overview

This module provides a comprehensive chunking system that supports multiple strategies for splitting text into meaningful chunks for RAG (Retrieval-Augmented Generation) systems. Each strategy is designed to handle different types of content and use cases.

## Features

- **Strategy Pattern**: Clean, extensible architecture
- **Multiple Chunking Strategies**: Fixed, Recursive, Structural, Semantic, and LLM-based
- **Consistent Interface**: All chunkers implement the same base interface
- **Rich Metadata**: Each chunk includes character count, word count, token count, and source information
- **Visual Display**: Pretty-printed chunk display with boxed formatting
- **Factory Pattern**: Easy instantiation of chunkers
- **Comprehensive Testing**: Full test suite with notebook compatibility

## Available Strategies

### 1. Fixed Chunking (`FixedChunker`)
Splits text into chunks of approximately the same character length.

**Use Case**: When you need consistent chunk sizes regardless of content structure.

**Parameters**:
- `chunk_size` (int): Target size for each chunk in characters (default: 500)

### 2. Recursive Chunking (`RecursiveChunker`)
Recursively splits text using multiple strategies in order of preference.

**Use Case**: When you want to respect natural boundaries (sections → paragraphs → sentences → words).

**Parameters**:
- `max_chunk_size` (int): Maximum size for chunks (default: 1000)
- `min_chunk_size` (int): Minimum size for chunks (default: 100)

### 3. Structural Chunking (`StructuralChunker`)
Groups text based on document structure (chapters, sections).

**Use Case**: When working with structured documents like textbooks or reports.

**Parameters**: None (automatically detects structure)

### 4. Semantic Chunking (`SemanticChunker`)
Groups text based on semantic similarity using sentence embeddings.

**Use Case**: When you want to keep semantically related content together.

**Parameters**:
- `similarity_threshold` (float): Threshold for semantic similarity (default: 0.8)
- `max_tokens` (int): Maximum tokens per chunk (default: 500)

**Dependencies**: `sentence-transformers`, `nltk`

### 5. LLM-Based Chunking (`LLMBasedChunker`)
Uses language models to find semantically coherent chunk boundaries.

**Use Case**: When you want the most sophisticated chunking with LLM intelligence.

**Parameters**:
- `chunk_size` (int): Target size for chunks (default: 1000)
- `model` (str): LLM model to use (default: "gpt-4o-mini")
- `api_key` (str): OpenAI API key (optional, can use OPENAI_API_KEY env var)

**Dependencies**: `openai`

**Note**: Requires a valid OpenAI API key. Will fail with a clear error message if no API key is provided.

## Usage

### Basic Usage

```python
from ops.chunking.chunker import create_chunker

# Create a chunker
chunker = create_chunker("fixed", chunk_size=500)

# Chunk text
chunks = chunker.chunk_text("Your text here...")

# Chunk pages
page_chunks = chunker.chunk_pages(pages_data)
```

### Using the Factory

```python
from ops.chunking.chunker import ChunkerFactory, ChunkingStrategy

# Using string strategy
chunker = ChunkerFactory.create_chunker("fixed", chunk_size=200)

# Using enum strategy
chunker = ChunkerFactory.create_chunker(ChunkingStrategy.RECURSIVE, max_chunk_size=800)

# Get available strategies
strategies = ChunkerFactory.get_available_strategies()
```

### Displaying Chunks

```python
# Show random chunks
chunker.show_random_chunks(pages_data, k=5, seed=42)

# Show specific chunk by index
chunker.show_chunk_by_index(pages_data, chunk_index=10)

# Get chunk statistics
stats = chunker.get_chunk_statistics(pages_data)
print(f"Total chunks: {stats['total_chunks']}")
print(f"Average size: {stats['avg_chunk_size']:.1f} characters")
```

## Demo Scripts

### Full Demo
```bash
python demo_chunkers.py
```
Shows comprehensive comparison of all strategies with statistics.

### Simple Demo
```bash
python demo_chunkers_simple.py
```
Shows random chunks from each strategy in a clean format.

## Chunk Metadata

Each chunk includes the following metadata:

```python
{
    "page_number": int,           # Page number where chunk originated
    "chunk_index": int,           # Index of chunk within the page
    "chunk_char_count": int,      # Character count
    "chunk_word_count": int,      # Word count
    "chunk_token_count": float,   # Estimated token count (chars/4)
    "chunk_text": str             # The actual chunk text
}
```

For structural chunks (chapters), additional metadata is included:

```python
{
    "chapter_index": int,         # Chapter number
    "title": str,                 # Chapter title
    "page_start": int,            # Starting page
    "page_end": int,              # Ending page
    # ... plus standard metadata
}
```

## Testing

The module includes comprehensive tests:

```bash
# Run all tests
python -m pytest tests/test_chunkers.py -v

# Run notebook compatibility tests
python -m pytest tests/test_chunker_notebook_compatibility.py -v

# Run PDF-based tests
python -m pytest tests/test_chunker_with_pdf.py -v
```

## Dependencies

### Required
- `textwrap` (built-in)
- `random` (built-in)
- `typing` (built-in)

### Optional
- `sentence-transformers` (for semantic chunking)
- `nltk` (for semantic chunking)
- `PyMuPDF` (for PDF processing in demos)
- `tqdm` (for progress bars in demos)

## Examples

### Fixed Chunking Example
```python
chunker = create_chunker("fixed", chunk_size=200)
chunks = chunker.chunk_text("This is a long text that will be split into chunks of approximately 200 characters each.")
# Results in multiple chunks of ~200 characters
```

### Recursive Chunking Example
```python
chunker = create_chunker("recursive", max_chunk_size=500, min_chunk_size=100)
chunks = chunker.chunk_text("""
Section 1: Introduction

This is the first section with multiple paragraphs.

Section 2: Methods

This is the second section with different content.
""")
# Respects section and paragraph boundaries
```

### Structural Chunking Example
```python
chunker = create_chunker("structural")
chunks = chunker.chunk_pages([
    {"page_number": 1, "text": "Chapter 1: Introduction\n\nThis is chapter content."},
    {"page_number": 2, "text": "More chapter content."}
])
# Groups content by chapters
```

## Performance Comparison

Based on the human nutrition PDF (1,208 pages):

| Strategy | Chunks | Avg Size | Min Size | Max Size |
|----------|--------|----------|----------|----------|
| Fixed | 3,321 | 406.9 | 4 | 499 |
| Recursive | 2,024 | 672.3 | 0 | 1,000 |
| Structural | 43 | 32,264.9 | 797 | 170,939 |
| LLM-Based | 1,932 | 700.2 | 3 | 1,000 |

## Contributing

To add a new chunking strategy:

1. Create a new class inheriting from `Chunker`
2. Implement `chunk_text()` and `chunk_pages()` methods
3. Add the strategy to `ChunkingStrategy` enum
4. Register it in `ChunkerFactory._chunkers`
5. Add tests for the new strategy

## License

This module is part of the open-slm-agents project.
