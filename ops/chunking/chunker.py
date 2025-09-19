"""
Chunking strategies for text processing in RAG systems.

This module provides a clean, extensible architecture for implementing
different text chunking strategies using the Strategy pattern.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
from enum import Enum
import random
import textwrap
import os
import re
from pathlib import Path


class ChunkingStrategy(Enum):
    """Enumeration of available chunking strategies."""
    FIXED = "fixed"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    STRUCTURAL = "structural"
    LLM_BASED = "llm_based"


class Chunker(ABC):
    """
    Abstract base class for text chunking strategies.
    
    All chunking strategies must implement the chunk_text and chunk_pages methods.
    """
    
    def __init__(self, **kwargs):
        """Initialize the chunker with strategy-specific parameters."""
        self.config = kwargs
    
    @abstractmethod
    def chunk_text(self, text: str) -> List[str]:
        """
        Split a single text into chunks.
        
        Args:
            text: The text to be chunked
            
        Returns:
            List of text chunks
        """
        pass
    
    @abstractmethod
    def chunk_pages(self, pages_and_texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split pages of text into chunks.
        
        Args:
            pages_and_texts: List of dictionaries containing page data with 'text' key
            
        Returns:
            List of dictionaries containing chunk data with metadata
        """
        pass
    
    def _create_chunk_metadata(self, chunk: str, page_number: int, chunk_index: int) -> Dict[str, Any]:
        """
        Create standardized metadata for a chunk.
        
        Args:
            chunk: The chunk text
            page_number: Page number where chunk originated
            chunk_index: Index of chunk within the page
            
        Returns:
            Dictionary containing chunk metadata
        """
        return {
            "page_number": page_number,
            "chunk_index": chunk_index,
            "chunk_char_count": len(chunk),
            "chunk_word_count": len(chunk.split()),
            "chunk_token_count": len(chunk) / 4,  # rough token estimate
            "chunk_text": chunk
        }
    
    def _scattered_indices(self, n: int, k: int, jitter_frac: float = 0.08) -> List[int]:
        """Evenly spaced anchors + random jitter → indices scattered across [0, n-1]."""
        if k <= 0:
            return []
        if k == 1:
            return [random.randrange(n)]
        anchors = [int(round(i * (n - 1) / (k - 1))) for i in range(k)]
        out, seen = [], set()
        radius = max(1, int(n * jitter_frac))
        for a in anchors:
            lo, hi = max(0, a - radius), min(n - 1, a + radius)
            j = random.randint(lo, hi)
            if j not in seen:
                out.append(j)
                seen.add(j)
        while len(out) < k:
            r = random.randrange(n)
            if r not in seen:
                out.append(r)
                seen.add(r)
        return out
    
    def _draw_boxed_chunk(self, chunk: Dict[str, Any], wrap_at: int = 96) -> str:
        """Draw a boxed chunk with metadata header."""
        # Handle different chunk types (page chunks vs chapter chunks)
        if 'chapter_index' in chunk:
            # Chapter chunk format
            header = (
                f" Chapter {chunk['chapter_index']}  |  {chunk['title'][:120]}  "
                f"| p{chunk['page_start']}–{chunk['page_end']}  |  ~tokens {chunk['chunk_token_count']}"
            )
        else:
            # Page chunk format
            approx_tokens = chunk.get('chunk_token_count', len(chunk.get('chunk_text', '')) / 4)
            header = (
                f" Chunk p{chunk['page_number']} · idx {chunk['chunk_index']}  |  "
                f"chars {chunk['chunk_char_count']} · words {chunk['chunk_word_count']} · ~tokens {round(approx_tokens, 2)} "
            )
        
        # Wrap body text, avoid breaking long words awkwardly
        wrapped_lines = textwrap.wrap(
            chunk["chunk_text"], width=wrap_at, break_long_words=False, replace_whitespace=False
        )
        content_width = max([0, *map(len, wrapped_lines)])
        box_width = max(len(header), content_width + 2)  # +2 for side padding

        top    = "╔" + "═" * box_width + "╗"
        hline  = "║" + header.ljust(box_width) + "║"
        sep    = "╟" + "─" * box_width + "╢"
        body   = "\n".join("║ " + line.ljust(box_width - 2) + " ║" for line in wrapped_lines) or \
                 ("║ " + "".ljust(box_width - 2) + " ║")
        bottom = "╚" + "═" * box_width + "╝"
        return "\n".join([top, hline, sep, body, bottom])
    
    def show_random_chunks(self, pages_and_texts: List[Dict[str, Any]], k: int = 5, seed: int = None) -> None:
        """
        Show random scattered chunks from the given pages.
        
        Args:
            pages_and_texts: List of page dictionaries with 'text' key
            k: Number of random chunks to show
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
        
        chunks = self.chunk_pages(pages_and_texts)
        if not chunks:
            print("No chunks to display.")
            return
        
        idxs = self._scattered_indices(len(chunks), k)
        strategy_name = self.__class__.__name__.replace('Chunker', '').upper()
        print(f"Showing {len(idxs)} scattered random {strategy_name} chunks out of {len(chunks)} total:\n")
        
        for i, idx in enumerate(idxs, 1):
            print(f"#{i}")
            print(self._draw_boxed_chunk(chunks[idx]))
            print()
    
    def show_chunk_by_index(self, pages_and_texts: List[Dict[str, Any]], chunk_index: int) -> None:
        """
        Show a specific chunk by index.
        
        Args:
            pages_and_texts: List of page dictionaries with 'text' key
            chunk_index: Index of the chunk to display
        """
        chunks = self.chunk_pages(pages_and_texts)
        if not chunks:
            print("No chunks to display.")
            return
        
        if chunk_index < 0 or chunk_index >= len(chunks):
            print(f"Chunk index {chunk_index} is out of range. Valid range: 0-{len(chunks)-1}")
            return
        
        strategy_name = self.__class__.__name__.replace('Chunker', '').upper()
        print(f"Showing {strategy_name} chunk #{chunk_index} out of {len(chunks)} total:\n")
        print(self._draw_boxed_chunk(chunks[chunk_index]))
    
    def get_chunk_statistics(self, pages_and_texts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the chunks created from the given pages.
        
        Args:
            pages_and_texts: List of page dictionaries with 'text' key
            
        Returns:
            Dictionary containing chunk statistics
        """
        chunks = self.chunk_pages(pages_and_texts)
        if not chunks:
            return {"total_chunks": 0}
        
        chunk_sizes = [len(chunk['chunk_text']) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "strategy": self.__class__.__name__.replace('Chunker', '').upper()
        }


class FixedChunker(Chunker):
    """
    Fixed-size chunking strategy that splits text into chunks of approximately
    the same character length.
    """
    
    def __init__(self, chunk_size: int = 500, **kwargs):
        """
        Initialize fixed chunker.
        
        Args:
            chunk_size: Target size for each chunk in characters
        """
        super().__init__(chunk_size=chunk_size, **kwargs)
        self.chunk_size = chunk_size
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks of approximately chunk_size characters.
        
        Args:
            text: The text to be chunked
            
        Returns:
            List of text chunks
        """
        chunks = []
        current_chunk = ''
        words = text.split()

        for word in words:
            # Check if adding the word exceeds chunk size
            if len(current_chunk) + len(word) + 1 <= self.chunk_size:
                current_chunk += (word + ' ')
            else:
                # Store current chunk and start new one
                chunks.append(current_chunk.strip())
                current_chunk = word + ' '

        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
    
    def chunk_pages(self, pages_and_texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split pages of text into fixed-size chunks.
        
        Args:
            pages_and_texts: List of dictionaries containing page data
            
        Returns:
            List of dictionaries containing chunk data with metadata
        """
        all_chunks = []
        for page in pages_and_texts:
            page_number = page["page_number"]
            page_text = page["text"]

            chunks = self.chunk_text(page_text)
            for i, chunk in enumerate(chunks):
                chunk_metadata = self._create_chunk_metadata(chunk, page_number, i)
                all_chunks.append(chunk_metadata)
        
        return all_chunks


class SemanticChunker(Chunker):
    """
    Semantic chunking strategy that groups text based on semantic similarity.
    """
    
    def __init__(self, similarity_threshold: float = 0.8, max_tokens: int = 500, **kwargs):
        """
        Initialize semantic chunker.
        
        Args:
            similarity_threshold: Threshold for semantic similarity
            max_tokens: Maximum tokens per chunk
        """
        super().__init__(similarity_threshold=similarity_threshold, max_tokens=max_tokens, **kwargs)
        self.similarity_threshold = similarity_threshold
        self.max_tokens = max_tokens
        self._semantic_model = None
    
    def _get_semantic_model(self):
        """Lazy load semantic model."""
        if self._semantic_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                raise ImportError("sentence-transformers is required for semantic chunking")
        return self._semantic_model
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into semantic chunks based on sentence similarity.
        
        Args:
            text: The text to be chunked
            
        Returns:
            List of text chunks
        """
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            sent_tokenize = nltk.sent_tokenize
        except ImportError:
            sent_tokenize = lambda s: [seg.strip() for seg in re.split(r"(?<=[.!?])\s+", s) if seg.strip()] or [s]

        sentences = sent_tokenize(text)
        if not sentences:
            return []

        semantic_model = self._get_semantic_model()

        try:
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError as exc:
            raise ImportError("scikit-learn is required for semantic chunking") from exc

        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("numpy is required for semantic chunking") from exc

        embeddings = semantic_model.encode(sentences)

        chunks: List[str] = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]

        for i in range(1, len(sentences)):
            similarity = float(cosine_similarity([current_embedding], [embeddings[i]])[0][0])
            chunk_token_count = len(" ".join(current_chunk)) // 4

            if similarity >= self.similarity_threshold and chunk_token_count < self.max_tokens:
                current_chunk.append(sentences[i])
                current_embedding = np.mean([current_embedding, embeddings[i]], axis=0)
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
                current_embedding = embeddings[i]

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
    
    def chunk_pages(self, pages_and_texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split pages of text into semantic chunks.
        
        Args:
            pages_and_texts: List of dictionaries containing page data
            
        Returns:
            List of dictionaries containing chunk data with metadata
        """
        all_chunks = []
        for page in pages_and_texts:
            page_number = page["page_number"]
            page_text = page["text"]

            chunks = self.chunk_text(page_text)
            for i, chunk in enumerate(chunks):
                chunk_metadata = self._create_chunk_metadata(chunk, page_number, i)
                all_chunks.append(chunk_metadata)
        
        return all_chunks


class RecursiveChunker(Chunker):
    """
    Recursive chunking strategy that tries multiple splitting methods
    in order of preference.
    """
    
    def __init__(self, max_chunk_size: int = 1000, min_chunk_size: int = 100, **kwargs):
        """
        Initialize recursive chunker.
        
        Args:
            max_chunk_size: Maximum size for chunks
            min_chunk_size: Minimum size for chunks
        """
        super().__init__(max_chunk_size=max_chunk_size, min_chunk_size=min_chunk_size, **kwargs)
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Recursively split text into chunks using multiple strategies.
        
        Args:
            text: The text to be chunked
            
        Returns:
            List of text chunks
        """
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            sent_tokenize = nltk.sent_tokenize
        except ImportError:
            sent_tokenize = lambda s: [seg.strip() for seg in re.split(r"(?<=[.!?])\s+", s) if seg.strip()] or [s]

        def split_chunk(chunk: str) -> List[str]:
            if len(chunk) <= self.max_chunk_size:
                return [chunk]

            sections = [section.strip() for section in chunk.split('\n\n') if section.strip()]
            if len(sections) > 1:
                result: List[str] = []
                for section in sections:
                    result.extend(split_chunk(section))
                return result

            sections = [section.strip() for section in chunk.split('\n') if section.strip()]
            if len(sections) > 1:
                result: List[str] = []
                for section in sections:
                    result.extend(split_chunk(section))
                return result

            sentences = sent_tokenize(chunk)
            if len(sentences) > 1:
                chunks: List[str] = []
                current_chunk: List[str] = []
                current_size = 0

                for sentence in sentences:
                    if current_size + len(sentence) > self.max_chunk_size and current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = [sentence]
                        current_size = len(sentence)
                    else:
                        current_chunk.append(sentence)
                        current_size += len(sentence)

                if current_chunk:
                    chunks.append(" ".join(current_chunk))

                return chunks

            words = chunk.split()
            if len(words) > 1:
                mid = len(words) // 2
                left_chunk = ' '.join(words[:mid])
                right_chunk = ' '.join(words[mid:])
                result: List[str] = []
                result.extend(split_chunk(left_chunk))
                result.extend(split_chunk(right_chunk))
                return result

            return [chunk]

        return split_chunk(text)
    
    def chunk_pages(self, pages_and_texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split pages of text into recursive chunks.
        
        Args:
            pages_and_texts: List of dictionaries containing page data
            
        Returns:
            List of dictionaries containing chunk data with metadata
        """
        all_chunks = []
        for page in pages_and_texts:
            page_number = page["page_number"]
            page_text = page["text"]

            chunks = self.chunk_text(page_text)
            for i, chunk in enumerate(chunks):
                chunk_metadata = self._create_chunk_metadata(chunk, page_number, i)
                all_chunks.append(chunk_metadata)
        
        return all_chunks


class StructuralChunker(Chunker):
    """
    Structural chunking strategy that splits text based on document structure
    (e.g., chapters, sections).
    """
    
    def __init__(self, **kwargs):
        """Initialize structural chunker."""
        super().__init__(**kwargs)
    
    HEADER_PATTERN = re.compile(r"university\s+of\s+hawai", flags=re.IGNORECASE)

    def chunk_text(self, text: str) -> List[str]:
        """Expose chapter chunk text by delegating to chunk_pages."""
        chunks = self.chunk_pages([
            {
                "page_number": 0,
                "text": text,
            }
        ])
        return [chunk["chunk_text"] for chunk in chunks]
    
    def chunk_pages(self, pages_and_texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split pages of text into structural chunks.
        
        Args:
            pages_and_texts: List of dictionaries containing page data
            
        Returns:
            List of dictionaries containing chunk data with metadata
        """
        if not pages_and_texts:
            return []

        chapter_starts = [
            idx for idx, page in enumerate(pages_and_texts)
            if self._is_chapter_header_page(page["text"])
        ]

        if not chapter_starts:
            all_text = " ".join(page["text"] for page in pages_and_texts).strip()
            title = self._guess_title_from_page(pages_and_texts[0]["text"])
            return [{
                "chapter_index": 0,
                "title": title,
                "page_start": pages_and_texts[0]["page_number"],
                "page_end": pages_and_texts[-1]["page_number"],
                "chunk_char_count": len(all_text),
                "chunk_word_count": len(all_text.split()),
                "chunk_token_count": round(len(all_text) / 4, 2),
                "chunk_text": all_text,
            }]

        chapter_chunks: List[Dict[str, Any]] = []
        for chapter_index, start_idx in enumerate(chapter_starts):
            end_idx = chapter_starts[chapter_index + 1] - 1 if chapter_index + 1 < len(chapter_starts) else len(pages_and_texts) - 1
            if end_idx < start_idx:
                continue

            chapter_pages = pages_and_texts[start_idx:end_idx + 1]
            text_concat = " ".join(page["text"] for page in chapter_pages).strip()
            title = self._guess_title_from_page(chapter_pages[0]["text"])

            chapter_chunks.append({
                "chapter_index": chapter_index,
                "title": title,
                "page_start": chapter_pages[0]["page_number"],
                "page_end": chapter_pages[-1]["page_number"],
                "chunk_char_count": len(text_concat),
                "chunk_word_count": len(text_concat.split()),
                "chunk_token_count": round(len(text_concat) / 4, 2),
                "chunk_text": text_concat,
            })

        return chapter_chunks

    @classmethod
    def _is_chapter_header_page(cls, text: str) -> bool:
        return bool(cls.HEADER_PATTERN.search(text))

    @staticmethod
    def _guess_title_from_page(text: str) -> str:
        match = StructuralChunker.HEADER_PATTERN.search(text)
        if match:
            title = text[:match.start()].strip()
            title = re.sub(r"\s+", " ", title).strip()
            if 10 <= len(title) <= 180:
                return title

        condensed = re.sub(r"\s+", " ", text).strip()
        return condensed[:120] if condensed else "Untitled Chapter"


class LLMBasedChunker(Chunker):
    """
    LLM-based chunking strategy that uses language models to find
    semantically coherent chunk boundaries.
    """
    
    def __init__(self, chunk_size: int = 1000, model: str = "gpt-4o-mini", api_key: str = None, **kwargs):
        """
        Initialize LLM-based chunker.

        Args:
            chunk_size: Target size for chunks
            model: LLM model to use for chunking
            api_key: OpenAI API key. If None, will try to get from OPENAI_API_KEY env var
        """
        super().__init__(chunk_size=chunk_size, model=model, **kwargs)
        self.chunk_size = chunk_size
        self.model = model
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            self.api_key = self._load_api_key_from_dotenv()
            if self.api_key and "OPENAI_API_KEY" not in os.environ:
                os.environ["OPENAI_API_KEY"] = self.api_key
        
        # Initialize OpenAI client
        try:
            from openai import OpenAI
            try:
                self.client = OpenAI(api_key=self.api_key) if self.api_key else OpenAI()
            except Exception as exc:
                print(f"Warning: OpenAI client initialization failed: {exc}. Using fallback chunking.")
                self.client = None
        except ImportError:
            raise ImportError(
                "OpenAI package is required for LLM-based chunking. "
                "Please install it with: pip install openai"
            )
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks using LLM-based boundary detection.
        
        Args:
            text: The text to be chunked
            
        Returns:
            List of text chunks
        """
        def get_chunk_boundary(text_segment: str) -> int:
            """
            Ask the LLM where to split within this text segment.
            Returns an index (int) within text_segment.
            """
            prompt = f"""
Analyze the following text and identify the best point to split it
into two semantically coherent parts.
The split should occur near {self.chunk_size} characters.

Text:
\"\"\"{text_segment}\"\"\"

Return only the integer index (character position) within this text
where the split should occur. Do not return any explanation.
"""

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a text analysis expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
                
                # Extract and sanitize
                split_str = response.choices[0].message.content.strip()
                try:
                    split_point = int(split_str)
                except ValueError:
                    split_point = self.chunk_size
                return split_point
            except Exception as e:
                # If API call fails, fallback to chunk_size
                print(f"Warning: LLM API call failed: {e}. Using fallback chunking.")
                return self.chunk_size

        chunks = []
        remaining_text = text

        while len(remaining_text) > self.chunk_size:
            text_window = remaining_text[:self.chunk_size * 2]
            split_point = get_chunk_boundary(text_window)

            # Safety check
            if split_point < 100 or split_point > len(text_window) - 100:
                split_point = self.chunk_size

            chunks.append(remaining_text[:split_point].strip())
            remaining_text = remaining_text[split_point:].strip()

        if remaining_text:
            chunks.append(remaining_text)

        return chunks
    
    def chunk_pages(self, pages_and_texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split pages of text into LLM-based chunks.
        
        Args:
            pages_and_texts: List of dictionaries containing page data
            
        Returns:
            List of dictionaries containing chunk data with metadata
        """
        all_chunks = []
        for page in pages_and_texts:
            page_number = page["page_number"]
            page_text = page["text"]

            chunks = self.chunk_text(page_text)
            for i, chunk in enumerate(chunks):
                chunk_metadata = self._create_chunk_metadata(chunk, page_number, i)
                all_chunks.append(chunk_metadata)
        
        return all_chunks

    @staticmethod
    def _load_api_key_from_dotenv() -> str:
        """Attempt to read OPENAI_API_KEY from nearby .env files."""
        candidates = []
        current_file = Path(__file__).resolve()
        candidates.append(Path.cwd())
        candidates.extend(current_file.parents)

        seen = set()
        for directory in candidates:
            if directory in seen:
                continue
            seen.add(directory)
            env_path = directory / ".env"
            if not env_path.is_file():
                continue
            try:
                for line in env_path.read_text().splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith('#'):
                        continue
                    if '=' not in stripped:
                        continue
                    key, value = stripped.split('=', 1)
                    if key.strip() == "OPENAI_API_KEY":
                        cleaned = value.strip().strip('"').strip("'")
                        if cleaned:
                            return cleaned
            except OSError:
                continue
        return ""


class ChunkerFactory:
    """
    Factory class for creating chunker instances based on strategy.
    """
    
    _chunkers = {
        ChunkingStrategy.FIXED: FixedChunker,
        ChunkingStrategy.SEMANTIC: SemanticChunker,
        ChunkingStrategy.RECURSIVE: RecursiveChunker,
        ChunkingStrategy.STRUCTURAL: StructuralChunker,
        ChunkingStrategy.LLM_BASED: LLMBasedChunker,
    }
    
    @classmethod
    def create_chunker(cls, strategy: Union[ChunkingStrategy, str], **kwargs) -> Chunker:
        """
        Create a chunker instance based on the specified strategy.
        
        Args:
            strategy: The chunking strategy to use
            **kwargs: Additional parameters for the chunker
            
        Returns:
            Chunker instance
            
        Raises:
            ValueError: If strategy is not supported
        """
        if isinstance(strategy, str):
            try:
                strategy = ChunkingStrategy(strategy)
            except ValueError:
                raise ValueError(f"Unknown strategy: {strategy}")
        
        if strategy not in cls._chunkers:
            raise ValueError(f"Strategy {strategy} not supported")
        
        chunker_class = cls._chunkers[strategy]
        return chunker_class(**kwargs)
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """
        Get list of available chunking strategies.
        
        Returns:
            List of strategy names
        """
        return [strategy.value for strategy in ChunkingStrategy]


# Convenience function for easy usage
def create_chunker(strategy: Union[ChunkingStrategy, str], **kwargs) -> Chunker:
    """
    Convenience function to create a chunker instance.
    
    Args:
        strategy: The chunking strategy to use
        **kwargs: Additional parameters for the chunker
        
    Returns:
        Chunker instance
    """
    return ChunkerFactory.create_chunker(strategy, **kwargs)
