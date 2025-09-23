from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import List, Dict, Any, Optional, Iterable
import json
import fitz  # PyMuPDF
from tqdm.auto import tqdm


@dataclass
class DocumentPage:
    """Lightweight container for a single page of text and metadata."""

    page_number: int
    text: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.metadata)
        data.setdefault("page_number", self.page_number)
        data.setdefault("text", self.text)
        return data


class Document(ABC):
    """Abstract base class for document processing."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.content: Optional[Any] = None
        self.metadata: Dict[str, Any] = {}
    
    @abstractmethod
    def load(self) -> None:
        """Load the document content."""
        pass
    
    @abstractmethod
    def get_text(self) -> str:
        """Extract text content from the document."""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get document metadata."""
        return self.metadata
    
    def text_formatter(self, text: str) -> str:
        """Normalize whitespace and common PDF artifacts."""
        text = text.replace("\r", " ")
        text = re.sub(r"-\s*\n\s*", "", text)
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.replace("\n", " ").strip()

    def iter_pages(self) -> Iterable[DocumentPage]:
        """Optional iterator over document pages."""
        raise NotImplementedError("iter_pages is not implemented for this document type")


class JSONDocument(Document):
    """Document class for JSON files."""
    
    def load(self) -> None:
        """Load JSON document content."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.content = json.load(f)
        
        # Extract basic metadata
        self.metadata = {
            "file_type": "json",
            "file_path": self.file_path,
            "content_type": type(self.content).__name__
        }
    
    def get_text(self) -> str:
        """Extract text content from JSON document."""
        if self.content is None:
            self.load()
        
        if isinstance(self.content, str):
            return self.text_formatter(self.content)
        elif isinstance(self.content, dict):
            # Extract text from common JSON fields
            text_fields = ['text', 'content', 'body', 'description', 'summary']
            for field in text_fields:
                if field in self.content and isinstance(self.content[field], str):
                    return self.text_formatter(self.content[field])
            # If no text field found, convert entire dict to string
            return self.text_formatter(json.dumps(self.content, indent=2))
        elif isinstance(self.content, list):
            # Handle list of items
            texts = []
            for item in self.content:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    # Try to extract text from each item
                    for field in ['text', 'content', 'body', 'description', 'summary']:
                        if field in item and isinstance(item[field], str):
                            texts.append(item[field])
                            break
            return self.text_formatter(" ".join(texts))
        else:
            return self.text_formatter(str(self.content))


class PDFDocument(Document):
    """Document class for PDF files."""
    
    def __init__(self, file_path: str, page_offset: int = 0):
        super().__init__(file_path)
        self.page_offset = page_offset
        self.pages: List[DocumentPage] = []
    
    def load(self) -> None:
        """Load PDF document content page by page."""
        doc = fitz.open(self.file_path)
        self.pages = []
        
        for page_number, page in tqdm(enumerate(doc), desc="Processing PDF pages"):
            text = page.get_text()
            text = self.text_formatter(text)

            metadata = {
                "page_char_count": len(text),
                "page_word_count": len(text.split(" ")),
                "page_sentence_count_raw": len(text.split(". ")),
                "page_token_count": len(text) / 4,  # 1 token = ~4 chars
                "source": self.file_path,
            }

            self.pages.append(
                DocumentPage(
                    page_number=page_number + self.page_offset,
                    text=text,
                    metadata=metadata,
                )
            )
        
        doc.close()
        
        # Extract metadata
        total_chars = sum(page.metadata.get("page_char_count", 0) for page in self.pages)
        total_words = sum(page.metadata.get("page_word_count", 0) for page in self.pages)
        total_tokens = sum(page.metadata.get("page_token_count", 0) for page in self.pages)

        self.metadata = {
            "file_type": "pdf",
            "file_path": self.file_path,
            "total_pages": len(self.pages),
            "total_char_count": total_chars,
            "total_word_count": total_words,
            "total_token_count": total_tokens,
            "page_offset": self.page_offset
        }

    def get_text(self) -> str:
        """Extract all text content from PDF document."""
        if not self.pages:
            self.load()

        all_text = " ".join(page.text for page in self.pages)
        return self.text_formatter(all_text)
    
    def get_page_text(self, page_number: int) -> str:
        """Get text from a specific page."""
        if not self.pages:
            self.load()

        for page in self.pages:
            if page.page_number == page_number:
                return page.text
        
        raise ValueError(f"Page {page_number} not found")
    
    def get_pages_data(self) -> List[Dict[str, Any]]:
        """Get all pages data with statistics."""
        if not self.pages:
            self.load()

        return [page.to_dict() for page in self.pages]
    
    def get_page_range(self, start_page: int, end_page: int) -> str:
        """Get text from a range of pages."""
        if not self.pages:
            self.load()

        texts = []
        for page in self.pages:
            if start_page <= page.page_number <= end_page:
                texts.append(page.text)

        return self.text_formatter(" ".join(texts))

    def iter_pages(self) -> Iterable[DocumentPage]:
        if not self.pages:
            self.load()
        return iter(self.pages)


class TextDocument(Document):
    """Document class for plain text files."""
    
    def load(self) -> None:
        """Load text document content."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.content = f.read()
        
        # Extract metadata
        self.metadata = {
            "file_type": "text",
            "file_path": self.file_path,
            "char_count": len(self.content),
            "word_count": len(self.content.split()),
            "line_count": len(self.content.splitlines())
        }
    
    def get_text(self) -> str:
        """Extract text content from text document."""
        if self.content is None:
            self.load()
        
        return self.text_formatter(self.content)


def create_document(file_path: str, **kwargs) -> Document:
    """Factory function to create appropriate document type based on file extension."""
    file_ext = file_path.lower().split('.')[-1]
    
    if file_ext == 'pdf':
        return PDFDocument(file_path, **kwargs)
    elif file_ext == 'json':
        return JSONDocument(file_path)
    elif file_ext in ['txt', 'text']:
        return TextDocument(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")


# Example usage:
if __name__ == "__main__":
    # Example with PDF
    pdf_doc = PDFDocument("data/datasets/rag/human-nutrition-text.pdf", page_offset=-41)
    pdf_doc.load()
    print(f"PDF loaded: {pdf_doc.metadata['total_pages']} pages")
    
    # Example with JSON
    json_doc = JSONDocument("data/datasets/sft/instruction-data.json")
    json_doc.load()
    print(f"JSON loaded: {json_doc.metadata['content_type']}")
    
    # Example with factory function
    doc = create_document("some_file.pdf", page_offset=0)
    text = doc.get_text()
