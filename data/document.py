from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json
import fitz  # PyMuPDF
from tqdm.auto import tqdm


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
        """Performs minor formatting on text."""
        cleaned_text = text.replace("\n", " ").strip()
        # Other potential text formatting functions can go here
        return cleaned_text


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
        self.pages_and_texts: List[Dict[str, Any]] = []
    
    def load(self) -> None:
        """Load PDF document content page by page."""
        doc = fitz.open(self.file_path)
        self.pages_and_texts = []
        
        for page_number, page in tqdm(enumerate(doc), desc="Processing PDF pages"):
            text = page.get_text()
            text = self.text_formatter(text)
            
            self.pages_and_texts.append({
                "page_number": page_number + self.page_offset,
                "page_char_count": len(text),
                "page_word_count": len(text.split(" ")),
                "page_sentence_count_raw": len(text.split(". ")),
                "page_token_count": len(text) / 4,  # 1 token = ~4 chars
                "text": text
            })
        
        doc.close()
        
        # Extract metadata
        total_chars = sum(page["page_char_count"] for page in self.pages_and_texts)
        total_words = sum(page["page_word_count"] for page in self.pages_and_texts)
        total_tokens = sum(page["page_token_count"] for page in self.pages_and_texts)
        
        self.metadata = {
            "file_type": "pdf",
            "file_path": self.file_path,
            "total_pages": len(self.pages_and_texts),
            "total_char_count": total_chars,
            "total_word_count": total_words,
            "total_token_count": total_tokens,
            "page_offset": self.page_offset
        }
    
    def get_text(self) -> str:
        """Extract all text content from PDF document."""
        if not self.pages_and_texts:
            self.load()
        
        all_text = " ".join(page["text"] for page in self.pages_and_texts)
        return self.text_formatter(all_text)
    
    def get_page_text(self, page_number: int) -> str:
        """Get text from a specific page."""
        if not self.pages_and_texts:
            self.load()
        
        for page in self.pages_and_texts:
            if page["page_number"] == page_number:
                return page["text"]
        
        raise ValueError(f"Page {page_number} not found")
    
    def get_pages_data(self) -> List[Dict[str, Any]]:
        """Get all pages data with statistics."""
        if not self.pages_and_texts:
            self.load()
        
        return self.pages_and_texts
    
    def get_page_range(self, start_page: int, end_page: int) -> str:
        """Get text from a range of pages."""
        if not self.pages_and_texts:
            self.load()
        
        texts = []
        for page in self.pages_and_texts:
            if start_page <= page["page_number"] <= end_page:
                texts.append(page["text"])
        
        return self.text_formatter(" ".join(texts))


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
