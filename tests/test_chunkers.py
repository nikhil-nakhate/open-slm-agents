"""
Test chunker compatibility with notebook examples.

This test verifies that our chunker implementations produce the same results
as the examples shown in the RAG Chunking Strategies notebook.
"""

import unittest
from ops.chunking.chunker import create_chunker, ChunkingStrategy


class TestNotebookCompatibility(unittest.TestCase):
    """Test compatibility with notebook examples."""
    
    def setUp(self):
        """Set up test data exactly as shown in the notebook."""
        # Exact data from notebook output
        self.notebook_pages = [
            {
                'page_number': -41,
                'page_char_count': 29,
                'page_word_count': 4,
                'page_sentence_count_raw': 1,
                'page_token_count': 7.25,
                'text': 'Human Nutrition: 2020 Edition'
            },
            {
                'page_number': -40,
                'page_char_count': 0,
                'page_word_count': 1,
                'page_sentence_count_raw': 1,
                'page_token_count': 0.0,
                'text': ''
            }
        ]
        
        # Expected results from notebook
        self.expected_fixed_chunks = {
            'total_chunks': 1,  # Only non-empty page should produce chunks
            'first_chunk': {
                'page_number': -41,
                'chunk_text': 'Human Nutrition: 2020 Edition',
                'chunk_char_count': 29,
                'chunk_word_count': 4,
                'chunk_token_count': 7.25
            }
        }
    
    def test_fixed_chunker_notebook_compatibility(self):
        """Test FixedChunker produces same results as notebook example."""
        # Use same chunk_size as notebook (500)
        chunker = create_chunker("fixed", chunk_size=500)
        chunks = chunker.chunk_pages(self.notebook_pages)
        
        # Verify total chunks
        self.assertEqual(len(chunks), self.expected_fixed_chunks['total_chunks'])
        
        # Verify first chunk matches exactly
        first_chunk = chunks[0]
        expected = self.expected_fixed_chunks['first_chunk']
        
        self.assertEqual(first_chunk['page_number'], expected['page_number'])
        self.assertEqual(first_chunk['chunk_text'], expected['chunk_text'])
        self.assertEqual(first_chunk['chunk_char_count'], expected['chunk_char_count'])
        self.assertEqual(first_chunk['chunk_word_count'], expected['chunk_word_count'])
        self.assertAlmostEqual(first_chunk['chunk_token_count'], expected['chunk_token_count'], places=1)
    
    def test_fixed_chunker_metadata_structure(self):
        """Test that chunk metadata structure matches notebook format."""
        chunker = create_chunker("fixed", chunk_size=500)
        chunks = chunker.chunk_pages(self.notebook_pages)
        
        for chunk in chunks:
            # Verify all required metadata fields are present
            required_fields = [
                'page_number', 'chunk_index', 'chunk_char_count',
                'chunk_word_count', 'chunk_token_count', 'chunk_text'
            ]
            for field in required_fields:
                self.assertIn(field, chunk, f"Missing required field: {field}")
            
            # Verify data types match notebook format
            self.assertIsInstance(chunk['page_number'], int)
            self.assertIsInstance(chunk['chunk_index'], int)
            self.assertIsInstance(chunk['chunk_char_count'], int)
            self.assertIsInstance(chunk['chunk_word_count'], int)
            self.assertIsInstance(chunk['chunk_token_count'], (int, float))
            self.assertIsInstance(chunk['chunk_text'], str)
    
    def test_fixed_chunker_token_calculation(self):
        """Test that token calculation matches notebook formula."""
        chunker = create_chunker("fixed", chunk_size=500)
        chunks = chunker.chunk_pages(self.notebook_pages)
        
        for chunk in chunks:
            # Verify token count calculation: len(text) / 4
            expected_tokens = len(chunk['chunk_text']) / 4
            self.assertAlmostEqual(chunk['chunk_token_count'], expected_tokens, places=1)
    
    def test_fixed_chunker_word_count_calculation(self):
        """Test that word count calculation matches notebook method."""
        chunker = create_chunker("fixed", chunk_size=500)
        chunks = chunker.chunk_pages(self.notebook_pages)
        
        for chunk in chunks:
            # Verify word count calculation: len(text.split())
            expected_words = len(chunk['chunk_text'].split())
            self.assertEqual(chunk['chunk_word_count'], expected_words)
    
    def test_fixed_chunker_empty_page_handling(self):
        """Test that empty pages are handled correctly as in notebook."""
        chunker = create_chunker("fixed", chunk_size=500)
        chunks = chunker.chunk_pages(self.notebook_pages)
        
        # Should not create chunks for empty pages
        empty_page_chunks = [c for c in chunks if c['page_number'] == -40]
        self.assertEqual(len(empty_page_chunks), 0)
    
    def test_fixed_chunker_chunk_indexing(self):
        """Test that chunk indexing is correct."""
        chunker = create_chunker("fixed", chunk_size=500)
        chunks = chunker.chunk_pages(self.notebook_pages)
        
        # Verify chunk indices are sequential starting from 0
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk['chunk_index'], i)
    
    def test_fixed_chunker_with_larger_text(self):
        """Test fixed chunker with larger text to verify chunking behavior."""
        # Create a larger text that will be split into multiple chunks
        large_text = "This is a test document with multiple sentences. " * 20  # ~1000 characters
        
        large_pages = [
            {
                'page_number': 1,
                'page_char_count': len(large_text),
                'page_word_count': len(large_text.split()),
                'page_sentence_count_raw': len(large_text.split('. ')),
                'page_token_count': len(large_text) / 4,
                'text': large_text
            }
        ]
        
        chunker = create_chunker("fixed", chunk_size=200)
        chunks = chunker.chunk_pages(large_pages)
        
        # Should create multiple chunks
        self.assertGreater(len(chunks), 1)
        
        # All chunks should be from the same page
        for chunk in chunks:
            self.assertEqual(chunk['page_number'], 1)
        
        # Verify chunk indices are sequential
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk['chunk_index'], i)
    
    def test_recursive_chunker_basic_functionality(self):
        """Test recursive chunker basic functionality matches expected behavior."""
        chunker = create_chunker("recursive", max_chunk_size=1000, min_chunk_size=100)
        chunks = chunker.chunk_pages(self.notebook_pages)
        
        # Should produce at least one chunk for non-empty page
        self.assertGreaterEqual(len(chunks), 1)
        
        # Verify metadata structure
        for chunk in chunks:
            self.assertIn('page_number', chunk)
            self.assertIn('chunk_index', chunk)
            self.assertIn('chunk_text', chunk)
            self.assertIn('chunk_char_count', chunk)
            self.assertIn('chunk_word_count', chunk)
            self.assertIn('chunk_token_count', chunk)
    
    def test_structural_chunker_basic_functionality(self):
        """Test structural chunker basic functionality."""
        # Create test data with chapter markers
        pages_with_chapters = [
            {
                'page_number': 1,
                'text': 'Chapter 1: Introduction\n\nThis is the first chapter content.'
            },
            {
                'page_number': 2,
                'text': 'This is more content from chapter 1.'
            }
        ]
        
        chunker = create_chunker("structural")
        chunks = chunker.chunk_pages(pages_with_chapters)
        
        # Should produce at least one chunk for pages with content
        self.assertGreaterEqual(len(chunks), 1)
        
        # Verify metadata structure (structural chunker uses different format for chapters)
        for chunk in chunks:
            self.assertIn('chunk_text', chunk)
            # Structural chunker produces chapter metadata, not page metadata
            if 'chapter_index' in chunk:
                self.assertIn('title', chunk)
                self.assertIn('page_start', chunk)
                self.assertIn('page_end', chunk)
                self.assertIn('chunk_char_count', chunk)
                self.assertIn('chunk_word_count', chunk)
                self.assertIn('chunk_token_count', chunk)
            else:
                # For non-chapter chunks, expect page metadata
                self.assertIn('page_number', chunk)
    
    def test_llm_based_chunker_basic_functionality(self):
        """Test LLM-based chunker basic functionality."""
        chunker = create_chunker("llm_based", chunk_size=1000)
        chunks = chunker.chunk_pages(self.notebook_pages)
        
        # Should produce at least one chunk for non-empty page
        self.assertGreaterEqual(len(chunks), 1)
        
        # Verify metadata structure
        for chunk in chunks:
            self.assertIn('page_number', chunk)
            self.assertIn('chunk_index', chunk)
            self.assertIn('chunk_text', chunk)
            self.assertIn('chunk_char_count', chunk)
            self.assertIn('chunk_word_count', chunk)
            self.assertIn('chunk_token_count', chunk)
    
    def test_chunker_factory_consistency(self):
        """Test that factory creates consistent chunkers."""
        # Create multiple chunkers with same parameters
        chunker1 = create_chunker("fixed", chunk_size=500)
        chunker2 = create_chunker("fixed", chunk_size=500)
        
        # Should produce same results
        chunks1 = chunker1.chunk_pages(self.notebook_pages)
        chunks2 = chunker2.chunk_pages(self.notebook_pages)
        
        self.assertEqual(len(chunks1), len(chunks2))
        for c1, c2 in zip(chunks1, chunks2):
            self.assertEqual(c1['chunk_text'], c2['chunk_text'])
            self.assertEqual(c1['chunk_char_count'], c2['chunk_char_count'])
            self.assertEqual(c1['chunk_word_count'], c2['chunk_word_count'])


if __name__ == '__main__':
    unittest.main()
