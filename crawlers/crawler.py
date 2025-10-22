import os
import ssl
import urllib.request
import urllib.error
from urllib.parse import urlparse
from typing import Optional
import json
from abc import ABC, abstractmethod


class DatasetCrawler(ABC):
    """Base class for downloading datasets into structured folders.

    Categories supported: 'sft', 'rl', 'rag'. Files are saved under
    `root_dir/<category>/<filename>`.
    """

    ALLOWED_CATEGORIES = {"sft", "rl", "rag"}

    def __init__(self, root_dir: str = "data") -> None:
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

    def _category_dir(self, category: str) -> str:
        """Validate and return the category directory path."""
        if category not in self.ALLOWED_CATEGORIES:
            raise ValueError(
                f"Unknown category '{category}'. "
                f"Expected one of {sorted(self.ALLOWED_CATEGORIES)}"
            )
        path = os.path.join(self.root_dir, category)
        os.makedirs(path, exist_ok=True)
        return path

    @abstractmethod
    def download(self, *args, **kwargs) -> str:
        """Download a dataset. Must be implemented by subclasses."""
        pass


class URLDatasetCrawler(DatasetCrawler):
    """Downloads datasets from URLs.

    Example:
        >>> crawler = URLDatasetCrawler()
        >>> crawler.download(
        ...     url="https://example.com/dataset.json",
        ...     category="sft"
        ... )
    """

    def __init__(self, root_dir: str = "data") -> None:
        super().__init__(root_dir)
        # Relaxed SSL to avoid corp proxies issues
        self._ssl_context = ssl.create_default_context()
        self._ssl_context.check_hostname = False
        self._ssl_context.verify_mode = ssl.CERT_NONE

    def _filename_from_url(self, url: str) -> str:
        """Extract filename from URL."""
        parsed = urlparse(url)
        name = os.path.basename(parsed.path)
        return name

    def download(
        self,
        url: str,
        category: str,
        filename: Optional[str] = None,
        overwrite: bool = False
    ) -> str:
        """Downloads a file from URL to the category folder.

        Args:
            url: URL to download from
            category: Dataset category ('sft', 'rl', 'rag')
            filename: Optional custom filename
            overwrite: Whether to overwrite existing files

        Returns:
            Path to the downloaded file
        """
        out_dir = self._category_dir(category)
        fname = filename or self._filename_from_url(url)
        out_path = os.path.join(out_dir, fname)

        if os.path.exists(out_path) and not overwrite:
            print(f"File already exists at {out_path}")
            return out_path

        print(f"Downloading from {url}...")
        with urllib.request.urlopen(url, context=self._ssl_context) as resp:
            if resp.status != 200:
                raise urllib.error.HTTPError(
                    url, resp.status, f"HTTP {resp.status}: {resp.reason}",
                    resp.headers, None
                )
            data = resp.read()

        with open(out_path, "wb") as f:
            f.write(data)

        print(f"Downloaded to {out_path}")
        return out_path


class HFDatasetCrawler(DatasetCrawler):
    """Downloads datasets from HuggingFace Hub.

    Example:
        >>> crawler = HFDatasetCrawler()
        >>> crawler.download(
        ...     repo_id="tatsu-lab/alpaca",
        ...     category="sft",
        ...     split="train"
        ... )
    """

    def download(
        self,
        repo_id: str,
        category: str,
        split: Optional[str] = None,
        filename: Optional[str] = None,
        overwrite: bool = False,
        token: Optional[str] = None,
    ) -> str:
        """Downloads a dataset from HuggingFace Hub.

        Args:
            repo_id: HuggingFace dataset repository ID (e.g., "tatsu-lab/alpaca")
            category: Dataset category ('sft', 'rl', 'rag')
            split: Optional dataset split (e.g., 'train', 'test')
            filename: Optional custom filename to save as
            overwrite: Whether to overwrite existing files
            token: Optional HuggingFace API token for private repos

        Returns:
            Path to the downloaded dataset file

        Raises:
            ImportError: If datasets library is not installed
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "HuggingFace datasets library is required. "
                "Install with: pip install datasets"
            )

        # Load token from environment if not provided
        if token is None:
            token = os.getenv("HUGGINGFACE_TOKEN")

        out_dir = self._category_dir(category)

        # Generate filename from repo_id if not provided
        if filename is None:
            repo_name = repo_id.split("/")[-1]
            filename = f"{repo_name}.json" if split is None else f"{repo_name}_{split}.json"

        out_path = os.path.join(out_dir, filename)

        if os.path.exists(out_path) and not overwrite:
            print(f"Dataset already exists at {out_path}")
            return out_path

        print(f"Downloading dataset '{repo_id}' from HuggingFace...")

        # Download the dataset
        if split:
            dataset = load_dataset(repo_id, split=split, token=token)
        else:
            dataset = load_dataset(repo_id, token=token)

        # Save to JSON format
        if hasattr(dataset, 'to_json'):
            # Single split
            dataset.to_json(out_path)
        else:
            # Multiple splits - save as dict
            data = {
                split_name: list(split_data)
                for split_name, split_data in dataset.items()
            }
            with open(out_path, 'w') as f:
                json.dump(data, f, indent=2)

        print(f"Dataset saved to {out_path}")
        return out_path


