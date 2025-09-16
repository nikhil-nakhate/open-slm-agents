import os
import ssl
import urllib.request
from urllib.parse import urlparse
from typing import Optional, Tuple, Any
import json


class DatasetCrawler:
    """Downloads remote files into structured dataset folders.

    Categories supported: 'sft', 'rl', 'rag'. Files are saved under
    `root_dir/<category>/<filename>`.
    """

    ALLOWED = {"sft", "rl", "rag"}

    def __init__(self, root_dir: str = "data") -> None:
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)
        # Relaxed SSL similar to sample script to avoid corp proxies issues
        self._ssl_context = ssl.create_default_context()
        self._ssl_context.check_hostname = False
        self._ssl_context.verify_mode = ssl.CERT_NONE

    def _category_dir(self, category: str) -> str:
        if category not in self.ALLOWED:
            raise ValueError(f"Unknown category '{category}'. Expected one of {sorted(self.ALLOWED)}")
        path = os.path.join(self.root_dir, category)
        os.makedirs(path, exist_ok=True)
        return path

    def _filename_from_url(self, url: str) -> str:
        parsed = urlparse(url)
        name = os.path.basename(parsed.path)
        return name or "downloaded.json"

    def download(self, url: str, category: str, filename: Optional[str] = None, overwrite: bool = False) -> str:
        """Downloads the URL to the category folder and returns the file path."""
        out_dir = self._category_dir(category)
        fname = filename or self._filename_from_url(url)
        out_path = os.path.join(out_dir, fname)

        if os.path.exists(out_path) and not overwrite:
            return out_path

        with urllib.request.urlopen(url, context=self._ssl_context) as resp:
            data = resp.read()
        with open(out_path, "wb") as f:
            f.write(data)
        return out_path

    def load_json(self, path: str) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def download_json(self, url: str, category: str, filename: Optional[str] = None, overwrite: bool = False) -> Tuple[str, Any]:
        path = self.download(url, category, filename=filename, overwrite=overwrite)
        return path, self.load_json(path)

