import os
from typing import Optional

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional, progress bar falls back to huggingface default
    tqdm = None  # type: ignore


class ModelCrawler:
    """Downloads model weights from HuggingFace Hub.

    Saves models under `weights/<model-name>/` by default.
    """

    def __init__(self, root_dir: str = "weights") -> None:
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

    def download_hf_model(
        self,
        repo_id: str,
        output_dir: Optional[str] = None,
        filename: Optional[str] = None,
        overwrite: bool = False,
        token: Optional[str] = None,
        revision: str = "main",
        max_workers: int = 8,
    ) -> str:
        """Downloads model weights from HuggingFace Hub.

        Args:
            repo_id: HuggingFace model repository ID (e.g., "gpt2", "meta-llama/Llama-2-7b")
            output_dir: Optional custom output directory (default: weights/<repo_name>)
            filename: Optional specific file to download (downloads all if not specified)
            overwrite: Whether to overwrite existing files
            token: Optional HuggingFace API token for private repos
            revision: Git revision (branch/tag/commit) to download from
            max_workers: Maximum number of parallel downloads (default: 8)

        Returns:
            Path to the downloaded model directory or file

        Examples:
            >>> crawler = ModelCrawler()
            >>> # Download entire model
            >>> crawler.download_hf_model("gpt2")
            >>> # Download specific file
            >>> crawler.download_hf_model("gpt2", filename="pytorch_model.bin")
            >>> # Download with custom output
            >>> crawler.download_hf_model("gpt2", output_dir="my_models/gpt2")
        """
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError:
            raise ImportError(
                "HuggingFace hub library is required. "
                "Install with: pip install huggingface-hub"
            )

        # Load token from environment if not provided
        if token is None:
            token = os.getenv("HUGGINGFACE_TOKEN")

        # Set output directory
        if output_dir is None:
            repo_name = repo_id.split("/")[-1]
            output_dir = os.path.join(self.root_dir, repo_name)

        os.makedirs(output_dir, exist_ok=True)

        if filename:
            # Download specific file
            out_path = os.path.join(output_dir, filename)

            if os.path.exists(out_path) and not overwrite:
                print(f"Model file already exists at {out_path}")
                return out_path

            print(f"Downloading file '{filename}' from model '{repo_id}'...")
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=token,
                revision=revision,
                cache_dir=None,
                local_dir=output_dir,
                local_dir_use_symlinks=False,
                tqdm_class=tqdm if tqdm is not None else None,
            )
            print(f"Model file saved to {downloaded_path}")
            return downloaded_path
        else:
            # Download entire model repository
            if os.path.exists(output_dir) and os.listdir(output_dir) and not overwrite:
                print(f"Model already exists at {output_dir}")
                return output_dir

            print(f"Downloading model '{repo_id}' from HuggingFace...")
            print(f"Using {max_workers} parallel workers for faster download...")
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                token=token,
                revision=revision,
                cache_dir=None,
                local_dir=output_dir,
                local_dir_use_symlinks=False,
                max_workers=max_workers,  # Enable parallel downloads
                tqdm_class=tqdm if tqdm is not None else None,
            )
            print(f"Model saved to {downloaded_path}")
            return downloaded_path
