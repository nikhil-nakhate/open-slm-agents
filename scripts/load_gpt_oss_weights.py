"""Script to download GPT-OSS weights from HuggingFace Hub.

The refactored GPT-OSS implementation can consume the downloaded
checkpoint directly (no conversion step required).

Usage:
    python scripts/load_gpt_oss_weights.py --repo-id openai/gpt-oss-20b --output-dir weights/gpt-oss-20b
"""

import argparse
import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download


def download_gpt_oss_weights(
    repo_id: str = "openai/gpt-oss-20b",
    output_dir: str = "weights/gpt-oss-20b",
    use_auth_token: bool = True,
) -> str:
    """Download GPT-OSS weights from HuggingFace Hub."""
    load_dotenv()

    token = None
    if use_auth_token:
        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            print("Warning: HUGGINGFACE_TOKEN not found in environment.")
            print("For private repos, set HUGGINGFACE_TOKEN in .env")

    print(f"Downloading {repo_id} to {output_dir} ...")
    local_dir = snapshot_download(
        repo_id=repo_id,
        local_dir=output_dir,
        token=token,
        resume_download=True,
    )
    print(f"✓ Downloaded checkpoint to {local_dir}")
    return local_dir


def main():
    parser = argparse.ArgumentParser(description="Download and load GPT-OSS weights")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="openai/gpt-oss-20b",
        help="HuggingFace repository ID (default: openai/gpt-oss-20b)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="weights/gpt-oss-20b",
        help="Directory to save downloaded weights",
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Don't use authentication token (for public repos)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download weights, don't test loading",
    )

    args = parser.parse_args()

    weights_dir = download_gpt_oss_weights(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        use_auth_token=not args.no_auth,
    )

    if args.download_only:
        print("\nDownload complete. Update configs/models/gpt_oss.yaml with the `model.weights` entry pointing to this directory.")
    else:
        print(
            "\nDownload complete. Set `model.weights: "
            f"{weights_dir}` inside configs/models/gpt_oss.yaml "
            "and run `python infer.py --config configs/models/gpt_oss.yaml`."
        )


if __name__ == "__main__":
    main()
