#!/usr/bin/env python3
"""Download model weights from HuggingFace Hub.

This script downloads HuggingFace models and saves them to the weights directory.

Usage:
    # Download entire model
    python crawlers/download_hf_model.py --repo-id "gpt2"

    # Download specific file from model
    python crawlers/download_hf_model.py \
        --repo-id "gpt2" \
        --filename "pytorch_model.bin"

    # Custom output directory
    python crawlers/download_hf_model.py \
        --repo-id "meta-llama/Llama-2-7b-hf" \
        --output-dir "my_models/llama2-7b"

    # Download specific revision (branch/tag/commit)
    python crawlers/download_hf_model.py \
        --repo-id "gpt2" \
        --revision "main"

    # Overwrite existing model
    python crawlers/download_hf_model.py \
        --repo-id "gpt2" \
        --overwrite

Environment:
    Set HUGGINGFACE_TOKEN in .env file for downloading private models or gated models.
    Get your token from: https://huggingface.co/settings/tokens

    For gated models (like Llama 2), you also need to:
    1. Request access on the model's HuggingFace page
    2. Wait for approval
    3. Use your token to download
"""

import argparse
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")

from model_crawler import ModelCrawler


def main():
    parser = argparse.ArgumentParser(
        description="Download model weights from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--repo-id',
        required=True,
        help='HuggingFace model repository ID (e.g., "gpt2", "meta-llama/Llama-2-7b-hf")'
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory for model files (default: weights/<model-name>)'
    )
    parser.add_argument(
        '--filename',
        help='Specific file to download. Downloads entire model if not specified.'
    )
    parser.add_argument(
        '--revision',
        default='main',
        help='Git revision (branch/tag/commit) to download (default: main)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing model files'
    )
    parser.add_argument(
        '--token',
        help='HuggingFace API token (or set HUGGINGFACE_TOKEN environment variable)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=8,
        help='Maximum number of parallel download workers (default: 8, increase for faster downloads)'
    )

    args = parser.parse_args()

    # Create crawler
    crawler = ModelCrawler(root_dir='weights')

    print(f"Downloading model from HuggingFace Hub")
    print(f"  Repository: {args.repo_id}")
    print(f"  Revision: {args.revision}")
    if args.filename:
        print(f"  File: {args.filename}")
    else:
        print(f"  Mode: Download entire model")

    # Calculate output directory
    output_location = args.output_dir or f"weights/{args.repo_id.split('/')[-1]}"
    print(f"  Output: {output_location}")
    print()

    try:
        path = crawler.download_hf_model(
            repo_id=args.repo_id,
            output_dir=args.output_dir,
            filename=args.filename,
            overwrite=args.overwrite,
            token=args.token,
            revision=args.revision,
            max_workers=args.max_workers,
        )

        print(f"\n✅ Model downloaded successfully!")
        print(f"   Saved to: {path}")

    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
            print("\nTip: This might be a private or gated model. Make sure you have:")
            print("  1. Set HUGGINGFACE_TOKEN in your .env file")
            print("  2. Requested and received access to the model on HuggingFace")
        raise


if __name__ == '__main__':
    main()
