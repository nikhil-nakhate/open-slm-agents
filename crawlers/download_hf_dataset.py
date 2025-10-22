#!/usr/bin/env python3
"""Download datasets from HuggingFace Hub.

This script downloads HuggingFace datasets and saves them to the appropriate
category folder (sft, rl, or rag) in JSON format.

Usage:
    # Download entire dataset
    python crawlers/download_hf_dataset.py \
        --repo-id "tatsu-lab/alpaca" \
        --category sft

    # Download specific split
    python crawlers/download_hf_dataset.py \
        --repo-id "tatsu-lab/alpaca" \
        --category sft \
        --split train

    # Custom filename and data directory
    python crawlers/download_hf_dataset.py \
        --repo-id "yahma/alpaca-cleaned" \
        --category sft \
        --filename alpaca_cleaned.json \
        --data-dir my_data

    # Overwrite existing dataset
    python crawlers/download_hf_dataset.py \
        --repo-id "tatsu-lab/alpaca" \
        --category sft \
        --overwrite

Environment:
    Set HUGGINGFACE_TOKEN in .env file for downloading private datasets.
    Get your token from: https://huggingface.co/settings/tokens
"""

import argparse
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")

from crawler import HFDatasetCrawler


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--repo-id',
        required=True,
        help='HuggingFace dataset repository ID (e.g., "tatsu-lab/alpaca")'
    )
    parser.add_argument(
        '--category',
        required=True,
        choices=['sft', 'rl', 'rag'],
        help='Dataset category: sft (supervised fine-tuning), rl (reinforcement learning), or rag (retrieval)'
    )
    parser.add_argument(
        '--split',
        help='Dataset split to download (e.g., "train", "test", "validation"). Downloads all splits if not specified.'
    )
    parser.add_argument(
        '--filename',
        help='Custom filename for saved dataset. Auto-generated from repo name if not specified.'
    )
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Root directory for datasets (default: data)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing dataset files'
    )
    parser.add_argument(
        '--token',
        help='HuggingFace API token (or set HUGGINGFACE_TOKEN environment variable)'
    )

    args = parser.parse_args()

    # Create HuggingFace dataset crawler
    crawler = HFDatasetCrawler(root_dir=args.data_dir)

    print(f"Downloading dataset from HuggingFace Hub")
    print(f"  Repository: {args.repo_id}")
    print(f"  Category: {args.category}")
    print(f"  Split: {args.split or 'all'}")
    print(f"  Data directory: {args.data_dir}")
    print()

    try:
        path = crawler.download(
            repo_id=args.repo_id,
            category=args.category,
            split=args.split,
            filename=args.filename,
            overwrite=args.overwrite,
            token=args.token,
        )

        print(f"\n✅ Dataset downloaded successfully!")
        print(f"   Saved to: {path}")

    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        if "401" in str(e) or "403" in str(e):
            print("\nTip: This might be a private dataset. Make sure you have:")
            print("  1. Set HUGGINGFACE_TOKEN in your .env file")
            print("  2. Been granted access to the dataset on HuggingFace")
        raise


if __name__ == '__main__':
    main()
