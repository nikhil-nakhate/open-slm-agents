import argparse
from crawlers import DatasetCrawler


def main():
    parser = argparse.ArgumentParser(description="Download JSON dataset into structured folders (sft|rl|rag)")
    parser.add_argument("--url", type=str, required=True, help="URL to the JSON dataset")
    parser.add_argument("--category", type=str, default="sft", choices=["sft", "rl", "rag"], help="Dataset category")
    parser.add_argument("--filename", type=str, default=None, help="Optional output filename (defaults to name from URL)")
    parser.add_argument("--root_dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing file if present")
    args = parser.parse_args()

    crawler = DatasetCrawler(root_dir=args.root_dir)
    path, data = crawler.download_json(args.url, category=args.category, filename=args.filename, overwrite=args.overwrite)
    print(f"Saved: {path}")
    try:
        print("Number of entries:", len(data))
    except Exception:
        print("Downloaded file loaded; could not infer length (non-list JSON)")


if __name__ == "__main__":
    main()
