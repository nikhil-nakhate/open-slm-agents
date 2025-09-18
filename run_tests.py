#!/usr/bin/env python3
"""Test runner script for open-slm-agents."""

import sys
import subprocess
import os
from pathlib import Path

def run_tests():
    """Run all unit tests."""
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Add project root to Python path
    sys.path.insert(0, str(project_root))
    
    # Run pytest with verbose output
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--color=yes"
    ]
    
    print("Running unit tests...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, cwd=project_root, check=True)
        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        return 0
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 50)
        print("❌ Some tests failed!")
        return e.returncode
    except FileNotFoundError:
        print("❌ pytest not found. Please install it with: pip install pytest")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())
