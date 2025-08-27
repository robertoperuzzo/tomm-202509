#!/usr/bin/env python3
"""
Test runner for preprocessing pipeline.

This script provides an easy way to run all tests for the preprocessing pipeline.

Usage:
    python3 run_tests.py [test_type]

Where test_type can be:
    - unit: Run only unit tests
    - integration: Run only integration tests  
    - all: Run all tests (default)
    - fast: Run all tests except slow ones
"""

import subprocess
import sys
from pathlib import Path


def run_tests(test_type="all"):
    """Run tests based on the specified type."""
    
    # Change to project root directory
    project_root = Path(__file__).parent
    
    # Base pytest command
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
    
    # Add test type specific options
    if test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "fast":
        cmd.extend(["-m", "not slow"])
    elif test_type == "all":
        pass  # Run all tests
    else:
        print(f"Unknown test type: {test_type}")
        print("Valid types: unit, integration, all, fast")
        return 1
    
    print(f"Running {test_type} tests...")
    print("Command:", " ".join(cmd))
    print("=" * 60)
    
    # Run the tests
    try:
        result = subprocess.run(cmd, cwd=project_root, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTests interrupted by user.")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def main():
    """Main entry point."""
    # Parse command line arguments
    test_type = "all"
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
    
    # Check if pytest is available
    try:
        subprocess.run([sys.executable, "-m", "pytest", "--version"], 
                      capture_output=True, check=True)
    except subprocess.CalledProcessError:
        print("pytest is not installed or not available.")
        print("Please install it with: pip install pytest pytest-asyncio")
        return 1
    
    # Run the tests
    return run_tests(test_type)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
