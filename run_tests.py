#!/usr/bin/env python3
"""
Simple test runner for the moralhazard package.
Run with: python run_tests.py
"""

import subprocess
import sys
import os

def run_tests():
    """Run the test suite using pytest."""
    print("Running tests for moralhazard package...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("tests"):
        print("Error: tests directory not found. Make sure you're in the project root.")
        sys.exit(1)
    
    # Run pytest
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", "tests/", "-v"
        ], capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n" + "=" * 50)
            print("✅ All tests passed!")
        else:
            print("\n" + "=" * 50)
            print("❌ Some tests failed!")
            sys.exit(result.returncode)
            
    except FileNotFoundError:
        print("Error: pytest not found. Install it with:")
        print("  uv add --dev pytest pytest-cov")
        sys.exit(1)
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
