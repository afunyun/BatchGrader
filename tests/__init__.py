"""
Initialize the tests package and configure the Python path.
This ensures that we can import modules from the src directory in our tests.
"""
import os
import sys
from pathlib import Path

# Add the parent directory (project root) to the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / 'src'

# Add src directory to Python path if not already there
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
