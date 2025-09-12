import sys
import os

# Get the absolute path to the project root directory
# (one level up from the 'tests' directory)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Get the absolute path to the 'src' directory
src_path = os.path.join(project_root, 'src')

# Add the project root and src directory to the system path
# This allows imports from 'src' to be resolved
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

# Optional: Print the paths to verify
print("Project root added to sys.path:", project_root)
print("Src path added to sys.path:", src_path)