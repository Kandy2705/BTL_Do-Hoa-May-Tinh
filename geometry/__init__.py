# Import Cube directly from the 3d directory
import sys
import os
cube3d_path = os.path.join(os.path.dirname(__file__), '3d')
if cube3d_path not in sys.path:
    sys.path.insert(0, cube3d_path)

from cube3d import Cube

__all__ = ['Cube']
