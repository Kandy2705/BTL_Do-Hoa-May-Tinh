"""
Base shape class with transform and color support
"""
import numpy as np
import sys
import os

# Add libs path for transform
libs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'libs')
if libs_path not in sys.path:
    sys.path.insert(0, libs_path)

import transform as T


class BaseShape:
    def __init__(self):
        # Initialize transform
        self.position = [0.0, 0.0, 0.0]
        self.rotation = [0.0, 0.0, 0.0]
        self.scale = [1.0, 1.0, 1.0]
    
    def set_transform(self, position, rotation, scale):
        """Set transform for the shape"""
        self.position = position
        self.rotation = rotation  
        self.scale = scale
    
    def set_color(self, color):
        """Set color for the shape - to be overridden by subclasses"""
        pass
    
    def get_transform_matrix(self):
        """Get the combined transform matrix"""
        translate_mat = T.translate(*self.position)
        rotate_x = T.rotate((1, 0, 0), self.rotation[0])
        rotate_y = T.rotate((0, 1, 0), self.rotation[1]) 
        rotate_z = T.rotate((0, 0, 1), self.rotation[2])
        scale_mat = T.scale(*self.scale)
        
        # Combine transforms: T * R * S
        return translate_mat @ rotate_x @ rotate_y @ rotate_z @ scale_mat
    
    def draw_with_transform(self, projection, view, model):
        """Draw with transform applied - to be called by subclasses"""
        object_transform = self.get_transform_matrix()
        final_model = object_transform @ (model if model is not None else np.identity(4, dtype=np.float32))
        return self.draw(projection, view, final_model)
