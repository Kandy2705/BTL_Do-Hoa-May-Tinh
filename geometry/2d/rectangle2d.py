# Hình chữ nhật

import numpy as np
import OpenGL.GL as GL
import sys
import os

# Add parent directory to path to import libs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.shader import Shader
from libs.buffer import VAO, UManager
from libs.lighting import LightingManager


# Import base shape
from base_shape import BaseShape


class Rectangle(BaseShape):
    def __init__(self, vert_shader, frag_shader):
        super().__init__()  # Initialize transform from BaseShape
        self.vert_shader = vert_shader
        self.frag_shader = frag_shader
        self.vertices = np.array([
            [-2, -1, 0],
            [ 2, -1, 0],
            [ 2,  1, 0],
            [-2,  1, 0] 
        ], dtype=np.float32)
        self.normals = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1]
        ], dtype=np.float32)
        self.colors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0]
        ], dtype=np.float32)
        self.indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.lighting = LightingManager(self.uma)

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        if 'gouraud' in self.vert_shader.lower() or 'phong' in self.vert_shader.lower():
            self.vao.add_vbo(2, self.normals, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_ebo(self.indices)
        return self

    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)
        
        # Use BaseShape transform
        object_transform = self.get_transform_matrix()
        final_model = object_transform @ (model if model is not None else np.identity(4, dtype=np.float32))
        modelview = view @ final_model
        
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        if 'gouraud' in self.vert_shader.lower():
            self.lighting.setup_gouraud()
        elif 'phong' in self.vert_shader.lower():
            self.lighting.setup_phong(mode=1)
        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
        self.vao.deactivate()
    
    def set_color(self, color):
        """Set color for the rectangle - override BaseShape method"""
        # Update colors with new color
        self.colors = np.array([color] * len(self.vertices), dtype=np.float32)
        # Re-setup the VBO to update colors
        self.vao.activate()
        buffer_idx = self.vao.vbo[1]  # Get the color VBO at location 1
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_idx)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.colors, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)