# Hình tam giác

import OpenGL.GL as GL              # standard Python OpenGL wrapper
import numpy as np
import sys
import os

# Add parent directory to path to import libs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.shader import *
from libs import transform as T
from libs.buffer import *
from libs.lighting import LightingManager
import ctypes

# Import base shape
from base_shape import BaseShape


class Triangle(BaseShape):
    def __init__(self, vert_shader, frag_shader):
        super().__init__()  # Initialize transform from BaseShape
        self.vert_shader = vert_shader
        self.frag_shader = frag_shader
        self.vertices = np.array([
            [-1, -1, 0],
            [1, -1, 0],
            [0, 1, 0]
        ], dtype=np.float32)
        normals = np.random.normal(0, 3, (3, 3)).astype(np.float32)
        normals[:, 2] = np.abs(normals[:, 2])
        self.normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        self.colors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], 
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        self.vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.lighting = LightingManager(self.uma)

    def setup(self):
        self.vao.add_vbo(0, 
                         self.vertices,
                         ncomponents=3,
                         dtype=GL.GL_FLOAT,
                         normalized=False,
                         stride=0,
                         offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)

        GL.glUseProgram(self.shader.render_idx)
        projection = T.ortho(-1, 1, -1, 1, -1, 1)
        modelview = np.identity(4, 'f')
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        if 'gouraud' in self.vert_shader.lower():
            self.lighting.setup_gouraud()
        else:
            self.lighting.setup_phong(mode=1)
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
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        self.vao.deactivate()
    
    def set_color(self, color):
        """Set color for the triangle - override BaseShape method"""
        # Update colors with new color
        self.colors = np.array([color] * len(self.vertices), dtype=np.float32)
        # Re-setup the VBO to update colors
        self.vao.activate()
        buffer_idx = self.vao.vbo[1]  # Get the color VBO at location 1
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_idx)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.colors, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
