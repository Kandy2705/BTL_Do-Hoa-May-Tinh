# Mathematical Surface: z = f(x,y)

import sys
import os
import numpy as np
import ctypes
import math

# Add parent directory to path to import libs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.shader import *
from libs import transform as T
from libs.buffer import *
from libs.lighting import LightingManager
import OpenGL.GL as GL


class MathematicalSurface(object):
    def __init__(self, vert_shader, frag_shader, func=None, x_range=(-5, 5), y_range=(-5, 5), resolution=50):
        self.vert_shader = vert_shader
        self.frag_shader = frag_shader
        
        # Default function: Himmelblau's function
        if func is None:
            self.func = lambda x, y: (x**2 + y - 11)**2 + (x + y**2 - 7)**2
        else:
            self.func = func
            
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        
        self._generate_surface()
        
    def _generate_surface(self):
        """Generate vertices, indices, normals and colors for the mathematical surface"""
        x = np.linspace(self.x_range[0], self.x_range[1], self.resolution)
        y = np.linspace(self.y_range[0], self.y_range[1], self.resolution)
        
        # Create mesh grid
        X, Y = np.meshgrid(x, y)
        
        # Calculate Z values
        Z = self.func(X, Y)
        
        # Normalize Z to reasonable range
        z_min, z_max = Z.min(), Z.max()
        if z_max - z_min > 0:
            Z = 2 * (Z - z_min) / (z_max - z_min) - 1
        
        # Generate vertices
        vertices = []
        normals = []
        colors = []
        
        for i in range(self.resolution):
            for j in range(self.resolution):
                vertices.append([X[i, j], Y[i, j], Z[i, j]])
                
                # Calculate normal using central differences
                if 0 < i < self.resolution-1 and 0 < j < self.resolution-1:
                    dx = (Z[i, j+1] - Z[i, j-1]) / (2 * (x[1] - x[0]))
                    dy = (Z[i+1, j] - Z[i-1, j]) / (2 * (y[1] - y[0]))
                    normal = np.array([-dx, -dy, 1.0])
                    normal = normal / np.linalg.norm(normal)
                else:
                    normal = np.array([0, 0, 1])
                
                normals.append(normal)
                
                # Color based on height
                height_normalized = (Z[i, j] - Z.min()) / (Z.max() - Z.min() + 1e-6)
                colors.append([
                    0.2 + 0.8 * height_normalized,  # R
                    0.3 + 0.4 * (1 - height_normalized),  # G  
                    0.8 - 0.6 * height_normalized   # B
                ])
        
        self.vertices = np.array(vertices, dtype=np.float32)
        self.normals = np.array(normals, dtype=np.float32)
        self.colors = np.array(colors, dtype=np.float32)
        
        # Generate indices for triangle strips
        indices = []
        for i in range(self.resolution - 1):
            for j in range(self.resolution - 1):
                # Two triangles per grid cell
                v0 = i * self.resolution + j
                v1 = i * self.resolution + (j + 1)
                v2 = (i + 1) * self.resolution + j
                v3 = (i + 1) * self.resolution + (j + 1)
                
                # First triangle
                indices.extend([v0, v1, v2])
                # Second triangle
                indices.extend([v2, v1, v3])
        
        self.indices = np.array(indices, dtype=np.int32)

    def setup(self):
        """Setup buffers and shader"""
        self.vao = VAO()
        
        # Setup vertex buffer
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        
        # Setup color buffer
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        
        # Setup normal buffer
        self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)
        
        # Setup index buffer
        self.vao.add_ebo(self.indices)
        
        # Setup shader
        self.shader = Shader(self.vert_shader, self.frag_shader)
        self.uma = UManager(self.shader)
        
        # Setup lighting
        self.lighting = LightingManager(self.uma)
        
        return self

    def draw(self, projection, view, model):
        """Draw the mathematical surface"""
        if model is None:
            model = T.identity()
            
        GL.glUseProgram(self.shader.render_idx)
        modelview = view @ model
        
        # Set uniforms
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        
        # Setup lighting if using Gouraud or Phong shader
        if 'gouraud' in self.vert_shader.lower():
            self.lighting.setup_gouraud()
        elif 'phong' in self.vert_shader.lower():
            self.lighting.setup_phong(mode=1)
        else:
            # For color interpolation shader
            self.lighting.setup_phong(mode=0)
        
        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLES, len(self.indices), GL.GL_UNSIGNED_INT, None)

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'vao'):
            self.vao.delete()
        if hasattr(self, 'shader'):
            self.shader.delete()
