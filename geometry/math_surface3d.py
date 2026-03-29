import sys
import os
import numpy as np
import ctypes
import math
import warnings

# Add parent directory to path to import libs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.shader import *
from libs import transform as T
from libs.buffer import *
from libs.lighting import LightingManager
import OpenGL.GL as GL

# Import base shape
from base_shape import BaseShape


class MathematicalSurface(BaseShape):
    def __init__(self, vert_shader, frag_shader, func=None, x_range=(-5, 5), y_range=(-5, 5), resolution=50):
        super().__init__()  # Initialize transform from BaseShape
        self.vert_shader = vert_shader
        self.frag_shader = frag_shader
        self.use_custom_color = False  # Flag to use custom color or auto-generated colors
        self.use_flat_color = False  # Flag for flat color override
        self.original_colors = None  # Store original auto-generated colors
        
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
        import warnings
        
        x_real = []
        for i in range(self.resolution):
            x = self.x_range[0] + (self.x_range[1] - self.x_range[0]) * i / (self.resolution - 1)
            x_real.append(x)
        
        y_real = []
        for i in range(self.resolution):
            y = self.y_range[0] + (self.y_range[1] - self.y_range[0]) * i / (self.resolution - 1)
            y_real.append(y)
        
        X_real = []
        Y_real = []
        for y_idx in range(len(y_real)):
            row_x = []
            row_y = []
            for x_idx in range(len(x_real)):
                row_x.append(x_real[x_idx])
                row_y.append(y_real[y_idx])
            X_real.append(row_x)
            Y_real.append(row_y)
        
        X_real = np.array(X_real, dtype=np.float32)
        Y_real = np.array(Y_real, dtype=np.float32)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z_real = self.func(X_real, Y_real)
            
        Z_real = np.nan_to_num(Z_real, nan=0.0, posinf=100.0, neginf=-100.0)
        Z_real = np.clip(Z_real, -100.0, 100.0)
        
        def normalize_for_draw(arr, target_min=-2.0, target_max=2.0):
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max - arr_min == 0:
                return np.zeros_like(arr)
            return (arr - arr_min) / (arr_max - arr_min) * (target_max - target_min) + target_min
            
        X_draw = normalize_for_draw(X_real)
        Y_draw = normalize_for_draw(Y_real)
        Z_draw = normalize_for_draw(Z_real)
        
        vertices = []
        normals = []
        colors = []
        
        dx_draw = X_draw[0, 1] - X_draw[0, 0]
        dy_draw = Y_draw[1, 0] - Y_draw[0, 0]
        
        for i in range(self.resolution):
            for j in range(self.resolution):
                vertices.append([X_draw[i, j], Y_draw[i, j], Z_draw[i, j]])
                
                if 0 < i < self.resolution-1 and 0 < j < self.resolution-1:
                    dz_dx = (Z_draw[i, j+1] - Z_draw[i, j-1]) / (2 * dx_draw)
                    dz_dy = (Z_draw[i+1, j] - Z_draw[i-1, j]) / (2 * dy_draw)
                    normal = np.array([-dz_dx, -dz_dy, 1.0])
                    normal = normal / np.linalg.norm(normal)
                else:
                    normal = np.array([0.0, 0.0, 1.0])
                normals.append(normal)
                
                h = (Z_draw[i, j] + 2.0) / 4.0
                colors.append([
                    0.2 + 0.8 * h,
                    0.3 + 0.4 * (1 - abs(2*h-1)),
                    0.8 - 0.6 * h
                ])
                
        self.vertices = np.array(vertices, dtype=np.float32)
        self.normals = np.array(normals, dtype=np.float32)
        self.colors = np.array(colors, dtype=np.float32)
        self.original_colors = self.colors.copy()  # Store original auto-generated colors
        
        indices = []
        for i in range(self.resolution - 1):
            for j in range(self.resolution - 1):
                v0 = i * self.resolution + j
                v1 = i * self.resolution + (j + 1)
                v2 = (i + 1) * self.resolution + j
                v3 = (i + 1) * self.resolution + (j + 1)
                
                indices.extend([v0, v1, v2])
                indices.extend([v2, v1, v3])
        
        self.indices = np.array(indices, dtype=np.int32)
        
    def setup(self):
        """Setup buffers and shader"""
        self.vao = VAO()
        
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        
        self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)
        
        self.vao.add_ebo(self.indices)
        
        self.shader = Shader(self.vert_shader, self.frag_shader)
        self.uma = UManager(self.shader)
        
        self.lighting = LightingManager(self.uma)
        
        return self

    def draw(self, projection, view, model):
        """Draw the mathematical surface"""
        GL.glUseProgram(self.shader.render_idx)
        
        # Use BaseShape transform
        object_transform = self.get_transform_matrix()
        final_model = object_transform @ (model if model is not None else np.identity(4, dtype=np.float32))
        modelview = view @ final_model
        
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        
        if 'gouraud' in self.vert_shader.lower():
            self.lighting.setup_gouraud(view_matrix=view)
        elif 'phong' in self.vert_shader.lower():
            self.lighting.setup_phong(mode=1, view_matrix=view)
        else:
            self.lighting.setup_phong(mode=0, view_matrix=view)
        
        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLES, len(self.indices), GL.GL_UNSIGNED_INT, None)
    
    def set_color(self, color):
        """Set color for the mathematical surface - override BaseShape method"""
        if self.use_custom_color:
            # Use custom color
            self.colors = np.array([color] * len(self.vertices), dtype=np.float32)
        else:
            # Use original auto-generated colors (height-based)
            self.colors = self.original_colors.copy()
        
        # Re-setup the VBO to update colors
        self.vao.activate()
        buffer_idx = self.vao.vbo[1]  # Get the color VBO at location 1
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_idx)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.colors, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
    
    def set_color_mode(self, use_custom_color):
        """Toggle between auto-color and custom color mode"""
        self.use_custom_color = use_custom_color

    def set_solid_color(self, color):
        """Set solid color for the mathematical surface"""
        self.use_flat_color = True
        self.flat_color = np.array(color[:3], dtype=np.float32)
        self.colors = np.array([color[:3]] * len(self.vertices), dtype=np.float32)
        self.vao.activate()
        buffer_idx = self.vao.vbo[1]
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_idx)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.colors, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'vao'):
            self.vao.delete()
        if hasattr(self, 'shader'):
            self.shader.delete()
