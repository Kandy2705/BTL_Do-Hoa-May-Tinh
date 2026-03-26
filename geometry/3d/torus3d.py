# Hình xuyến

import numpy as np
import OpenGL.GL as GL
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.shader import Shader
from libs.buffer import VAO, UManager
from libs.lighting import LightingManager

# Import base shape
from base_shape import BaseShape


class Torus(BaseShape):
    def __init__(self, vert_shader, frag_shader, R=0.7, r=0.3, slices=40, stacks=20):
        super().__init__()  # Initialize transform from BaseShape
        self.vert_shader = vert_shader
        self.frag_shader = frag_shader
        self.R = R
        self.r = r
        self.slices = slices
        self.stacks = stacks

        self.vertices, self.colors = self._generate_torus()

        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)

    def _generate_torus(self):
        all_verts = []
        all_colors = []

        for i in range(self.slices):
            theta = 2 * np.pi * i / self.slices
            next_theta = 2 * np.pi * (i + 1) / self.slices

            for j in range(self.stacks):
                phi = 2 * np.pi * j / self.stacks
                next_phi = 2 * np.pi * (j + 1) / self.stacks

                def get_p(t, p):
                    curr_r = self.R + self.r * np.cos(p)
                    return [curr_r * np.cos(t), self.r * np.sin(p), curr_r * np.sin(t)]

                p1 = get_p(theta, phi)
                p2 = get_p(next_theta, phi)
                p3 = get_p(theta, next_phi)
                p4 = get_p(next_theta, next_phi)

                all_verts.extend([p1, p2, p3, p2, p4, p3])

                for p in [p1, p2, p3, p2, p4, p3]:
                    color = np.abs(np.array(p)) / (self.R + self.r)
                    all_colors.append(color)

        return np.array(all_verts, dtype=np.float32), np.array(all_colors, dtype=np.float32)

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors,   ncomponents=3, stride=0, offset=None)
        return self

    def draw(self, projection, view, model=None):
        GL.glUseProgram(self.shader.render_idx)
        
        # Use BaseShape transform
        object_transform = self.get_transform_matrix()
        final_model = object_transform @ (model if model is not None else np.identity(4, dtype=np.float32))
        modelview = view @ final_model

        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        self.vao.activate()
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertices.shape[0])
        self.vao.deactivate()
    
    def set_color(self, color):
        """Set color for the torus - override BaseShape method"""
        # Update colors with new color
        self.colors = np.array([color] * len(self.vertices), dtype=np.float32)
        # Re-setup the VBO to update colors
        self.vao.activate()
        buffer_idx = self.vao.vbo[1]  # Get the color VBO at location 1
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_idx)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.colors, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)