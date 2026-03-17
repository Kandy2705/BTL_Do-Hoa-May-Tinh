import numpy as np
import OpenGL.GL as GL
import sys
import os

# Add parent directory to path to import libs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.shader import Shader
from libs.buffer import VAO, UManager
from libs.lighting import LightingManager

class TruncatedCone:
    def __init__(self, vert_shader, frag_shader, radius_bottom=0.8, radius_top=0.4, height=1.0, slices=30, stacks=1):
        self.vert_shader = vert_shader
        self.frag_shader = frag_shader
        self.radius_bottom = radius_bottom
        self.radius_top = radius_top
        self.height = height
        self.slices = slices
        self.stacks = stacks

        self.vertices, self.colors = self._generate_truncated_cone()

        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)

    def _generate_truncated_cone(self):
        all_verts = []
        all_colors = []

        for i in range(self.stacks):
            h_low = -self.height / 2 + (i / self.stacks) * self.height
            h_high = -self.height / 2 + ((i + 1) / self.stacks) * self.height
            
            r_low = self.radius_bottom + (self.radius_top - self.radius_bottom) * (i / self.stacks)
            r_high = self.radius_bottom + (self.radius_top - self.radius_bottom) * ((i + 1) / self.stacks)

            for j in range(self.slices):
                theta = 2 * np.pi * j / self.slices
                next_theta = 2 * np.pi * (j + 1) / self.slices

                p1 = [r_low * np.cos(theta), h_low, r_low * np.sin(theta)]
                p2 = [r_low * np.cos(next_theta), h_low, r_low * np.sin(next_theta)]
                p3 = [r_high * np.cos(theta), h_high, r_high * np.sin(theta)]
                p4 = [r_high * np.cos(next_theta), h_high, r_high * np.sin(next_theta)]

                all_verts.extend([p1, p2, p3, p2, p4, p3])
                
                for p in [p1, p2, p3, p2, p4, p3]:
                    color = np.abs(np.array(p) / max(self.radius_bottom, self.height))
                    all_colors.append(color)

        self._add_cap(all_verts, all_colors, self.radius_bottom, -self.height / 2, clockwise=True)
        self._add_cap(all_verts, all_colors, self.radius_top, self.height / 2, clockwise=False)

        return np.array(all_verts, dtype=np.float32), np.array(all_colors, dtype=np.float32)

    def _add_cap(self, verts, colors, radius, height, clockwise=True):
        center = [0.0, height, 0.0]
        for i in range(self.slices):
            theta = 2 * np.pi * i / self.slices
            next_theta = 2 * np.pi * (i + 1) / self.slices
            
            p1 = [radius * np.cos(theta), height, radius * np.sin(theta)]
            p2 = [radius * np.cos(next_theta), height, radius * np.sin(next_theta)]
            
            if clockwise:
                verts.extend([center, p2, p1])
            else:
                verts.extend([center, p1, p2])
            
            for _ in range(3):
                colors.append([0.5, 0.5, 0.5])

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors,   ncomponents=3, stride=0, offset=None)
        return self

    def draw(self, projection, view, model=None):
        GL.glUseProgram(self.shader.render_idx) 
        if model is None: model = np.identity(4, dtype=np.float32)
        modelview = view @ model 
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        self.vao.activate()
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertices.shape[0])
        self.vao.deactivate()