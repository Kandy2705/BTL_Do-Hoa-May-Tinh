import numpy as np
import OpenGL.GL as GL
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.shader import Shader
from libs.buffer import VAO, UManager
from libs.lighting import LightingManager


class SphereLatLong:

    def __init__(self, vert_shader, frag_shader, lat_div=20, long_div=20):
        self.vert_shader = vert_shader
        self.frag_shader = frag_shader
        self.lat_div = lat_div
        self.long_div = long_div
        self.vertices, self.indices = self._generate_sphere()
        self.normals = self.vertices.copy()
        self.colors = np.random.rand(len(self.vertices), 3).astype(np.float32)

        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.lighting = LightingManager(self.uma)

    def _generate_sphere(self):
        vertices = []
        for i in range(self.lat_div + 1):
            lat = np.pi * (-0.5 + i / self.lat_div)
            for j in range(self.long_div + 1):
                lon = 2 * np.pi * j / self.long_div
                x = np.cos(lat) * np.cos(lon)
                y = np.sin(lat)
                z = np.cos(lat) * np.sin(lon)
                vertices.append([x, y, z])
        vertices = np.array(vertices, dtype=np.float32)

        indices = []
        for i in range(self.lat_div):
            for j in range(self.long_div):
                first = i * (self.long_div + 1) + j
                second = first + self.long_div + 1
                indices.extend([first, second, first + 1, second, second + 1, first + 1])
        indices = np.array(indices, dtype=np.uint32)

        return vertices, indices

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        if 'gouraud' in self.vert_shader.lower() or 'phong' in self.vert_shader.lower():
            self.vao.add_vbo(2, self.normals, ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None)
        self.vao.add_ebo(self.indices)
        return self

    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)
        modelview = view @ (model if model is not None else np.identity(4, dtype=np.float32))
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        if 'gouraud' in self.vert_shader.lower():
            self.lighting.setup_gouraud()
        elif 'phong' in self.vert_shader.lower():
            self.lighting.setup_phong(mode=1)
        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLES, self.indices.size, GL.GL_UNSIGNED_INT, None)
        self.vao.deactivate()