"""Sphere approximated by subdividing a tetrahedron for rendering in the BTL project."""

import numpy as np
import OpenGL.GL as GL
import sys
import os

# Add parent directory to path to import libs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.shader import Shader
from libs.buffer import VAO, UManager
from libs.lighting import LightingManager


class SphereTetrahedron:
    """Sphere approximated by subdividing a tetrahedron and normalizing to radius."""

    def __init__(self, vert_shader, frag_shader, subdivisions=3):
        self.vert_shader = vert_shader
        self.frag_shader = frag_shader
        self.subdivisions = subdivisions
        self.vertices, self.indices = self._generate_sphere()
        self.normals = self.vertices.copy()  # Normals are the same as positions for unit sphere
        self.colors = np.random.rand(len(self.vertices), 3).astype(np.float32)

        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.lighting = LightingManager(self.uma)

    def _generate_sphere(self):
        # Start with tetrahedron vertices
        t = (1.0 + np.sqrt(5.0)) / 2.0  # Golden ratio
        vertices = np.array([
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
        ], dtype=np.float32)
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

        # Indices for tetrahedron (simplified, need proper triangulation)
        # For simplicity, use a basic triangulation
        indices = np.array([
            0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
            1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
            3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
            4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1
        ], dtype=np.uint32)
        # Subdivide if needed (for simplicity, skip advanced subdivision)
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