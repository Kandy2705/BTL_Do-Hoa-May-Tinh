"""Basic 3D shapes for rendering in the BTL project."""

import numpy as np
import OpenGL.GL as GL

from libs.shader import Shader
from libs.buffer import VAO


class Cube:
    """Simple colored cube mesh."""

    def __init__(self, vertex_shader, fragment_shader):
        self.shader = Shader(vertex_shader, fragment_shader)
        self.vao = VAO()
        self.vertex_count = 0

    def setup(self):
        # 8 corner vertices (x, y, z) for a cube centered at origin
        positions = np.array([
            # front face
            -1.0, -1.0,  1.0,
             1.0, -1.0,  1.0,
             1.0,  1.0,  1.0,
            -1.0,  1.0,  1.0,
            # back face
            -1.0, -1.0, -1.0,
             1.0, -1.0, -1.0,
             1.0,  1.0, -1.0,
            -1.0,  1.0, -1.0,
        ], dtype=np.float32)

        # Per-vertex color (one color per corner)
        colors = np.array([
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 0.0,
            1.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            0.2, 0.2, 0.2,
        ], dtype=np.float32)

        # Indices for 12 triangles (two per face)
        indices = np.array([
            0, 1, 2,  2, 3, 0,  # front
            1, 5, 6,  6, 2, 1,  # right
            5, 4, 7,  7, 6, 5,  # back
            4, 0, 3,  3, 7, 4,  # left
            3, 2, 6,  6, 7, 3,  # top
            4, 5, 1,  1, 0, 4,  # bottom
        ], dtype=np.uint32)

        self.vao.add_vbo(0, positions, ncomponents=3)
        self.vao.add_vbo(1, colors, ncomponents=3)
        self.vao.add_ebo(indices)
        self.vertex_count = indices.size
        return self

    def draw(self, projection, view, model=None):
        GL.glUseProgram(self.shader.render_idx)

        if model is None:
            model = np.identity(4, dtype=np.float32)

        modelview = view @ model

        # Upload matrices
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shader.render_idx, "projection"),
                              1, GL.GL_TRUE, projection)
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.shader.render_idx, "modelview"),
                              1, GL.GL_TRUE, modelview)

        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLES, self.vertex_count, GL.GL_UNSIGNED_INT, None)
        self.vao.deactivate()

        GL.glUseProgram(0)
