"""Mesh upload and draw wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_ELEMENT_ARRAY_BUFFER,
    GL_FLOAT,
    GL_STATIC_DRAW,
    GL_TRIANGLES,
    GL_UNSIGNED_INT,
    glBindBuffer,
    glBindVertexArray,
    glBufferData,
    glDrawElements,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenVertexArrays,
    glVertexAttribPointer,
)

from btl2.scene.object_loader import MeshData


@dataclass
class GLMesh:
    """GPU-side mesh buffers with one VAO and packed position/normal VBO."""

    vao: int
    vbo: int
    ebo: int
    index_count: int

    def draw(self) -> None:
        """Issue the indexed draw call for this mesh."""
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)


def upload_mesh(mesh: MeshData) -> GLMesh:
    """Pack one mesh into a VAO with position and normal attributes."""
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    ebo = glGenBuffers(1)
    interleaved = np.hstack((mesh.vertices, mesh.normals)).astype(np.float32)

    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.nbytes, mesh.indices, GL_STATIC_DRAW)

    stride = 6 * 4
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, False, stride, None)
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, False, stride, ctypes.c_void_p(12))
    glBindVertexArray(0)
    return GLMesh(vao=vao, vbo=vbo, ebo=ebo, index_count=int(mesh.indices.size))


import ctypes  # noqa: E402  # Imported late so pointer helper sits near usage.
