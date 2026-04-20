"""Mesh upload and draw wrapper."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_ELEMENT_ARRAY_BUFFER,
    GL_FLOAT,
    GL_STATIC_DRAW,
    GL_LINEAR,
    GL_TRIANGLES,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_REPEAT,
    GL_RGBA,
    GL_UNSIGNED_BYTE,
    GL_UNSIGNED_INT,
    glActiveTexture,
    glBindBuffer,
    glBindTexture,
    glBindVertexArray,
    glBufferData,
    glDrawElements,
    glEnableVertexAttribArray,
    glGenTextures,
    glTexImage2D,
    glTexParameteri,
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
    texture_id: int | None = None
    material_groups: list[dict] | None = None
    material_texture_ids: dict[str, int] | None = None

    def draw(self) -> None:
        """Issue the indexed draw call for this mesh."""
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)

    def draw_range(self, start: int, count: int) -> None:
        """Draw a contiguous index range, used for OBJ material groups."""
        if count <= 0:
            return
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, int(count), GL_UNSIGNED_INT, ctypes.c_void_p(int(start) * 4))


def _load_texture(texture_path) -> int | None:
    """Load a diffuse image file into an OpenGL texture."""
    if texture_path is None:
        return None
    try:
        from PIL import Image

        image = Image.open(texture_path).convert("RGBA")
        image_data = image.tobytes("raw", "RGBA", 0, -1)
        texture_id = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
        glBindTexture(GL_TEXTURE_2D, 0)
        return int(texture_id)
    except Exception as exc:
        print(f"BTL2 texture load failed for {texture_path}: {exc}")
        return None


def upload_mesh(mesh: MeshData) -> GLMesh:
    """Pack one mesh into a VAO with position, normal, and UV attributes."""
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    ebo = glGenBuffers(1)
    texcoords = mesh.texcoords
    if texcoords is None or texcoords.shape != (mesh.vertices.shape[0], 2):
        texcoords = np.zeros((mesh.vertices.shape[0], 2), dtype=np.float32)
    interleaved = np.hstack((mesh.vertices, mesh.normals, texcoords)).astype(np.float32)

    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.nbytes, mesh.indices, GL_STATIC_DRAW)

    stride = 8 * 4
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, False, stride, None)
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, False, stride, ctypes.c_void_p(12))
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 2, GL_FLOAT, False, stride, ctypes.c_void_p(24))
    glBindVertexArray(0)

    texture_id = _load_texture(mesh.texture_path)
    material_texture_ids: dict[str, int] = {}
    for material_name, material_texture_path in (mesh.material_texture_paths or {}).items():
        if texture_id is not None and material_texture_path == mesh.texture_path:
            material_texture_ids[material_name] = texture_id
            continue
        material_texture_id = _load_texture(material_texture_path)
        if material_texture_id is not None:
            material_texture_ids[material_name] = material_texture_id
    if texture_id is None and material_texture_ids:
        texture_id = next(iter(material_texture_ids.values()))

    return GLMesh(
        vao=vao,
        vbo=vbo,
        ebo=ebo,
        index_count=int(mesh.indices.size),
        texture_id=texture_id,
        material_groups=mesh.material_groups,
        material_texture_ids=material_texture_ids or None,
    )
