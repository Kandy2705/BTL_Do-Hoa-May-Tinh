"""Đưa mesh từ CPU lên GPU và cung cấp hàm draw cho render pass."""

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
    """Mesh phía GPU: VAO/VBO/EBO, texture và nhóm material nếu có."""

    vao: int
    vbo: int
    ebo: int
    index_count: int
    texture_id: int | None = None
    material_groups: list[dict] | None = None
    material_texture_ids: dict[str, int] | None = None

    def draw(self) -> None:
        """Vẽ toàn bộ mesh bằng index buffer."""
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)

    def draw_range(self, start: int, count: int) -> None:
        """Vẽ một đoạn index liên tục, dùng cho OBJ có nhiều material."""
        if count <= 0:
            return
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, int(count), GL_UNSIGNED_INT, ctypes.c_void_p(int(start) * 4))


def _load_texture(texture_path) -> int | None:
    """Nạp ảnh diffuse thành OpenGL texture."""
    if texture_path is None:
        return None
    try:
        from PIL import Image

        # Chuyển sang RGBA để mọi loại ảnh đầu vào có số kênh thống nhất khi upload.
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
    """Đóng gói mesh thành VAO với attribute position, normal và UV."""
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    ebo = glGenBuffers(1)
    texcoords = mesh.texcoords
    if texcoords is None or texcoords.shape != (mesh.vertices.shape[0], 2):
        # Primitive fallback có thể không có UV; dùng UV 0 để shader vẫn chạy.
        texcoords = np.zeros((mesh.vertices.shape[0], 2), dtype=np.float32)
    # Layout mỗi vertex: 3 float vị trí + 3 float normal + 2 float UV = 8 float.
    interleaved = np.hstack((mesh.vertices, mesh.normals, texcoords)).astype(np.float32)

    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.nbytes, mesh.indices, GL_STATIC_DRAW)

    stride = 8 * 4
    # location 0/1/2 phải khớp với layout trong shader `shaders/btl2/*.vert`.
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, False, stride, None)
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, False, stride, ctypes.c_void_p(12))
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 2, GL_FLOAT, False, stride, ctypes.c_void_p(24))
    glBindVertexArray(0)

    texture_id = _load_texture(mesh.texture_path)
    material_texture_ids: dict[str, int] = {}
    # Một số OBJ có nhiều material, mỗi material một texture. Lưu map này để
    # RGBRenderPass bind đúng texture cho từng index range.
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
