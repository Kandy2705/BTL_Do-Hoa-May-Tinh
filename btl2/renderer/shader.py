"""Helper biên dịch GLSL shader program và upload uniform."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from OpenGL.GL import (
    GL_FALSE,
    GL_TRUE,
    GL_FRAGMENT_SHADER,
    GL_VERTEX_SHADER,
    glGetUniformLocation,
    glUniform1f,
    glUniform1i,
    glUniform3fv,
    glUniformMatrix4fv,
    glUseProgram,
)
from OpenGL.GL.shaders import compileProgram, compileShader


class ShaderProgram:
    """Wrapper nhỏ quanh OpenGL shader program dùng trong các render pass."""

    def __init__(self, vertex_path: str | Path, fragment_path: str | Path) -> None:
        # Mỗi pass BTL 2 có cặp vertex/fragment shader riêng trong `shaders/btl2`.
        vertex_source = Path(vertex_path).read_text(encoding="utf-8")
        fragment_source = Path(fragment_path).read_text(encoding="utf-8")
        vshader = compileShader(vertex_source, GL_VERTEX_SHADER)
        fshader = compileShader(fragment_source, GL_FRAGMENT_SHADER)
        # OpenGL core profile, đặc biệt trên macOS, có thể fail validation với lỗi
        # "No vertex array object bound" dù shader hợp lệ. Link không validate ngay
        # để setup ổn định; VAO sẽ được bind trong render pass.
        try:
            self.program = compileProgram(vshader, fshader, validate=False)
        except TypeError:
            self.program = compileProgram(vshader, fshader)

    def use(self) -> None:
        """Bind program này cho các draw call tiếp theo."""
        glUseProgram(self.program)

    def set_mat4(self, name: str, matrix: np.ndarray) -> None:
        """Upload một uniform ma trận 4x4."""
        location = glGetUniformLocation(self.program, name)
        # Codebase ghép ma trận theo kiểu row-major giống BTL 1. Upload với
        # transpose=True để shader nhận đúng transform mong muốn.
        glUniformMatrix4fv(location, 1, GL_TRUE, matrix.astype(np.float32))

    def set_vec3(self, name: str, vector: np.ndarray) -> None:
        """Upload một uniform vec3."""
        location = glGetUniformLocation(self.program, name)
        glUniform3fv(location, 1, vector.astype(np.float32))

    def set_float(self, name: str, value: float) -> None:
        """Upload một uniform float."""
        location = glGetUniformLocation(self.program, name)
        glUniform1f(location, float(value))

    def set_int(self, name: str, value: int) -> None:
        """Upload một uniform int."""
        location = glGetUniformLocation(self.program, name)
        glUniform1i(location, int(value))
