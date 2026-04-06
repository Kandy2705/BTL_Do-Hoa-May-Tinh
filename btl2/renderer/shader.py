"""GLSL program compilation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from OpenGL.GL import (
    GL_FALSE,
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
    """Small wrapper around an OpenGL shader program and its uniforms."""

    def __init__(self, vertex_path: str | Path, fragment_path: str | Path) -> None:
        vertex_source = Path(vertex_path).read_text(encoding="utf-8")
        fragment_source = Path(fragment_path).read_text(encoding="utf-8")
        self.program = compileProgram(
            compileShader(vertex_source, GL_VERTEX_SHADER),
            compileShader(fragment_source, GL_FRAGMENT_SHADER),
        )

    def use(self) -> None:
        """Bind this program for subsequent draw calls."""
        glUseProgram(self.program)

    def set_mat4(self, name: str, matrix: np.ndarray) -> None:
        """Upload one 4x4 float matrix uniform."""
        location = glGetUniformLocation(self.program, name)
        glUniformMatrix4fv(location, 1, GL_FALSE, matrix.astype(np.float32))

    def set_vec3(self, name: str, vector: np.ndarray) -> None:
        """Upload one vec3 uniform."""
        location = glGetUniformLocation(self.program, name)
        glUniform3fv(location, 1, vector.astype(np.float32))

    def set_float(self, name: str, value: float) -> None:
        """Upload one float uniform."""
        location = glGetUniformLocation(self.program, name)
        glUniform1f(location, float(value))

    def set_int(self, name: str, value: int) -> None:
        """Upload one integer uniform."""
        location = glGetUniformLocation(self.program, name)
        glUniform1i(location, int(value))
