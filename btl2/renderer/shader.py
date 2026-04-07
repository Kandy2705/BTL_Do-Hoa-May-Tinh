"""GLSL program compilation helpers."""

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
    """Small wrapper around an OpenGL shader program and its uniforms."""

    def __init__(self, vertex_path: str | Path, fragment_path: str | Path) -> None:
        vertex_source = Path(vertex_path).read_text(encoding="utf-8")
        fragment_source = Path(fragment_path).read_text(encoding="utf-8")
        vshader = compileShader(vertex_source, GL_VERTEX_SHADER)
        fshader = compileShader(fragment_source, GL_FRAGMENT_SHADER)
        # OpenGL core profile (notably macOS) may fail program validation
        # with "No vertex array object bound" although shaders are valid.
        # Link without immediate runtime-state validation to keep setup stable.
        try:
            self.program = compileProgram(vshader, fshader, validate=False)
        except TypeError:
            self.program = compileProgram(vshader, fshader)

    def use(self) -> None:
        """Bind this program for subsequent draw calls."""
        glUseProgram(self.program)

    def set_mat4(self, name: str, matrix: np.ndarray) -> None:
        """Upload one 4x4 float matrix uniform."""
        location = glGetUniformLocation(self.program, name)
        # The codebase composes matrices in row-major style (same as BTL1).
        # Upload with transpose=True so shader receives the intended transform.
        glUniformMatrix4fv(location, 1, GL_TRUE, matrix.astype(np.float32))

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
