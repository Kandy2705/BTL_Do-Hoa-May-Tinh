"""Offscreen framebuffer used by each render pass."""

from __future__ import annotations

import numpy as np
from OpenGL.GL import (
    GL_CLAMP_TO_EDGE,
    GL_COLOR_ATTACHMENT0,
    GL_DEPTH_ATTACHMENT,
    GL_DEPTH_COMPONENT,
    GL_DEPTH_COMPONENT24,
    GL_FLOAT,
    GL_FRAMEBUFFER,
    GL_FRAMEBUFFER_COMPLETE,
    GL_LINEAR,
    GL_RENDERBUFFER,
    GL_RGB,
    GL_RGB8,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_UNSIGNED_BYTE,
    glBindFramebuffer,
    glBindRenderbuffer,
    glBindTexture,
    glCheckFramebufferStatus,
    glFramebufferRenderbuffer,
    glFramebufferTexture2D,
    glGenFramebuffers,
    glGenRenderbuffers,
    glGenTextures,
    glReadPixels,
    glRenderbufferStorage,
    glTexImage2D,
    glTexParameteri,
)


class RenderTarget:
    """Color texture plus depth renderbuffer for one offscreen pass."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.fbo = glGenFramebuffers(1)
        self.color_texture = glGenTextures(1)
        self.depth_rbo = glGenRenderbuffers(1)

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

        glBindTexture(GL_TEXTURE_2D, self.color_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.color_texture, 0)

        glBindRenderbuffer(GL_RENDERBUFFER, self.depth_rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depth_rbo)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Offscreen framebuffer is incomplete.")

    def bind(self) -> None:
        """Bind this framebuffer before rendering."""
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

    def read_rgb(self) -> np.ndarray:
        """Read back the color attachment and flip it to top-left image order."""
        raw = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
        return np.flipud(image).copy()

    def read_depth(self) -> np.ndarray:
        """Read the depth buffer as float values and flip to image order."""
        raw = glReadPixels(0, 0, self.width, self.height, GL_DEPTH_COMPONENT, GL_FLOAT)
        image = np.frombuffer(raw, dtype=np.float32).reshape((self.height, self.width))
        return np.flipud(image).copy()
