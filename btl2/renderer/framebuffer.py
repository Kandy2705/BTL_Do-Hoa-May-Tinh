"""Framebuffer offscreen cho từng render pass của BTL 2.

BTL 2 cần render mà không nhất thiết hiển thị ra cửa sổ. Vì vậy mỗi pass ghi vào
FBO riêng, sau đó đọc lại pixel thành numpy array để lưu ảnh hoặc tính annotation.
"""

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
    """Một target gồm color texture và depth renderbuffer cho một pass offscreen."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        # FBO giữ framebuffer, color_texture chứa ảnh màu đọc ra được, depth_rbo
        # chỉ dùng cho depth test trong lúc render.
        self.fbo = glGenFramebuffers(1)
        self.color_texture = glGenTextures(1)
        self.depth_rbo = glGenRenderbuffers(1)

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

        # Color attachment dùng RGB8 vì output cuối cùng là ảnh 8-bit thông thường.
        glBindTexture(GL_TEXTURE_2D, self.color_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.color_texture, 0)

        # Depth renderbuffer giúp OpenGL che khuất object đúng như render thật.
        glBindRenderbuffer(GL_RENDERBUFFER, self.depth_rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depth_rbo)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Offscreen framebuffer is incomplete.")

    def bind(self) -> None:
        """Bind framebuffer này trước khi một render pass bắt đầu vẽ."""
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

    def read_rgb(self) -> np.ndarray:
        """Đọc color attachment và lật ảnh về hệ tọa độ top-left của ảnh PNG."""
        raw = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
        # OpenGL gốc tọa độ ảnh ở góc dưới trái, còn thư viện ảnh/annotation dùng
        # góc trên trái. Nếu không flip, mask và bbox sẽ bị ngược dọc.
        return np.flipud(image).copy()

    def read_depth(self) -> np.ndarray:
        """Đọc depth buffer dạng float và lật về hệ tọa độ ảnh."""
        raw = glReadPixels(0, 0, self.width, self.height, GL_DEPTH_COMPONENT, GL_FLOAT)
        image = np.frombuffer(raw, dtype=np.float32).reshape((self.height, self.width))
        return np.flipud(image).copy()
