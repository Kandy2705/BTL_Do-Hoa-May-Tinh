"""Cửa sổ GLFW ẩn dùng để sở hữu OpenGL context cho render offscreen."""

from __future__ import annotations

import glfw


class OffscreenWindow:
    """Tạo cửa sổ ẩn để BTL 2 render mà không cần mở viewport riêng."""

    def __init__(self, width: int, height: int, title: str = "SyntheticRoadGenerator") -> None:
        self.width = width
        self.height = height
        self.title = title
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed. Make sure OpenGL is available on this machine.")
        # Window ẩn vẫn tạo được context OpenGL đầy đủ, nhưng không hiện UI.
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self._window = glfw.create_window(width, height, title, None, None)
        if self._window is None:
            glfw.terminate()
            raise RuntimeError("Failed to create hidden GLFW window.")
        glfw.make_context_current(self._window)

    def make_current(self) -> None:
        """Kích hoạt context này trước khi gọi bất kỳ hàm OpenGL nào."""
        glfw.make_context_current(self._window)

    def destroy(self) -> None:
        """Hủy window ẩn và terminate GLFW khi BTL 2 tự sở hữu context."""
        if self._window is not None:
            glfw.destroy_window(self._window)
            self._window = None
        glfw.terminate()
