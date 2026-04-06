"""Hidden GLFW window used to own the OpenGL context."""

from __future__ import annotations

import glfw


class OffscreenWindow:
    """Create a hidden GLFW window so rendering can happen offscreen."""

    def __init__(self, width: int, height: int, title: str = "SyntheticRoadGenerator") -> None:
        self.width = width
        self.height = height
        self.title = title
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed. Make sure OpenGL is available on this machine.")
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
        """Activate this context before any GL call."""
        glfw.make_context_current(self._window)

    def destroy(self) -> None:
        """Release window and terminate GLFW."""
        if self._window is not None:
            glfw.destroy_window(self._window)
            self._window = None
        glfw.terminate()
