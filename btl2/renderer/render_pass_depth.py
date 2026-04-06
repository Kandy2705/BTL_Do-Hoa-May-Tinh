"""Depth-only pass used to export aligned depth maps."""

from __future__ import annotations

from pathlib import Path

from OpenGL.GL import GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, glBindVertexArray, glClear, glClearColor, glEnable, glViewport

from btl2.renderer.camera import CameraMatrices
from btl2.renderer.mesh import GLMesh
from btl2.renderer.shader import ShaderProgram
from btl2.scene.scene import Scene


class DepthRenderPass:
    """Render scene depth while keeping the same camera and geometry."""

    def __init__(self, shader_dir: str | Path) -> None:
        self.shader = ShaderProgram(Path(shader_dir) / "depth.vert", Path(shader_dir) / "depth.frag")

    def render(self, scene: Scene, camera: CameraMatrices, target, meshes: dict[str, GLMesh]) -> None:
        """Render all objects so the depth buffer matches the RGB pass."""
        target.bind()
        glViewport(0, 0, camera.width, camera.height)
        glEnable(GL_DEPTH_TEST)
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if meshes:
            glBindVertexArray(next(iter(meshes.values())).vao)
        self.shader.use()
        self.shader.set_mat4("u_view", camera.view)
        self.shader.set_mat4("u_projection", camera.projection)
        self.shader.set_float("u_near", camera.near)
        self.shader.set_float("u_far", camera.far)

        for obj in scene.objects:
            self.shader.set_mat4("u_model", obj.model_matrix)
            meshes[obj.mesh_key].draw()
        glBindVertexArray(0)
