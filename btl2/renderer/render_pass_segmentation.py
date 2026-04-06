"""Segmentation pass that renders a stable color per object instance."""

from __future__ import annotations

from pathlib import Path

from OpenGL.GL import GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, glBindVertexArray, glClear, glClearColor, glEnable, glViewport

from btl2.renderer.camera import CameraMatrices
from btl2.renderer.material import Material
from btl2.renderer.mesh import GLMesh
from btl2.renderer.shader import ShaderProgram
from btl2.scene.scene import Scene


class SegmentationRenderPass:
    """Render class or instance colors into a mask image."""

    def __init__(self, shader_dir: str | Path) -> None:
        self.shader = ShaderProgram(Path(shader_dir) / "seg.vert", Path(shader_dir) / "seg.frag")

    def render(self, scene: Scene, camera: CameraMatrices, target, meshes: dict[str, GLMesh], materials: dict[int, Material]) -> None:
        """Render every object with its segmentation color."""
        target.bind()
        glViewport(0, 0, camera.width, camera.height)
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if meshes:
            glBindVertexArray(next(iter(meshes.values())).vao)
        self.shader.use()
        self.shader.set_mat4("u_view", camera.view)
        self.shader.set_mat4("u_projection", camera.projection)

        for obj in scene.objects:
            self.shader.set_mat4("u_model", obj.model_matrix)
            self.shader.set_vec3("u_color", materials[obj.instance_id].segmentation_color)
            meshes[obj.mesh_key].draw()
        glBindVertexArray(0)
