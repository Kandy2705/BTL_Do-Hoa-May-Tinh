"""Render pass segmentation: mã hóa class/instance thành màu ổn định."""

from __future__ import annotations

from pathlib import Path

from OpenGL.GL import GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, glBindVertexArray, glClear, glClearColor, glEnable, glViewport

from btl2.renderer.camera import CameraMatrices
from btl2.renderer.material import Material
from btl2.renderer.mesh import GLMesh
from btl2.renderer.shader import ShaderProgram
from btl2.scene.scene import Scene


class SegmentationRenderPass:
    """Vẽ scene thành mask RGB, mỗi object nhận một màu định danh."""

    def __init__(self, shader_dir: str | Path) -> None:
        self.shader = ShaderProgram(Path(shader_dir) / "seg.vert", Path(shader_dir) / "seg.frag")

    def render(self, scene: Scene, camera: CameraMatrices, target, meshes: dict[str, GLMesh], materials: dict[int, Material]) -> None:
        """Render mọi object bằng màu segmentation thay vì màu/texture thật."""
        target.bind()
        glViewport(0, 0, camera.width, camera.height)
        glEnable(GL_DEPTH_TEST)
        # Nền đen biểu thị không có object; các class/instance hợp lệ dùng màu khác.
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if meshes:
            glBindVertexArray(next(iter(meshes.values())).vao)
        self.shader.use()
        self.shader.set_mat4("u_view", camera.view)
        self.shader.set_mat4("u_projection", camera.projection)

        for obj in scene.objects:
            # Depth test vẫn bật, nên mask cuối cùng chỉ giữ phần object nhìn thấy
            # sau che khuất, rất quan trọng để tính occlusion và polygon COCO.
            self.shader.set_mat4("u_model", obj.model_matrix)
            self.shader.set_vec3("u_color", materials[obj.instance_id].segmentation_color)
            meshes[obj.mesh_key].draw()
        glBindVertexArray(0)
