"""Render pass RGB: tạo ảnh màu giống camera thật nhất trong pipeline BTL 2."""

from __future__ import annotations

from pathlib import Path

from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    glActiveTexture,
    glBindTexture,
    glBindVertexArray,
    glClear,
    glClearColor,
    glEnable,
    glViewport,
)

from btl2.renderer.camera import CameraMatrices
from btl2.renderer.material import Material
from btl2.renderer.mesh import GLMesh
from btl2.renderer.shader import ShaderProgram
from btl2.scene.scene import Scene
from btl2.scene.scene_object import SceneObject


class RGBRenderPass:
    """Vẽ scene thành ảnh RGB, có texture và ánh sáng."""

    def __init__(self, shader_dir: str | Path) -> None:
        self.shader = ShaderProgram(Path(shader_dir) / "rgb.vert", Path(shader_dir) / "rgb.frag")

    def render(self, scene: Scene, camera: CameraMatrices, target, meshes: dict[str, GLMesh], materials: dict[int, Material]) -> None:
        """Render toàn bộ object với cùng camera của depth/segmentation pass."""
        target.bind()
        glViewport(0, 0, camera.width, camera.height)
        glEnable(GL_DEPTH_TEST)
        # Nền trời lấy từ scene config để ảnh sinh ra có màu ổn định theo dataset.
        glClearColor(*scene.background_color.tolist(), 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if meshes:
            glBindVertexArray(next(iter(meshes.values())).vao)
        self.shader.use()
        self.shader.set_mat4("u_view", camera.view)
        self.shader.set_mat4("u_projection", camera.projection)
        self.shader.set_vec3("u_camera_pos", camera.position)
        self.shader.set_vec3("u_light_dir", scene.light.direction)
        self.shader.set_vec3("u_light_color", scene.light.color)
        self.shader.set_float("u_light_intensity", scene.light.intensity)
        self.shader.set_float("u_ambient_strength", scene.light.ambient_strength)
        self.shader.set_int("u_texture", 0)
        self.shader.set_float("u_texture_brightness", 1.0)
        self.shader.set_float("u_texture_saturation", 1.0)

        for obj in scene.objects:
            # Mỗi object có model matrix và base color riêng; mesh có thể dùng lại
            # giữa nhiều instance cùng class.
            self._draw_object(obj, meshes[obj.mesh_key], materials[obj.instance_id])
        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)

    def _draw_object(self, obj: SceneObject, mesh: GLMesh, material: Material) -> None:
        """Đẩy uniform riêng của object rồi vẽ mesh tương ứng."""
        self.shader.set_mat4("u_model", obj.model_matrix)
        self.shader.set_vec3("u_base_color", material.base_color)
        use_texture = mesh.texture_id is not None
        self.shader.set_int("u_use_texture", 1 if use_texture else 0)
        brightness, saturation = self._texture_adjustment(obj)
        self.shader.set_float("u_texture_brightness", brightness)
        self.shader.set_float("u_texture_saturation", saturation)
        if mesh.material_groups and mesh.material_texture_ids:
            # Với OBJ nhiều material, vẽ từng đoạn index để bind đúng texture.
            drew_group = False
            for group in mesh.material_groups:
                texture_id = mesh.material_texture_ids.get(group.get("material"))
                self.shader.set_int("u_use_texture", 1 if texture_id is not None else 0)
                if texture_id is not None:
                    glActiveTexture(GL_TEXTURE0)
                    glBindTexture(GL_TEXTURE_2D, texture_id)
                else:
                    glBindTexture(GL_TEXTURE_2D, 0)
                mesh.draw_range(int(group.get("start", 0)), int(group.get("count", 0)))
                drew_group = True
            if drew_group:
                return

        if use_texture:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, mesh.texture_id)
        else:
            glBindTexture(GL_TEXTURE_2D, 0)
        mesh.draw()

    @staticmethod
    def _texture_adjustment(obj: SceneObject) -> tuple[float, float]:
        """Làm backdrop city/intersection sáng hơn mà không đổi màu xe/người."""
        name = f"{obj.name} {obj.metadata.get('original_name', '')} {obj.metadata.get('source_asset', '')}".lower()
        if any(token in name for token in ("city", "intersection", "building", "gas_station", "gas station")):
            return 1.20, 1.28
        return 1.0, 1.0
