"""RGB shading pass with a simple directional-light Phong model."""

from __future__ import annotations

from pathlib import Path

from OpenGL.GL import GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, glBindVertexArray, glClear, glClearColor, glEnable, glViewport

from btl2.renderer.camera import CameraMatrices
from btl2.renderer.material import Material
from btl2.renderer.mesh import GLMesh
from btl2.renderer.shader import ShaderProgram
from btl2.scene.scene import Scene
from btl2.scene.scene_object import SceneObject


class RGBRenderPass:
    """Render the scene into an RGB image."""

    def __init__(self, shader_dir: str | Path) -> None:
        self.shader = ShaderProgram(Path(shader_dir) / "rgb.vert", Path(shader_dir) / "rgb.frag")

    def render(self, scene: Scene, camera: CameraMatrices, target, meshes: dict[str, GLMesh], materials: dict[int, Material]) -> None:
        """Render all scene objects with lighting enabled."""
        target.bind()
        glViewport(0, 0, camera.width, camera.height)
        glEnable(GL_DEPTH_TEST)
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

        for obj in scene.objects:
            self._draw_object(obj, meshes[obj.mesh_key], materials[obj.instance_id])
        glBindVertexArray(0)

    def _draw_object(self, obj: SceneObject, mesh: GLMesh, material: Material) -> None:
        """Upload per-object uniforms and draw one mesh."""
        self.shader.set_mat4("u_model", obj.model_matrix)
        self.shader.set_vec3("u_base_color", material.base_color)
        mesh.draw()
