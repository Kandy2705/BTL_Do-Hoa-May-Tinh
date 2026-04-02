#Hình mũi tên
import numpy as np
import OpenGL.GL as GL
import sys
import os
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.shader import Shader
from libs.buffer import VAO, UManager
from libs.lighting import LightingManager

# Import base shape
from base_shape import BaseShape


class Arrow(BaseShape):
    def __init__(self, vert_shader, frag_shader):
        """Khởi tạo hình mũi tên 2D với shader được chỉ định"""
        super().__init__()  # Initialize transform from BaseShape
        self.vert_shader = vert_shader
        self.frag_shader = frag_shader
        self.vertices = self._generate_arrow()  # Tạo đỉnh mũi tên
        
        self.use_flat_color = False  # Có dùng màu phẳng không
        self.flat_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # Màu phẳng mặc định (trắng)
        self.use_texture = False  # Có dùng texture không
        self.texture_id = None  # ID texture
        self.render_mode = 2  # 2D nên mặc định là 2 (Phong Shading)

        self.vertices = np.array(self.vertices, dtype=np.float32)

        if not hasattr(self, 'normals') or self.normals is None:
            # Vector pháp tuyến cho 2D: hướng ra ngoài mặt phẳng (0, 0, 1)
            self.normals = np.array([[0.0, 0.0, 1.0]] * len(self.vertices), dtype=np.float32)
            
        if not hasattr(self, 'colors') or self.colors is None:
            # Màu mặc định cho tất cả đỉnh: trắng
            self.colors = np.array([[1.0, 1.0, 1.0]] * len(self.vertices), dtype=np.float32)

        if not hasattr(self, 'texcoords') or self.texcoords is None:
            # Tự động sinh tọa độ texture UV từ tọa độ vertex
            if len(self.vertices) > 0:
                # Chuẩn hóa tọa độ x, y về [0, 1] cho texture mapping
                u = self.vertices[:, 0] * 0.5 + 0.5  # x: [-2, 2] -> [0, 1]
                v = self.vertices[:, 1] * 0.5 + 0.5  # y: [-2, 2] -> [0, 1]
                self.texcoords = np.column_stack((u, v)).astype(np.float32)
            else:
                self.texcoords = np.zeros((0, 2), dtype=np.float32)

        # Khởi tạo VAO, shader và uniform manager
        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.lighting = LightingManager(self.uma)

    def _generate_arrow(self):
        """Tạo các đỉnh của hình mũi tên 2D"""
        vertices = np.array([
            [-2, -1, 0],  # Đỉnh 1: thân trái dưới
            [-2,  1, 0],  # Đỉnh 2: thân trái trên
            [0, -1, 0],  # Đỉnh 3: thân phải dưới
            [0, 1, 0],   # Đỉnh 4: thân phải trên
            [0, 2, 0],   # Đỉnh 5: mũi tên trên
            [2, 0, 0],   # Đỉnh 6: mũi tên phải
            [0, -2, 0],  # Đỉnh 7: mũi tên trái
        ], dtype=np.float32)
        return vertices

    def setup(self):
        """Thiết lập VAO với các buffer (vertices, colors, normals, texcoords)"""
        # Gán dữ liệu vào các VBO tại locations:
        # Location 0: vertices (position)
        # Location 1: colors
        # Location 2: normals
        # Location 3: texture coordinates
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)
        # Thêm EBO nếu có indices (chỉ số thứ tự vẽ)
        if hasattr(self, 'indices') and self.indices is not None and len(self.indices) > 0:
            self.vao.add_ebo(self.indices)
        return self

    def set_texture(self, filepath):
        """Tải và thiết lập texture từ file hình ảnh"""
        if not filepath:
            self.use_texture = False
            return
        try:
            # Mở file ảnh và chuyển sang định dạng RGBA
            img = Image.open(filepath).convert("RGBA")
            # Lật ảnh theo chiều dọc (OpenGL yêu cầu origin ở dưới cùng)
            #img = img.transpose(Image.FLIP_TOP_BOTTOM)
            # Chuyển ảnh sang raw bytes
            # "raw" = lấy dữ liệu pixel thô, "RGBA" = định dạng màu, 0 = bước nhảy, -1 = số hàng
            img_data = img.tobytes("raw", "RGBA", 0, -1)
            
            # Tạo texture ID nếu chưa có
            if self.texture_id is None:
                self.texture_id = GL.glGenTextures(1)
            
            # Bind texture và thiết lập tham số
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
            # Thiết lập wrap mode (lặp lại texture khi vượt beyond)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
            # Thiết lập filter (nội suy khi scale texture)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            # Tải dữ liệu ảnh lên GPU
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, img.width, img.height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_data)
            # Unbind texture
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            self.use_texture = True
        except Exception as e:
            print(f"Lỗi load texture: {e}")
            self.use_texture = False

    def draw(self, projection, view, model=None):
        """Vẽ hình mũi tên với shader và uniforms"""
        # Kích hoạt shader program
        GL.glUseProgram(self.shader.render_idx)
        
        # Tính ma trận biến đổi cuối cùng
        object_transform = self.get_transform_matrix()
        final_model = object_transform @ (model if model is not None else np.identity(4, dtype=np.float32))
        modelview = view @ final_model
        
        # Upload ma trận projection và modelview lên shader
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        
        # Upload view matrix cho light transform
        loc_view = GL.glGetUniformLocation(self.shader.render_idx, "view")
        if loc_view != -1: self.uma.upload_uniform_matrix4fv(view, 'view', True)
        
        # --- 1. Truyền công tắc Flat Color ---
        loc_flat = GL.glGetUniformLocation(self.shader.render_idx, "u_use_flat_color")
        if loc_flat != -1: GL.glUniform1i(loc_flat, 1 if self.use_flat_color else 0)
        
        loc_flat_col = GL.glGetUniformLocation(self.shader.render_idx, "u_flat_color")
        if loc_flat_col != -1: 
            GL.glUniform3f(loc_flat_col, self.flat_color[0], self.flat_color[1], self.flat_color[2])
        
        # --- 2. Truyền công tắc Texture ---
        loc_tex = GL.glGetUniformLocation(self.shader.render_idx, "u_use_texture")
        if loc_tex != -1: GL.glUniform1i(loc_tex, 1 if self.use_texture else 0)
        
        # --- 3. Truyền chế độ Render (0: None, 1: Gouraud, 2: Phong) ---
        loc_mode = GL.glGetUniformLocation(self.shader.render_idx, "u_render_mode")
        if loc_mode != -1: GL.glUniform1i(loc_mode, self.render_mode)
        
        # Bind texture nếu có
        if self.use_texture and self.texture_id is not None:
            GL.glActiveTexture(GL.GL_TEXTURE0)  # Kích hoạt texture unit 0
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
            loc_sampler = GL.glGetUniformLocation(self.shader.render_idx, "u_texture")
            if loc_sampler != -1: GL.glUniform1i(loc_sampler, 0)
        
        # --- HỆ THỐNG ĐA NGUỒN SÁNG (MULTI-LIGHTING) ---
        lights = getattr(self, 'scene_lights', [])
        loc_num_lights = GL.glGetUniformLocation(self.shader.render_idx, "u_num_lights")
        if loc_num_lights != -1: GL.glUniform1i(loc_num_lights, len(lights)) # Hàm gửi dữ liệu từ CPU lên GPU shader
        
        for i, l in enumerate(lights[:4]): # Hỗ trợ tối đa 4 nguồn sáng cùng lúc
            GL.glUniform3f(GL.glGetUniformLocation(self.shader.render_idx, f"u_light_pos[{i}]"), *l.position)
            GL.glUniform3f(GL.glGetUniformLocation(self.shader.render_idx, f"u_light_color[{i}]"), *l.light_color)
            GL.glUniform1f(GL.glGetUniformLocation(self.shader.render_idx, f"u_light_intensity[{i}]"), l.light_intensity)
            GL.glUniform1i(GL.glGetUniformLocation(self.shader.render_idx, f"u_light_active[{i}]"), 1 if l.visible else 0)

        # Kích hoạt VAO và vẽ
        self.vao.activate()
        
        if hasattr(self, 'indices') and self.indices is not None and len(self.indices) > 0:
            # Vẽ theo indices (nếu có)
            GL.glDrawElements(GL.GL_TRIANGLES, len(self.indices), GL.GL_UNSIGNED_INT, None)
        else:
            # Vẽ theo mảng vertices (nếu không có indices)
            # Chọn mode phù hợp: TRIANGLES cho 3 đỉnh, TRIANGLE_FAN cho nhiều đỉnh
            mode = GL.GL_TRIANGLES if len(self.vertices) == 3 else GL.GL_TRIANGLE_FAN
            GL.glDrawArrays(mode, 0, self.vertices.shape[0])
            
        # Deactivate VAO
        self.vao.deactivate()
        
        # Unbind texture sau khi vẽ xong
        if self.use_texture:
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def set_color(self, color):
        """Thiết lập màu cho tất cả các đỉnh của hình mũi tên"""
        # Tạo mảng màu mới với màu được chỉ định cho tất cả các đỉnh
        self.colors = np.array([color] * len(self.vertices), dtype=np.float32)
        
        # Cập nhật VBO màu trên GPU
        self.vao.activate()
        buffer_idx = self.vao.vbo[1]  # Location 1 là color buffer
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_idx)
        # Tải dữ liệu màu mới lên GPU
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.colors, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        # Cập nhật cả flat color để nhất quán
        self.flat_color = np.array(color[:3], dtype=np.float32)

    def set_solid_color(self, color):
        """Bật chế độ màu phẳng và thiết lập màu"""
        self.use_flat_color = True  # Bật chế độ flat color
        self.flat_color = np.array(color[:3], dtype=np.float32)  # Chỉ lấy RGB

    def cleanup(self):
        """Dọn dẹp tài nguyên GPU khi object bị hủy"""
        # Xóa VAO (tự động xóa các VBO liên quan)
        if hasattr(self, 'vao'): 
            self.vao.delete()
        # Xóa shader program
        if hasattr(self, 'shader'): 
            self.shader.delete()
        # Xóa texture nếu có
        if hasattr(self, 'texture_id') and self.texture_id is not None:
            GL.glDeleteTextures(1, [self.texture_id])