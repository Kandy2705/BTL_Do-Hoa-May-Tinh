import sys
import os
import numpy as np
import ctypes
import math
import warnings

# Add parent directory to path to import libs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.shader import *
from libs import transform as T
from libs.buffer import *
from libs.lighting import LightingManager
import OpenGL.GL as GL

# Import base shape
from base_shape import BaseShape


class MathematicalSurface(BaseShape):
    def __init__(self, vert_shader, frag_shader, func=None, x_range=(-5, 5), y_range=(-5, 5), resolution=50):
        super().__init__()  # Khởi tạo transform từ BaseShape
        self.vert_shader = vert_shader
        self.frag_shader = frag_shader
        self.use_custom_color = False  # Cờ sử dụng màu tùy chỉnh hoặc màu tự sinh
        self.use_flat_color = False  # Cờ ghi đè màu đơn sắc
        self.use_texture = False  # Hỗ trợ texture
        self.texture_id = None
        self.render_mode = 2  # Mặc định dùng Phong shading
        self.flat_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.original_colors = None  # Lưu trữ màu tự sinh ban đầu
        
        if func is None:
            self.func = lambda x, y: (x**2 + y - 11)**2 + (x + y**2 - 7)**2
        else:
            self.func = func
            
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        
        # Khi object được tạo, mesh của bề mặt toán học sẽ được sinh ngay từ hàm f(x, y).
        self._generate_surface()
        self._generate_texcoords()
        
    def _generate_surface(self):
        """Generate vertices, indices, normals and colors for the mathematical surface"""
        import warnings
        
        # Bước 1: tạo lưới mẫu trong miền giá trị thật của x và y.
        x_real = []
        for i in range(self.resolution):
            x = self.x_range[0] + (self.x_range[1] - self.x_range[0]) * i / (self.resolution - 1)
            x_real.append(x)
        
        y_real = []
        for i in range(self.resolution):
            y = self.y_range[0] + (self.y_range[1] - self.y_range[0]) * i / (self.resolution - 1)
            y_real.append(y)
        
        X_real = []
        Y_real = []
        for y_idx in range(len(y_real)):
            row_x = []
            row_y = []
            for x_idx in range(len(x_real)):
                row_x.append(x_real[x_idx])
                row_y.append(y_real[y_idx])
            X_real.append(row_x)
            Y_real.append(row_y)
        
        X_real = np.array(X_real, dtype=np.float32)
        Y_real = np.array(Y_real, dtype=np.float32)
        
        # Bước 2: tính z = f(x, y) trên cả lưới.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z_real = self.func(X_real, Y_real)
            
        Z_real = np.nan_to_num(Z_real, nan=0.0, posinf=100.0, neginf=-100.0)
        Z_real = np.clip(Z_real, -100.0, 100.0)
        
        # Bước 3: chuẩn hóa dữ liệu để bề mặt không quá to hoặc quá nhỏ khi render.
        def normalize_for_draw(arr, target_min=-2.0, target_max=2.0):
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max - arr_min == 0:
                return np.zeros_like(arr)
            return (arr - arr_min) / (arr_max - arr_min) * (target_max - target_min) + target_min
            
        X_draw = normalize_for_draw(X_real)
        Y_draw = normalize_for_draw(Y_real)
        Z_draw = normalize_for_draw(Z_real)
        
        vertices = []
        normals = []
        colors = []
        
        dx_draw = X_draw[0, 1] - X_draw[0, 0]
        dy_draw = Y_draw[1, 0] - Y_draw[0, 0]
        
        # Bước 4: từ lưới điểm, sinh ra:
        # - vertices để vẽ
        # - normals để chiếu sáng
        # - colors để tô theo độ cao
        for i in range(self.resolution):
            for j in range(self.resolution):
                vertices.append([X_draw[i, j], Y_draw[i, j], Z_draw[i, j]])
                
                if 0 < i < self.resolution-1 and 0 < j < self.resolution-1:
                    # Vì bề mặt có dạng z = f(x, y), pháp tuyến có thể lấy từ gradient.
                    # Về lý thuyết, nếu viết bề mặt dưới dạng F(x, y, z) = z - f(x, y) = 0
                    # thì normal tỉ lệ với [-df/dx, -df/dy, 1].
                    # Ở đây em xấp xỉ df/dx và df/dy bằng sai phân hữu hạn trung tâm.
                    # ∇F = [∂F/∂x, ∂F/∂y, ∂F/∂z] = [-df/dx, -df/dy, 1]
                    dz_dx = (Z_draw[i, j+1] - Z_draw[i, j-1]) / (2 * dx_draw)
                    dz_dy = (Z_draw[i+1, j] - Z_draw[i-1, j]) / (2 * dy_draw)
                    normal = np.array([-dz_dx, -dz_dy, 1.0])
                    normal = normal / np.linalg.norm(normal)
                else:
                    normal = np.array([0.0, 0.0, 1.0])
                normals.append(normal)
                
                # h là độ cao đã được chuẩn hóa về [0, 1].
                # Từ đó ta suy ra màu theo độ cao để nhìn trực quan phần lồi/lõm của bề mặt.
                h = (Z_draw[i, j] + 2.0) / 4.0
                colors.append([
                    0.2 + 0.8 * h,
                    0.3 + 0.4 * (1 - abs(2*h-1)),
                    0.8 - 0.6 * h
                ])
                
        self.vertices = np.array(vertices, dtype=np.float32)
        self.normals = np.array(normals, dtype=np.float32)
        self.colors = np.array(colors, dtype=np.float32)
        self.original_colors = self.colors.copy()  # Store original auto-generated colors
        
        indices = []
        for i in range(self.resolution - 1):
            for j in range(self.resolution - 1):
                # v0 = 0*4 + 0 = 0
                # v1 = 0*4 + 1 = 1  
                # v2 = 1*4 + 0 = 4
                # v3 = 1*4 + 1 = 5
                # Triangles: [0, 1, 4] và [4, 1, 5]

                v0 = i * self.resolution + j
                v1 = i * self.resolution + (j + 1)
                v2 = (i + 1) * self.resolution + j
                v3 = (i + 1) * self.resolution + (j + 1)
                
                # Mỗi ô vuông của lưới được tách thành 2 tam giác.
                # Đây là cách chuẩn để GPU render bề mặt bằng primitive tam giác.
                indices.extend([v0, v1, v2])
                indices.extend([v2, v1, v3])
        
        self.indices = np.array(indices, dtype=np.int32)

    def _generate_texcoords(self):
        # UV của MathematicalSurface được gán theo lưới đều:
        # cột -> u, hàng -> v.
        # Cách này phù hợp vì mesh bản chất đã là một grid trên miền x-y.
        texcoords = []
        for i in range(self.resolution):
            v = i / max(self.resolution - 1, 1)
            for j in range(self.resolution):
                u = j / max(self.resolution - 1, 1)
                texcoords.append([u, v])
        self.texcoords = np.array(texcoords, dtype=np.float32)
        
    def setup(self):
        """Setup buffers and shader"""
        self.vao = VAO()
        
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        
        self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(3, self.texcoords, ncomponents=2, stride=0, offset=None)
        
        self.vao.add_ebo(self.indices)
        
        self.shader = Shader(self.vert_shader, self.frag_shader)
        self.uma = UManager(self.shader)
        self.lighting = LightingManager(self.uma)
        
        return self

    def set_texture(self, filepath):
        if not filepath:
            self.use_texture = False
            return
        try:
            from PIL import Image

            img = Image.open(filepath).convert("RGBA")
            #img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img_data = img.tobytes("raw", "RGBA", 0, -1)

            if self.texture_id is None:
                self.texture_id = GL.glGenTextures(1)

            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, img.width, img.height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_data)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

            self.use_texture = True
            print(f"Đã load texture thành công cho MathematicalSurface: {filepath}")
        except Exception as e:
            print(f"Lỗi load texture cho MathematicalSurface: {e}")
            self.use_texture = False

    def draw(self, projection, view, model):
        """Draw the mathematical surface"""
        GL.glUseProgram(self.shader.render_idx)
        
        # Use BaseShape transform
        object_transform = self.get_transform_matrix()
        final_model = object_transform @ (model if model is not None else np.identity(4, dtype=np.float32))
        modelview = view @ final_model
        
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        
        # Upload view matrix for light transform
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
        
        if self.use_texture and self.texture_id is not None:
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
            loc_sampler = GL.glGetUniformLocation(self.shader.render_idx, "u_texture")
            if loc_sampler != -1: GL.glUniform1i(loc_sampler, 0)
        
        # --- HỆ THỐNG ĐA NGUỒN SÁNG (MULTI-LIGHTING) ---
        lights = getattr(self, 'scene_lights', [])
        loc_num_lights = GL.glGetUniformLocation(self.shader.render_idx, "u_num_lights")
        if loc_num_lights != -1: GL.glUniform1i(loc_num_lights, len(lights))
        
        for i, l in enumerate(lights[:4]): # Hỗ trợ tối đa 4 nguồn sáng cùng lúc
            GL.glUniform3f(GL.glGetUniformLocation(self.shader.render_idx, f"u_light_pos[{i}]"), *l.position)
            GL.glUniform3f(GL.glGetUniformLocation(self.shader.render_idx, f"u_light_color[{i}]"), *l.light_color)
            GL.glUniform1f(GL.glGetUniformLocation(self.shader.render_idx, f"u_light_intensity[{i}]"), l.light_intensity)
            GL.glUniform1i(GL.glGetUniformLocation(self.shader.render_idx, f"u_light_active[{i}]"), 1 if l.visible else 0)
        
        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLES, len(self.indices), GL.GL_UNSIGNED_INT, None)
        self.vao.deactivate()
    
    def set_color(self, color):
        """Set color for the mathematical surface - override BaseShape method"""
        self.use_custom_color = True
        self.colors = np.array([color[:3]] * len(self.vertices), dtype=np.float32)
        self.flat_color = np.array(color[:3], dtype=np.float32)
        
        # Re-setup the VBO to update colors
        self.vao.activate()
        buffer_idx = self.vao.vbo[1]  # Get the color VBO at location 1
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_idx)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.colors, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        
        # Cleanup texture if needed
        if self.use_texture:
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    
    def set_color_mode(self, use_custom_color):
        """Toggle between auto-color and custom color mode"""
        self.use_custom_color = use_custom_color

    def restore_auto_colors(self):
        self.use_custom_color = False
        self.colors = self.original_colors.copy()
        self.vao.activate()
        buffer_idx = self.vao.vbo[1]
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_idx)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.colors, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def set_solid_color(self, color):
        """Set solid color for the mathematical surface"""
        self.use_flat_color = True
        self.flat_color = np.array(color[:3], dtype=np.float32)
        self.colors = np.array([color[:3]] * len(self.vertices), dtype=np.float32)
        self.vao.activate()
        buffer_idx = self.vao.vbo[1]
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_idx)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.colors, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'vao'):
            self.vao.delete()
        if hasattr(self, 'shader'):
            self.shader.delete()
        if hasattr(self, 'texture_id') and self.texture_id is not None:
            GL.glDeleteTextures(1, [self.texture_id])
