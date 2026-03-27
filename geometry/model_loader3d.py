import sys
import os
import numpy as np
import ctypes
from PIL import Image

# Add parent directory to path to import libs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.shader import *
from libs import transform as T
from libs.buffer import *
from libs.lighting import LightingManager
import OpenGL.GL as GL

# Import base shape
from geometry.base_shape import BaseShape


class ModelLoader(BaseShape):
    def __init__(self, vert_shader, frag_shader, filename=None):
        super().__init__()
        self.vert_shader = vert_shader
        self.frag_shader = frag_shader
        self.filename = filename
        
        # --- CÁC BIẾN TRẠNG THÁI CHO SIÊU SHADER ---
        self.use_flat_color = False
        self.flat_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.use_texture = False
        self.texture_id = None
        self.render_mode = 2  # Mặc định là Phong Shading
        
        if filename:
            self.load_model(filename)
        else:
            self._create_default_cube()
            
    def _create_default_cube(self):
        self.vertices = np.array([
            [-1, -1, +1], [+1, -1, +1], [+1, -1, -1], [-1, -1, -1],
            [-1, +1, +1], [+1, +1, +1], [+1, +1, -1], [-1, +1, -1]
        ], dtype=np.float32)
        
        self.indices = np.array([
            0, 4, 1, 5, 2, 6, 3, 7, 0, 4, 4, 0, 0, 3, 1, 2, 2, 4, 4, 7, 5, 6
        ], dtype=np.int32)
        
        self.normals = self.vertices.copy()
        self.normals = self.normals / np.linalg.norm(self.normals, axis=1, keepdims=True)
        
        self.colors = np.array([
            [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]
        ], dtype=np.float32)
        
        self._generate_texcoords()
    
    def load_model(self, filename):
        if not os.path.exists(filename):
            self._create_default_cube()
            return
            
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext == '.obj':
            self._load_obj(filename)
        elif file_ext == '.ply':
            self._load_ply(filename)
        else:
            self._create_default_cube()
            
        # SAU KHI LOAD XONG, TỰ ĐỘNG SINH UV ĐỂ DÁN ẢNH
        self._generate_texcoords()
    
    def _load_obj(self, filename):
        vertices = []
        normals = []
        faces = []
        colors = []
        
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                    
                parts = line.split()
                if not parts: continue
                    
                if parts[0] == 'v':
                    vertices.append([float(x) for x in parts[1:4]])
                    if len(parts) >= 7:
                        r, g, b = float(parts[4]), float(parts[5]), float(parts[6])
                        if r > 1.0 or g > 1.0 or b > 1.0:
                            colors.append([r/255.0, g/255.0, b/255.0])
                        else:
                            colors.append([r, g, b])
                elif parts[0] == 'vn':
                    normals.append([float(x) for x in parts[1:4]])
                elif parts[0] == 'f':
                    face_indices = []
                    for part in parts[1:]:
                        indices = part.split('/')
                        face_indices.append(int(indices[0]) - 1)
                    faces.append(face_indices)
        
        if not vertices:
            self._create_default_cube()
            return
            
        self.vertices = np.array(vertices, dtype=np.float32)
        
        indices = []
        for face in faces:
            if len(face) == 3:
                indices.extend(face)
            elif len(face) == 4:
                indices.extend([face[0], face[1], face[2]])
                indices.extend([face[2], face[1], face[3]])
        
        self.indices = np.array(indices, dtype=np.int32)
        
        self._generate_normals()
        
        if len(colors) == len(vertices):
            self.colors = np.array(colors, dtype=np.float32)
        else:
            self._generate_colors()
        
        self._normalize_model()
    
    def _load_ply(self, filename):
        import struct
        vertices = []
        faces = []
        colors = []
        normals = []
        
        with open(filename, 'rb') as f:
            header_lines = []
            while True:
                line = f.readline().decode('utf-8', errors='ignore').strip()
                header_lines.append(line)
                if line == 'end_header': break
                    
            format_type = 'ascii'
            vertex_count = 0
            face_count = 0
            vertex_props = [] 
            
            current_element = None
            for line in header_lines:
                if line.startswith('format'):
                    format_type = line.split()[1] 
                elif line.startswith('element'):
                    parts = line.split()
                    current_element = parts[1]
                    if current_element == 'vertex':
                        vertex_count = int(parts[2])
                    elif current_element == 'face':
                        face_count = int(parts[2])
                elif line.startswith('property') and current_element == 'vertex':
                    parts = line.split()
                    vertex_props.append((parts[1], parts[2]))
                    
            if vertex_count == 0:
                self._create_default_cube()
                return

            if format_type == 'ascii':
                lines = f.read().decode('utf-8', errors='ignore').splitlines()
                lines = [l.strip() for l in lines if l.strip()]
                
                for i in range(vertex_count):
                    parts = lines[i].split()
                    if len(parts) >= 3:
                        vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        if len(parts) >= 6:
                            try:
                                r, g, b = float(parts[-3]), float(parts[-2]), float(parts[-1])
                                colors.append([r/255.0 if r>1 else r, g/255.0 if g>1 else g, b/255.0 if b>1 else b])
                            except: pass
                
                for i in range(vertex_count, vertex_count + face_count):
                    if i >= len(lines): break
                    parts = lines[i].split()
                    if len(parts) >= 4:
                        n_verts = int(parts[0])
                        face_verts = [int(x) for x in parts[1:1+n_verts]]
                        if n_verts == 3: faces.append(face_verts)
                        elif n_verts == 4:
                            faces.append([face_verts[0], face_verts[1], face_verts[2]])
                            faces.append([face_verts[2], face_verts[1], face_verts[3]])
                            
            else:
                endian = '<' if format_type == 'binary_little_endian' else '>'
                v_fmt = endian
                for p_type, _ in vertex_props:
                    if p_type in ['float', 'float32']: v_fmt += 'f'
                    elif p_type in ['double', 'float64']: v_fmt += 'd'
                    elif p_type in ['uchar', 'uint8']: v_fmt += 'B'
                    elif p_type in ['int', 'int32']: v_fmt += 'i'
                    else: v_fmt += 'f'
                    
                v_size = struct.calcsize(v_fmt)
                
                for _ in range(vertex_count):
                    data = f.read(v_size)
                    if not data: break
                    unpacked = struct.unpack(v_fmt, data)
                    v = [0.0, 0.0, 0.0]
                    c = [-1.0, -1.0, -1.0]
                    
                    for idx, (p_type, p_name) in enumerate(vertex_props):
                        val = unpacked[idx]
                        if p_name == 'x': v[0] = float(val)
                        elif p_name == 'y': v[1] = float(val)
                        elif p_name == 'z': v[2] = float(val)
                        elif p_name in ['r', 'red']: c[0] = val/255.0 if p_type in ['uchar', 'uint8'] else float(val)
                        elif p_name in ['g', 'green']: c[1] = val/255.0 if p_type in ['uchar', 'uint8'] else float(val)
                        elif p_name in ['b', 'blue']: c[2] = val/255.0 if p_type in ['uchar', 'uint8'] else float(val)
                        
                    vertices.append(v)
                    if c[0] >= 0: colors.append(c)
                
                for _ in range(face_count):
                    count_data = f.read(1)
                    if not count_data: break
                    count = struct.unpack(endian + 'B', count_data)[0]
                    idx_data = f.read(count * 4) 
                    if len(idx_data) < count * 4: break
                    indices = struct.unpack(endian + str(count) + 'i', idx_data)
                    if count == 3:
                        faces.append(list(indices))
                    elif count == 4:
                        faces.append([indices[0], indices[1], indices[2]])
                        faces.append([indices[2], indices[1], indices[3]])

        if face_count == 0 or len(faces) == 0:
            self._create_default_cube()
            return
            
        self.vertices = np.array(vertices, dtype=np.float32)
        
        indices = []
        for face in faces:
            indices.extend(face)
        self.indices = np.array(indices, dtype=np.int32)
        
        if len(colors) == len(vertices):
            self.colors = np.array(colors, dtype=np.float32)
        else:
            self._generate_colors()
            
        self._generate_normals()
        self._normalize_model()

    def _generate_normals(self):
        normals = np.zeros_like(self.vertices)
        for i in range(0, len(self.indices), 3):
            if i + 2 >= len(self.indices): break
            i0, i1, i2 = self.indices[i], self.indices[i+1], self.indices[i+2]
            if i0 >= len(self.vertices) or i1 >= len(self.vertices) or i2 >= len(self.vertices): continue
            
            v0, v1, v2 = self.vertices[i0], self.vertices[i1], self.vertices[i2]
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            
            normals[i0] += face_normal
            normals[i1] += face_normal
            normals[i2] += face_normal
        
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.normals = normals / norms
    
    def _generate_colors(self):
        """Khởi tạo toàn bộ các đỉnh thành màu Trắng thay vì Cầu vồng"""
        self.colors = np.array([[1.0, 1.0, 1.0]] * len(self.vertices), dtype=np.float32)
        
    def _generate_texcoords(self):
        """Ma thuật Auto UV Mapping (Spherical Projection)"""
        norms = np.linalg.norm(self.vertices, axis=1, keepdims=True)
        norms[norms == 0] = 1.0 # Tránh chia cho 0
        norm_v = self.vertices / norms
        
        u = 0.5 + np.arctan2(norm_v[:, 2], norm_v[:, 0]) / (2 * np.pi)
        v = 0.5 - np.arcsin(norm_v[:, 1]) / np.pi
        self.texcoords = np.column_stack((u, v)).astype(np.float32)
    
    def _normalize_model(self):
        if len(self.vertices) == 0: return
        center = self.vertices.mean(axis=0)
        self.vertices -= center
        max_dist = np.max(np.abs(self.vertices))
        if max_dist > 0:
            scale = 2.0 / max_dist
            self.vertices *= scale

    def setup(self):
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
            img = Image.open(filepath).convert("RGBA")
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
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
            print(f"Đã load texture thành công cho Model: {filepath}")
        except Exception as e:
            print(f"Lỗi load texture: {e}")
            self.use_texture = False

    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)
        
        object_transform = self.get_transform_matrix()
        final_model = object_transform @ (model if model is not None else np.identity(4, dtype=np.float32))
        modelview = view @ final_model
        
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        
        # --- CÁC CÔNG TẮC CHO SIÊU SHADER ---
        loc_flat = GL.glGetUniformLocation(self.shader.render_idx, "u_use_flat_color")
        if loc_flat != -1: GL.glUniform1i(loc_flat, 1 if self.use_flat_color else 0)
        
        loc_flat_col = GL.glGetUniformLocation(self.shader.render_idx, "u_flat_color")
        if loc_flat_col != -1: 
            GL.glUniform3f(loc_flat_col, self.flat_color[0], self.flat_color[1], self.flat_color[2])
            
        loc_tex = GL.glGetUniformLocation(self.shader.render_idx, "u_use_texture")
        if loc_tex != -1: GL.glUniform1i(loc_tex, 1 if self.use_texture else 0)
        
        loc_mode = GL.glGetUniformLocation(self.shader.render_idx, "u_render_mode")
        if loc_mode != -1: GL.glUniform1i(loc_mode, self.render_mode)
        
        # --- HỆ THỐNG ĐA NGUỒN SÁNG (MULTI-LIGHTING) ---
        lights = getattr(self, 'scene_lights', [])
        loc_num_lights = GL.glGetUniformLocation(self.shader.render_idx, "u_num_lights")
        if loc_num_lights != -1: GL.glUniform1i(loc_num_lights, len(lights))
        
        for i, l in enumerate(lights[:4]): # Hỗ trợ tối đa 4 nguồn sáng cùng lúc
            GL.glUniform3f(GL.glGetUniformLocation(self.shader.render_idx, f"u_light_pos[{i}]"), *l.position)
            GL.glUniform3f(GL.glGetUniformLocation(self.shader.render_idx, f"u_light_color[{i}]"), *l.light_color)
            GL.glUniform1f(GL.glGetUniformLocation(self.shader.render_idx, f"u_light_intensity[{i}]"), l.light_intensity)
            GL.glUniform1i(GL.glGetUniformLocation(self.shader.render_idx, f"u_light_active[{i}]"), 1 if l.visible else 0)
        
        if self.use_texture and self.texture_id is not None:
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
            loc_sampler = GL.glGetUniformLocation(self.shader.render_idx, "u_texture")
            if loc_sampler != -1: GL.glUniform1i(loc_sampler, 0)
        
        self.vao.activate()
        # Tự động nhận diện vẽ theo Indices hoặc Arrays
        if hasattr(self, 'indices') and self.indices is not None:
            GL.glDrawElements(GL.GL_TRIANGLES, len(self.indices), GL.GL_UNSIGNED_INT, None)
        else:
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertices.shape[0])
        self.vao.deactivate()
        
        if self.use_texture:
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    
    def set_color(self, color):
        self.colors = np.array([color] * len(self.vertices), dtype=np.float32)
        self.vao.activate()
        buffer_idx = self.vao.vbo[1]
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_idx)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.colors, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        self.flat_color = np.array(color[:3], dtype=np.float32)

    def set_solid_color(self, color):
        self.use_flat_color = True
        self.flat_color = np.array(color[:3], dtype=np.float32)

    def cleanup(self):
        if hasattr(self, 'vao'): self.vao.delete()
        if hasattr(self, 'shader'): self.shader.delete()
        if hasattr(self, 'texture_id') and self.texture_id is not None:
            GL.glDeleteTextures(1, [self.texture_id])