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
        
        # Đây là các state để object model có thể dùng chung standard shader:
        # màu, flat shading, texture, nhiều material texture...
        self.use_flat_color = False
        self.flat_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.use_texture = False
        self.texture_id = None
        self.material_texture_path = None
        self.material_texture_paths = {}
        self.material_texture_ids = {}
        self.material_groups = []
        self.manual_texture_override = False
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
        # Hàm tổng quát: nhìn đuôi file để quyết định nên gọi parser OBJ hay PLY.
        if not os.path.exists(filename):
            self._create_default_cube()
            return
        
        self.material_texture_path = None
        self.material_texture_paths = {}
        self.material_texture_ids = {}
        self.material_groups = []
        self.manual_texture_override = False
        self.use_texture = False
            
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext == '.obj':
            self._load_obj(filename)
        elif file_ext == '.ply':
            self._load_ply(filename)
        else:
            self._create_default_cube()
            
        if not hasattr(self, 'texcoords') or len(self.texcoords) != len(self.vertices):
            self._generate_texcoords()

    def _parse_obj_index(self, raw_value, count):
        if raw_value == '':
            return None
        idx = int(raw_value)
        return idx - 1 if idx > 0 else count + idx

    def _normalize_material_name(self, name):
        if not name:
            return ''

        normalized = ''.join(ch.lower() for ch in name if ch.isalnum())
        for suffix in ('material', 'mat', 'shader', 'sg'):
            if normalized.endswith(suffix) and len(normalized) > len(suffix):
                normalized = normalized[:-len(suffix)]
        return normalized

    def _guess_material_texture_path(self, material_name, search_dir):
        if not search_dir or not os.path.isdir(search_dir):
            return None

        supported_exts = {'.png', '.jpg', '.jpeg', '.tga', '.bmp', '.tif', '.tiff'}
        preferred_tags = ('basecolor', 'base_color', 'albedo', 'diffuse', 'diff', 'dif', 'color')
        reject_tags = ('normal', 'roughness', 'metal', 'metallic', 'ao', 'ambient', 'spec', 'gloss', 'opacity', 'invert')
        material_key = self._normalize_material_name(material_name)
        best_score = 0
        best_path = None

        try:
            entries = os.listdir(search_dir)
        except OSError:
            return None

        for entry in entries:
            candidate_path = os.path.join(search_dir, entry)
            stem, ext = os.path.splitext(entry)
            if ext.lower() not in supported_exts or not os.path.isfile(candidate_path):
                continue

            stem_key = self._normalize_material_name(stem)
            score = 0

            if any(tag in stem_key for tag in reject_tags):
                score -= 10

            if any(tag in stem_key for tag in preferred_tags):
                score += 8

            if material_key and stem_key:
                if material_key in stem_key or stem_key in material_key:
                    score += 10
                else:
                    material_prefix = material_key[:max(3, len(material_key) // 2)]
                    if material_prefix and material_prefix in stem_key:
                        score += 4

            if score > best_score:
                best_score = score
                best_path = candidate_path

        return os.path.normpath(best_path) if best_score > 0 else None

    def _guess_folder_texture_path(self, search_dir):
        """Pick a diffuse texture from the OBJ folder when no MTL is provided."""
        if not search_dir or not os.path.isdir(search_dir):
            return None

        supported_exts = {'.png', '.jpg', '.jpeg', '.tga', '.bmp', '.tif', '.tiff'}
        preferred_tags = ('basecolor', 'base_color', 'albedo', 'diffuse', 'diff', 'dif', 'color')
        reject_tags = ('normal', 'roughness', 'metal', 'metallic', 'ao', 'ambient', 'spec', 'gloss', 'opacity', 'invert')
        candidates = []

        try:
            entries = os.listdir(search_dir)
        except OSError:
            return None

        for entry in entries:
            candidate_path = os.path.join(search_dir, entry)
            stem, ext = os.path.splitext(entry)
            if ext.lower() not in supported_exts or not os.path.isfile(candidate_path):
                continue

            stem_key = self._normalize_material_name(stem)
            if any(tag in stem_key for tag in reject_tags):
                continue

            score = 1
            if any(tag in stem_key for tag in preferred_tags):
                score += 10
            candidates.append((score, candidate_path))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (-item[0], os.path.basename(item[1]).lower()))
        return os.path.normpath(candidates[0][1])

    def _load_mtl_file(self, mtl_path):
        # Đọc file .mtl để lấy thông tin material, chủ yếu là:
        # - Kd: màu diffuse (màu cơ bản của vật liệu)
        # - map_Kd: texture diffuse (ảnh bề mặt của vật liệu)
        materials = {}          # Dictionary lưu tất cả materials
        current_material = None # Material hiện tại đang parse

        if not os.path.exists(mtl_path):
            return materials    # Return empty dict nếu file không tồn tại

        with open(mtl_path, 'r', encoding='utf-8', errors='ignore') as mtl_file:
            for line in mtl_file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue    # Bỏ qua dòng trống và comment

                parts = line.split()
                key = parts[0]    # Lệnh đầu tiên (newmtl, Kd, map_Kd)

                # Tạo material mới
                if key == 'newmtl' and len(parts) > 1:
                    current_material = " ".join(parts[1:])  # Tên material
                    materials[current_material] = {}       # Khởi tạo dict cho material
                
                # Lấy màu diffuse (RGB)
                elif current_material is not None and key == 'Kd' and len(parts) >= 4:
                    materials[current_material]['kd'] = [
                        float(parts[1]),  # Red (0-1)
                        float(parts[2]),  # Green (0-1)
                        float(parts[3])   # Blue (0-1)
                    ]
                
                # Lấy đường dẫn texture
                elif current_material is not None and key == 'map_Kd' and len(parts) > 1:
                    texture_rel = " ".join(parts[1:])  # Tên file texture
                    # Chuyển đường dẫn tương đối thành tuyệt đối
                    texture_path = texture_rel if os.path.isabs(texture_rel) else os.path.join(os.path.dirname(mtl_path), texture_rel)
                    materials[current_material]['map_kd'] = os.path.normpath(texture_path)

        search_dir = os.path.dirname(mtl_path)
        for material_name, material_info in materials.items():
            if material_info.get('map_kd'):
                continue
            guessed_texture = self._guess_material_texture_path(material_name, search_dir)
            if guessed_texture:
                material_info['map_kd'] = guessed_texture

        return materials
    
    def _load_obj(self, filename):
        # Parser OBJ này đọc:
        # - v  : vị trí đỉnh
        # - vt : UV
        # - vn : normal
        # - mtllib / usemtl : vật liệu và texture tương ứng
        positions = []
        texcoords = []
        normals = []
        colors = []
        out_vertices = []
        out_texcoords = []
        out_normals = []
        out_colors = []
        out_indices = []
        vertex_map = {}
        materials = {}
        used_materials = []
        current_material = None
        obj_dir = os.path.dirname(filename)
        
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                    
                parts = line.split()
                if not parts: continue
                    
                if parts[0] == 'v':
                    positions.append([float(x) for x in parts[1:4]])
                    if len(parts) >= 7:
                        r, g, b = float(parts[4]), float(parts[5]), float(parts[6])
                        if r > 1.0 or g > 1.0 or b > 1.0:
                            colors.append([r/255.0, g/255.0, b/255.0])
                        else:
                            colors.append([r, g, b])
                    else:
                        colors.append(None)
                elif parts[0] == 'vt':
                    u = float(parts[1]) if len(parts) > 1 else 0.0
                    v = float(parts[2]) if len(parts) > 2 else 0.0
                    texcoords.append([u, v])
                elif parts[0] == 'vn':
                    normals.append([float(x) for x in parts[1:4]])
                elif parts[0] == 'mtllib': # vd: Sử dụng file cube.mtl
                    material_name = " ".join(parts[1:])
                    mtl_path = material_name if os.path.isabs(material_name) else os.path.join(obj_dir, material_name)
                    materials.update(self._load_mtl_file(os.path.normpath(mtl_path)))
                elif parts[0] == 'usemtl': # vd: Sử dụng Material (file.pnd)
                    current_material = " ".join(parts[1:]) if len(parts) > 1 else None
                    if current_material and current_material not in used_materials:
                        used_materials.append(current_material)
                elif parts[0] == 'f':
                    face_vertices = []
                    for part in parts[1:]:
                        indices = part.split('/')
                        v_idx = self._parse_obj_index(indices[0], len(positions))
                        vt_idx = self._parse_obj_index(indices[1], len(texcoords)) if len(indices) > 1 else None
                        vn_idx = self._parse_obj_index(indices[2], len(normals)) if len(indices) > 2 else None
                        key = (v_idx, vt_idx, vn_idx, current_material)

                        if key not in vertex_map:
                            out_vertices.append(positions[v_idx])
                            out_texcoords.append(texcoords[vt_idx] if vt_idx is not None and 0 <= vt_idx < len(texcoords) else [0.0, 0.0])
                            out_normals.append(normals[vn_idx] if vn_idx is not None and 0 <= vn_idx < len(normals) else [0.0, 0.0, 0.0])

                            material_info = materials.get(current_material, {})
                            vertex_color = colors[v_idx] if 0 <= v_idx < len(colors) else None
                            out_colors.append(vertex_color or material_info.get('kd', [1.0, 1.0, 1.0]))
                            vertex_map[key] = len(out_vertices) - 1

                        face_vertices.append(vertex_map[key])

                    # Mỗi lần material đổi, ta mở một "group" mới.
                    # Lúc draw sẽ bind đúng texture cho từng group này.
                    if not self.material_groups or self.material_groups[-1]['material'] != current_material:
                        self.material_groups.append({
                            'material': current_material,
                            'start': len(out_indices),
                            'count': 0,
                        })

                    for i in range(1, len(face_vertices) - 1):
                        tri_indices = [face_vertices[0], face_vertices[i], face_vertices[i + 1]]
                        out_indices.extend(tri_indices)
                        self.material_groups[-1]['count'] += len(tri_indices)
        
        if not positions or not out_vertices:
            self._create_default_cube()
            return
            
        self.vertices = np.array(out_vertices, dtype=np.float32)
        self.indices = np.array(out_indices, dtype=np.int32)
        self.texcoords = np.array(out_texcoords, dtype=np.float32)
        self.normals = np.array(out_normals, dtype=np.float32)
        self.vertices, self.normals = self._apply_source_orientation(self.vertices, self.normals, filename)

        if len(out_colors) == len(out_vertices):
            self.colors = np.array(out_colors, dtype=np.float32)
        else:
            self._generate_colors()

        if len(normals) == 0 or not np.any(self.normals):
            self._generate_normals()

        for material_name in used_materials:
            texture_path = materials.get(material_name, {}).get('map_kd')
            if texture_path and os.path.exists(texture_path):
                self.material_texture_paths[material_name] = texture_path
                if self.material_texture_path is None:
                    self.material_texture_path = texture_path

        if not self.material_texture_paths:
            fallback_texture = self._guess_folder_texture_path(obj_dir)
            if fallback_texture and os.path.exists(fallback_texture):
                fallback_material = current_material
                if self.material_groups:
                    fallback_material = self.material_groups[0]['material']
                self.material_texture_paths[fallback_material] = fallback_texture
                self.material_texture_path = fallback_texture

        self.material_groups = [group for group in self.material_groups if group['count'] > 0]
        
        self._normalize_model()
    
    def _load_ply(self, filename):
        """Load PLY file format (Stanford Polygon Library)"""
        import struct
        vertices = []
        faces = []
        colors = []
        normals = []
        
        # Đọc header để biết thông tin file
        with open(filename, 'rb') as f:
            header_lines = []
            while True:
                line = f.readline().decode('utf-8', errors='ignore').strip()
                header_lines.append(line)
                if line == 'end_header': break  # Kết thúc header
                    
            # Phân tích header
            format_type = 'ascii'  # Mặc định là ASCII
            vertex_count = 0
            face_count = 0
            vertex_props = []  # Các thuộc tính của vertex
            
            current_element = None  # Element hiện tại đang parse (vertex/face)
            for line in header_lines:
                if line.startswith('format'):
                    # Lấy định dạng file: ascii/binary_little_endian/binary_big_endian
                    format_type = line.split()[1]  # Ví dụ: "ascii" hoặc "binary_little_endian"
                elif line.startswith('element'):
                    # Định nghĩa element type và số lượng
                    parts = line.split()
                    current_element = parts[1]  # "vertex" hoặc "face"
                    if current_element == 'vertex':
                        vertex_count = int(parts[2])  # Số lượng vertices: element vertex 8
                    elif current_element == 'face':
                        face_count = int(parts[2])    # Số lượng faces: element face 12
                elif line.startswith('property') and current_element == 'vertex':
                    # Định nghĩa thuộc tính của vertex (chỉ quan tâm vertex properties)
                    parts = line.split()
                    # Lưu cặp (type, name): property float x, property float y, property uchar red
                    vertex_props.append((parts[1], parts[2]))  # Ví dụ: ("float", "x")
                    
            if vertex_count == 0:
                self._create_default_cube()
                return

            # Đọc file ASCII
            if format_type == 'ascii':
                lines = f.read().decode('utf-8', errors='ignore').splitlines()
                lines = [l.strip() for l in lines if l.strip()]
                
                # Đọc vertices
                for i in range(vertex_count):
                    parts = lines[i].split()
                    if len(parts) >= 3:
                        # Lấy tọa độ x,y,z
                        vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        # Lấy màu nếu có (thường ở cuối)
                        if len(parts) >= 6:
                            try:
                                r, g, b = float(parts[-3]), float(parts[-2]), float(parts[-1])
                                # Chuyển đổi về range [0,1]
                                colors.append([r/255.0 if r>1 else r, g/255.0 if g>1 else g, b/255.0 if b>1 else b])
                            except: pass
                
                # Đọc faces
                for i in range(vertex_count, vertex_count + face_count):
                    if i >= len(lines): break
                    parts = lines[i].split()
                    if len(parts) >= 4:
                        n_verts = int(parts[0])  # Số vertices trong face
                        face_verts = [int(x) for x in parts[1:1+n_verts]]
                        # Tách quad thành 2 triangles
                        if n_verts == 3: faces.append(face_verts)
                        elif n_verts == 4:
                            faces.append([face_verts[0], face_verts[1], face_verts[2]])
                            faces.append([face_verts[2], face_verts[1], face_verts[3]])
                            
            # Đọc file BINARY (không phải ASCII)
            else:
                # Xác định endian: little-endian (<) hoặc big-endian (>)
                endian = '<' if format_type == 'binary_little_endian' else '>'
                
                # Xây dựng format string cho struct.unpack dựa trên các properties
                # Ví dụ: vertex_props = [("float", "x"), ("float", "y"), ("float", "z"), ("uchar", "r"), ("uchar", "g"), ("uchar", "b")]
                # → v_fmt = "<ffBBB" (little-endian: 3 floats + 3 unsigned chars)
                v_fmt = endian
                for p_type, _ in vertex_props:
                    if p_type in ['float', 'float32']: v_fmt += 'f'    # float = 4 bytes
                    elif p_type in ['double', 'float64']: v_fmt += 'd'  # double = 8 bytes
                    elif p_type in ['uchar', 'uint8']: v_fmt += 'B'    # unsigned char = 1 byte
                    elif p_type in ['int', 'int32']: v_fmt += 'i'      # int = 4 bytes
                    else: v_fmt += 'f'  # Mặc định là float
                    
                v_size = struct.calcsize(v_fmt)  # Tổng kích thước mỗi vertex (bytes)
                # Ví dụ: struct.calcsize("<ffBBB") = 4+4+4+1+1+1 = 15 bytes
                
                # Đọc vertices từ binary data
                for _ in range(vertex_count):
                    data = f.read(v_size)  # Đọc chính xác v_size bytes cho 1 vertex
                    if not data: break
                    unpacked = struct.unpack(v_fmt, data)  # Unpack theo format string
                    v = [0.0, 0.0, 0.0]  # Vertex position [x,y,z]
                    c = [-1.0, -1.0, -1.0]  # Color [r,g,b], -1 = không có màu
                    
                    # Map dữ liệu đã unpack vào các thuộc tính tương ứng
                    for idx, (p_type, p_name) in enumerate(vertex_props):
                        val = unpacked[idx]  # Giá trị tại vị trí idx
                        if p_name == 'x': v[0] = float(val)      # Gán tọa độ x
                        elif p_name == 'y': v[1] = float(val)      # Gán tọa độ y
                        elif p_name == 'z': v[2] = float(val)      # Gán tọa độ z
                        elif p_name in ['r', 'red']:              # Gán màu đỏ
                            # Nếu là unsigned char (0-255) → chia cho 255, nếu là float → giữ nguyên
                            c[0] = val/255.0 if p_type in ['uchar', 'uint8'] else float(val)
                        elif p_name in ['g', 'green']:            # Gán màu xanh lá
                            c[1] = val/255.0 if p_type in ['uchar', 'uint8'] else float(val)
                        elif p_name in ['b', 'blue']:             # Gán màu xanh dương
                            c[2] = val/255.0 if p_type in ['uchar', 'uint8'] else float(val)
                        
                    vertices.append(v)
                    if c[0] >= 0: colors.append(c)  # Chỉ add color nếu có giá trị hợp lệ
                
                # Đọc faces từ binary data
                for _ in range(face_count):
                    count_data = f.read(1)  # Đọc 1 byte: số vertices trong face
                    if not count_data: break
                    count = struct.unpack(endian + 'B', count_data)[0]  # Unpack unsigned char
                    idx_data = f.read(count * 4)  # Đọc count * 4 bytes (mỗi index = 4 bytes)
                    if len(idx_data) < count * 4: break
                    indices = struct.unpack(endian + str(count) + 'i', idx_data)  # Unpack count integers
                    if count == 3:
                        faces.append(list(indices))  # Triangle
                    elif count == 4:
                        # Tách quad thành 2 triangles
                        faces.append([indices[0], indices[1], indices[2]])
                        faces.append([indices[2], indices[1], indices[3]])

        # Fallback nếu không có faces
        if face_count == 0 or len(faces) == 0:
            self._create_default_cube()
            return
            
        # Chuyển đổi sang numpy arrays
        self.vertices = np.array(vertices, dtype=np.float32)
        self.vertices, _ = self._apply_source_orientation(self.vertices, None, filename)
        
        # Flatten faces thành indices array
        indices = []
        for face in faces:
            indices.extend(face)
        self.indices = np.array(indices, dtype=np.int32)
        
        # Sử dụng colors từ file hoặc generate
        if len(colors) == len(vertices):
            self.colors = np.array(colors, dtype=np.float32)
        else:
            self._generate_colors()
            
        # Generate normals và normalize model
        self._generate_normals()
        self._normalize_model()

    def _apply_source_orientation(self, vertices, normals, filename):
        """Fix known asset coordinate systems before the model is normalized."""
        if not filename:
            return vertices, normals

        lowered = str(filename).replace("\\", "/").lower()
        if not any(token in lowered for token in ("traffic_light", "trafficlights", "stoplight", "signal")):
            return vertices, normals

        vertices = np.nan_to_num(vertices, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if normals is not None:
            normals = np.nan_to_num(normals, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        # The bundled stoplight is authored Z-up; the scene and controls are Y-up.
        rotated_vertices = np.column_stack((vertices[:, 0], vertices[:, 2], -vertices[:, 1])).astype(np.float32)
        rotated_normals = (
            np.column_stack((normals[:, 0], normals[:, 2], -normals[:, 1])).astype(np.float32)
            if normals is not None
            else None
        )
        return rotated_vertices, rotated_normals

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
        # Step 1: Tính độ dài của mỗi vertex để normalize về unit sphere
        norms = np.linalg.norm(self.vertices, axis=1, keepdims=True)
        # Step 2: Tránh chia cho 0 nếu vertex tại origin
        norms[norms == 0] = 1.0 
        # Step 3: Normalize vertices về sphere radius = 1
        norm_v = self.vertices / norms
        
        # Step 4: Tính U coordinate từ longitude (kinh độ)
        # arctan2(z, x) = angle quanh trục Y [-π, π]
        # Convert sang [0, 1] range cho texture
        u = 0.5 + np.arctan2(norm_v[:, 2], norm_v[:, 0]) / (2 * np.pi) #lay cot thu 2 
        
        # Step 5: Tính V coordinate từ latitude (vĩ độ)  
        # arcsin(y) = angle từ equator [-π/2, π/2]
        # Convert sang [0, 1] range (flip cho correct orientation)
        v = 0.5 - np.arcsin(norm_v[:, 1]) / np.pi
        
        # Step 6: Ghép thành UV coordinates cho OpenGL
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
        self._load_material_textures()
        
        return self

    def _load_texture_file(self, filepath):
        try:
            img = Image.open(filepath).convert("RGBA")
            # #img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img_data = img.tobytes("raw", "RGBA", 0, -1)

            texture_id = GL.glGenTextures(1)

            GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, img.width, img.height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_data)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            return texture_id
        except Exception as e:
            print(f"Lỗi load texture: {e}")
            return None

    def _load_material_textures(self):
        # Sau khi biết material nào dùng texture nào, hàm này sẽ load hết chúng lên GPU.
        for texture_id in self.material_texture_ids.values():
            GL.glDeleteTextures(1, [texture_id])
        self.material_texture_ids = {}

        for material_name, texture_path in self.material_texture_paths.items():
            texture_id = self._load_texture_file(texture_path)
            if texture_id is not None:
                self.material_texture_ids[material_name] = texture_id

        if self.material_texture_ids:
            self.use_texture = True

    def set_texture(self, filepath):
        if not filepath:
            self.manual_texture_override = False
            if self.texture_id is not None:
                GL.glDeleteTextures(1, [self.texture_id])
                self.texture_id = None
            self.use_texture = bool(self.material_texture_ids)
            return

        texture_id = self._load_texture_file(filepath)
        if texture_id is None:
            self.use_texture = bool(self.material_texture_ids)
            return

        if self.texture_id is not None:
            GL.glDeleteTextures(1, [self.texture_id])

        self.texture_id = texture_id
        self.manual_texture_override = True
        self.use_texture = True
        print(f"Đã load texture thành công cho Model: {filepath}")

    def draw(self, projection, view, model):
        # Khi vẽ model OBJ, nếu model có nhiều material thì không thể draw một phát rồi xong.
        # Ta phải duyệt theo từng material group và bind đúng texture trước mỗi lệnh draw.
        GL.glUseProgram(self.shader.render_idx)
        
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
        
        loc_sampler = GL.glGetUniformLocation(self.shader.render_idx, "u_texture")
        if loc_sampler != -1:
            GL.glUniform1i(loc_sampler, 0)
        
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
        # Tự động nhận diện vẽ theo Indices hoặc Arrays
        if hasattr(self, 'indices') and self.indices is not None:
            if self.use_texture and not self.manual_texture_override and self.material_groups and self.material_texture_ids:
                for group in self.material_groups:
                    texture_id = self.material_texture_ids.get(group['material'])
                    if loc_tex != -1:
                        GL.glUniform1i(loc_tex, 1 if texture_id is not None else 0)
                    if texture_id is not None:
                        GL.glActiveTexture(GL.GL_TEXTURE0)
                        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
                    else:
                        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
                    GL.glDrawElements(
                        GL.GL_TRIANGLES,
                        group['count'],
                        GL.GL_UNSIGNED_INT,
                        ctypes.c_void_p(group['start'] * np.dtype(np.uint32).itemsize),
                    )
            else:
                active_texture_id = self.texture_id
                if active_texture_id is None and self.material_texture_ids:
                    active_texture_id = next(iter(self.material_texture_ids.values()))

                if loc_tex != -1:
                    GL.glUniform1i(loc_tex, 1 if self.use_texture and active_texture_id is not None else 0)
                if self.use_texture and active_texture_id is not None:
                    GL.glActiveTexture(GL.GL_TEXTURE0)
                    GL.glBindTexture(GL.GL_TEXTURE_2D, active_texture_id)
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
        for texture_id in self.material_texture_ids.values():
            GL.glDeleteTextures(1, [texture_id])
