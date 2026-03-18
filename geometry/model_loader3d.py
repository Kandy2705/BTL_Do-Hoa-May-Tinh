import sys
import os
import numpy as np
import ctypes

# Add parent directory to path to import libs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.shader import *
from libs import transform as T
from libs.buffer import *
from libs.lighting import LightingManager
import OpenGL.GL as GL


class ModelLoader(object):
    def __init__(self, vert_shader, frag_shader, filename=None):
        self.vert_shader = vert_shader
        self.frag_shader = frag_shader
        self.filename = filename
        
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
    
    def load_model(self, filename):
        """Load model from .obj or .ply file"""
        if not os.path.exists(filename):
            print(f"Warning: File {filename} not found. Using default cube.")
            self._create_default_cube()
            return
            
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext == '.obj':
            self._load_obj(filename)
        elif file_ext == '.ply':
            self._load_ply(filename)
        else:
            print(f"Warning: Unsupported file format {file_ext}. Using default cube.")
            self._create_default_cube()
    
    def _load_obj(self, filename):
        """Load OBJ file format"""
        vertices = []
        normals = []
        faces = []
        colors = []
        
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                parts = line.split()
                if not parts:
                    continue
                    
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
                        face_indices.append(int(indices[0]) - 1)  # OBJ is 1-indexed
                    faces.append(face_indices)
        
        if not vertices:
            print("Warning: No vertices found in OBJ file. Using default cube.")
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
        
        if normals:
            self.normals = np.array(normals, dtype=np.float32)
        else:
            self._generate_normals()
        
        if len(colors) == len(vertices):
            self.colors = np.array(colors, dtype=np.float32)
        else:
            self._generate_colors()
        
        self._normalize_model()
    
    def _load_ply(self, filename):
        vertices = []
        normals = []
        faces = []
        
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        header_end = -1
        vertex_count = 0
        face_count = 0
        has_normals = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('element face'):
                face_count = int(line.split()[-1])
            elif 'nx' in line and 'ny' in line and 'nz' in line:
                has_normals = True
            elif line == 'end_header':
                header_end = i
                break
        
        if header_end == -1:
            print("Warning: Invalid PLY file format. Using default cube.")
            self._create_default_cube()
            return
        
        for i in range(header_end + 1, header_end + 1 + vertex_count):
            if i >= len(lines):
                break
            parts = lines[i].strip().split()
            if len(parts) >= 3:
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                if has_normals and len(parts) >= 6:
                    normals.append([float(parts[3]), float(parts[4]), float(parts[5])])
        
        for i in range(header_end + 1 + vertex_count, min(header_end + 1 + vertex_count + face_count, len(lines))):
            line = lines[i].strip()
            parts = line.split()
            if len(parts) >= 4:  # At least triangle
                n_verts = int(parts[0])
                face_verts = [int(x) for x in parts[1:1+n_verts]]
                
                if n_verts == 3:
                    faces.append(face_verts)
                elif n_verts == 4:
                    faces.append([face_verts[0], face_verts[1], face_verts[2]])
                    faces.append([face_verts[2], face_verts[1], face_verts[3]])
        
        if not vertices:
            print("Warning: No vertices found in PLY file. Using default cube.")
            self._create_default_cube()
            return
            
        self.vertices = np.array(vertices, dtype=np.float32)
        
        indices = []
        for face in faces:
            indices.extend(face)
        
        self.indices = np.array(indices, dtype=np.int32)
        
        if normals:
            self.normals = np.array(normals, dtype=np.float32)
        else:
            self._generate_normals()
        
        self._generate_colors()
        
        self._normalize_model()
    
    def _generate_normals(self):
        """Generate vertex normals from face normals"""
        normals = np.zeros_like(self.vertices)
        
        for i in range(0, len(self.indices), 3):
            if i + 2 >= len(self.indices):
                break
                
            i0, i1, i2 = self.indices[i], self.indices[i+1], self.indices[i+2]
            
            if i0 >= len(self.vertices) or i1 >= len(self.vertices) or i2 >= len(self.vertices):
                continue
            
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
        """Generate colors based on vertex positions"""
        v_min = self.vertices.min(axis=0)
        v_max = self.vertices.max(axis=0)
        v_range = v_max - v_min
        v_range[v_range == 0] = 1
        
        normalized_vertices = (self.vertices - v_min) / v_range
        
        self.colors = np.array([
            [0.2 + 0.8 * v[0], 0.2 + 0.8 * v[1], 0.2 + 0.8 * v[2]] 
            for v in normalized_vertices
        ], dtype=np.float32)
    
    def _normalize_model(self):
        """Center and scale model to fit in [-2, 2] range"""
        if len(self.vertices) == 0:
            return
            
        center = self.vertices.mean(axis=0)
        self.vertices -= center
        
        max_dist = np.max(np.abs(self.vertices))
        if max_dist > 0:
            scale = 2.0 / max_dist
            self.vertices *= scale

    def setup(self):
        """Setup buffers and shader"""
        self.vao = VAO()
        
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        
        self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)
        
        self.vao.add_ebo(self.indices)
        
        self.shader = Shader(self.vert_shader, self.frag_shader)
        self.uma = UManager(self.shader)
        
        self.lighting = LightingManager(self.uma)
        
        return self

    def draw(self, projection, view, model):
        """Draw the loaded model"""
        if model is None:
            model = T.identity()
            
        GL.glUseProgram(self.shader.render_idx)
        modelview = view @ model
        
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        
        if 'gouraud' in self.vert_shader.lower():
            self.lighting.setup_gouraud()
        elif 'phong' in self.vert_shader.lower():
            self.lighting.setup_phong(mode=1)
        else:
            self.lighting.setup_phong(mode=0)
        
        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLES, len(self.indices), GL.GL_UNSIGNED_INT, None)

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'vao'):
            self.vao.delete()
        if hasattr(self, 'shader'):
            self.shader.delete()
