# Hình lăng trụ

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.shader import *
from libs import transform as T
from libs.buffer import *
from libs.lighting import LightingManager
import ctypes
import glfw
import numpy as np
import OpenGL.GL as GL

class Prism:
    def __init__(self, vert_shader, frag_shader, sides=3, radius=1.0, height=2.0):
        self.vert_shader = vert_shader
        self.frag_shader = frag_shader
        
        vertices = []
        normals = []
        colors = []
        indices = []

        for i in range(sides + 1):
            theta = 2.0 * np.pi * i / sides
            x = radius * np.cos(theta)
            z = radius * np.sin(theta)
            
            r = (np.cos(theta) + 1.0) / 2.0
            g = (np.sin(theta) + 1.0) / 2.0
            b = 1
            
            vertices.append([x, -height/2, z])
            normals.append([np.cos(theta), 0, np.sin(theta)])
            colors.append([r, g, b]) 
            
            vertices.append([x, height/2, z])
            normals.append([np.cos(theta), 0, np.sin(theta)])
            colors.append([r, g, b])

        bottom_center_idx = len(vertices)
        vertices.append([0, -height/2, 0])
        normals.append([0, -1, 0])
        colors.append([0.0, 0.0, 1.0])

        top_center_idx = len(vertices)
        vertices.append([0, height/2, 0])
        normals.append([0, 1, 0])
        colors.append([0.0, 0.0, 1.0])

        for i in range(sides):
            b1 = i * 2
            t1 = i * 2 + 1
            b2 = (i + 1) * 2
            t2 = (i + 1) * 2 + 1
            indices.extend([b1, t1, b2, t1, t2, b2])

        for i in range(sides):
            b1 = i * 2
            b2 = (i + 1) * 2
            indices.extend([bottom_center_idx, b2, b1])

        for i in range(sides):
            t1 = i * 2 + 1
            t2 = (i + 1) * 2 + 1
            indices.extend([top_center_idx, t1, t2])

        self.vertices = np.array(vertices, dtype=np.float32)
        self.normals = np.array(normals, dtype=np.float32)
        self.colors = np.array(colors, dtype=np.float32)
        
        self.indices = np.array(indices, dtype=np.int32)

        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.lighting = LightingManager(self.uma)

    def setup(self):
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        
        if 'gouraud' in self.vert_shader.lower() or 'phong' in self.vert_shader.lower():
            self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)

        self.vao.add_ebo(self.indices)

        return self

    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)
        
        modelview = view if model is None else view @ model

        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        
        if 'gouraud' in self.vert_shader.lower():
            self.lighting.setup_gouraud()
        elif 'phong' in self.vert_shader.lower():
            self.lighting.setup_phong(mode=1)

        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLES, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)

    def key_handler(self, key):

        if key == glfw.KEY_1:
            self.selected_texture = 1
        if key == glfw.KEY_2:
            self.selected_texture = 2