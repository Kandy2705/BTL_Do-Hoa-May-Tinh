import sys
import os
import numpy as np
import OpenGL.GL as GL

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.shader import Shader
from libs.buffer import VAO, UManager
from libs.lighting import LightingManager


class SGDVisualizer:
    def __init__(self, loss_func, x_range=(-5, 5), y_range=(-5, 5), resolution=80):
        self.loss_func = loss_func
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        
        self.optimizers = {}
        self.trajectories = {}
        self.optimizer_colors = {
            'GD': [1.0, 0.0, 0.0],
            'SGD': [0.0, 1.0, 0.0],
            'MiniBatch': [0.0, 0.0, 1.0],
            'Momentum': [1.0, 1.0, 0.0],
            'Nesterov': [1.0, 0.5, 0.0],
            'Adam': [1.0, 0.0, 1.0],
        }
        
        self.surface_vao = None
        self.surface_shader = None
        self.surface_uma = None
        self.surface_vertices = None
        self.surface_colors = None
        self.surface_normals = None
        self.surface_indices = None
        
        self.marker_vao = None
        self.marker_shader = None
        self.marker_uma = None
        self.marker_vertices = None
        
        self.trail_vao = None
        self.trail_shader = None
        self.trail_uma = None
        
        self.display_mode = 'surface'
        self.show_grid = True
        self.surface_z_min = -2.0  # Will be set in _generate_surface
        self.surface_z_max = 2.0   # Will be set in _generate_surface
        self.surface_x_scale = 1.0
        self.surface_y_scale = 1.0
        self.surface_x_offset = 0.0
        self.surface_y_offset = 0.0
        
        self._generate_surface()
    
    def _generate_surface(self):
        x_vals = np.linspace(self.x_range[0], self.x_range[1], self.resolution)
        y_vals = np.linspace(self.y_range[0], self.y_range[1], self.resolution)
        
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.zeros_like(X)
        
        for i in range(self.resolution):
            for j in range(self.resolution):
                Z[i, j] = self.loss_func.compute(X[i, j], Y[i, j])
        
        Z = np.nan_to_num(Z, nan=0.0, posinf=100.0, neginf=-100.0)
        Z = np.clip(Z, -50.0, 100.0)
        
        z_min, z_max = Z.min(), Z.max()
        if z_max - z_min > 0:
            Z_norm = (Z - z_min) / (z_max - z_min)
        else:
            Z_norm = np.zeros_like(Z)
        
        vertices = []
        colors = []
        normals = []
        indices = []
        
        dx = x_vals[1] - x_vals[0]
        dy = y_vals[1] - y_vals[0]
        
        x_scale = 4.0 / (self.x_range[1] - self.x_range[0])
        y_scale = 4.0 / (self.y_range[1] - self.y_range[0])
        z_scale = 4.0 / max(z_max - z_min, 1.0)
        
        # Store for ball positioning
        self.surface_x_scale = x_scale
        self.surface_y_scale = y_scale
        self.surface_x_offset = 2.0
        self.surface_y_offset = 2.0
        self.surface_z_scale = z_scale
        self.surface_z_min = z_min
        self.surface_z_max = z_max
        
        for i in range(self.resolution):
            for j in range(self.resolution):
                x = (x_vals[j] - self.x_range[0]) * x_scale - 2.0
                y = (y_vals[i] - self.y_range[0]) * y_scale - 2.0
                z = (Z[i, j] - z_min) * z_scale - 2.0
                
                vertices.append([x, y, z])
                
                h = Z_norm[i, j]
                colors.append([
                    0.2 + 0.6 * h,
                    0.4 * (1 - abs(h - 0.5) * 2),
                    0.6 + 0.4 * (1 - h)
                ])
                
                if 0 < i < self.resolution - 1 and 0 < j < self.resolution - 1:
                    dz_dx = (Z[i, j+1] - Z[i, j-1]) / (2 * dx)
                    dz_dy = (Z[i+1, j] - Z[i-1, j]) / (2 * dy)
                    
                    nx = -dz_dx * x_scale / z_scale
                    ny = -dz_dy * y_scale / z_scale
                    nz = 1.0
                    norm = np.sqrt(nx*nx + ny*ny + nz*nz)
                    normals.append([nx/norm, ny/norm, nz/norm])
                else:
                    normals.append([0.0, 0.0, 1.0])
                
                if i < self.resolution - 1 and j < self.resolution - 1:
                    v0 = i * self.resolution + j
                    v1 = i * self.resolution + (j + 1)
                    v2 = (i + 1) * self.resolution + j
                    v3 = (i + 1) * self.resolution + (j + 1)
                    indices.extend([v0, v1, v2, v2, v1, v3])
        
        self.surface_vertices = np.array(vertices, dtype=np.float32)
        self.surface_colors = np.array(colors, dtype=np.float32)
        self.surface_normals = np.array(normals, dtype=np.float32)
        self.surface_indices = np.array(indices, dtype=np.int32)
    
    def setup(self):
        self.surface_vao = VAO()
        self.surface_vao.add_vbo(0, self.surface_vertices, ncomponents=3, stride=0, offset=None)
        self.surface_vao.add_vbo(1, self.surface_colors, ncomponents=3, stride=0, offset=None)
        self.surface_vao.add_vbo(2, self.surface_normals, ncomponents=3, stride=0, offset=None)
        self.surface_vao.add_ebo(self.surface_indices)
        
        self.surface_shader = Shader("./shaders/phong.vert", "./shaders/phong.frag")
        self.surface_uma = UManager(self.surface_shader)
        self.surface_lighting = LightingManager(self.surface_uma)
        
        self.marker_shader = Shader("./shaders/phong.vert", "./shaders/phong.frag")
        self.marker_uma = UManager(self.marker_shader)
        self.marker_lighting = LightingManager(self.marker_uma)
        
        self.trail_shader = Shader("./shaders/color_interp.vert", "./shaders/color_interp.frag")
        self.trail_uma = UManager(self.trail_shader)
        
        self._setup_marker_vao()
        self._setup_trail_vao()
    
    def _setup_marker_vao(self):
        pass
    
    def _setup_trail_vao(self):
        pass
    
    def add_optimizer(self, name, optimizer_type, initial_pos=None):
        if initial_pos is None:
            initial_pos = [
                np.random.uniform(self.x_range[0], self.x_range[1]),
                np.random.uniform(self.y_range[0], self.y_range[1])
            ]
        
        self.optimizers[name] = {
            'type': optimizer_type,
            'position': np.array(initial_pos, dtype=np.float32),
            'velocity': np.zeros(2, dtype=np.float32),
            'loss': self.loss_func.compute(initial_pos[0], initial_pos[1]),
            'gradient_mag': 0.0,
            'step': 0,
            'history': [initial_pos.copy()],
            'running': False,
            'momentum_buffer': np.zeros(2, dtype=np.float32),
            'adam_m': np.zeros(2, dtype=np.float32),
            'adam_v': np.zeros(2, dtype=np.float32),
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
        }
        
        self.trajectories[name] = []
    
    def step_optimizer(self, name, learning_rate, momentum=0.0, batch_size=1):
        if name not in self.optimizers:
            return
        
        opt = self.optimizers[name]
        x, y = opt['position']
        
        grad = self.loss_func.gradient(x, y)
        opt['gradient_mag'] = np.linalg.norm(grad)
        
        if opt['type'] == 'GD':
            opt['position'] = opt['position'] - learning_rate * grad
            
        elif opt['type'] == 'SGD':
            opt['position'] = opt['position'] - learning_rate * grad
            
        elif opt['type'] == 'MiniBatch':
            grad_est = grad
            opt['position'] = opt['position'] - learning_rate * grad_est
            
        elif opt['type'] == 'Momentum':
            opt['momentum_buffer'] = momentum * opt['momentum_buffer'] + grad
            opt['position'] = opt['position'] - learning_rate * opt['momentum_buffer']
            
        elif opt['type'] == 'Nesterov':
            nesterov_lr = learning_rate * 0.5  # Slow down Nesterov
            lookahead_pos = opt['position'] - momentum * opt['momentum_buffer']
            nesterov_grad = self.loss_func.gradient(lookahead_pos[0], lookahead_pos[1])
            opt['momentum_buffer'] = momentum * opt['momentum_buffer'] + nesterov_grad
            opt['position'] = opt['position'] - nesterov_lr * opt['momentum_buffer']
            
        elif opt['type'] == 'Adam':
            t = opt['step'] + 1
            opt['adam_m'] = opt['beta1'] * opt['adam_m'] + (1 - opt['beta1']) * grad
            opt['adam_v'] = opt['beta2'] * opt['adam_v'] + (1 - opt['beta2']) * (grad ** 2)
            
            m_hat = opt['adam_m'] / (1 - opt['beta1'] ** t)
            v_hat = opt['adam_v'] / (1 - opt['beta2'] ** t)
            
            update = learning_rate * m_hat / (np.sqrt(v_hat) + opt['epsilon'])
            opt['position'] = opt['position'] - update
        
        # Clip position to domain bounds
        opt['position'][0] = np.clip(opt['position'][0], self.x_range[0], self.x_range[1])
        opt['position'][1] = np.clip(opt['position'][1], self.y_range[0], self.y_range[1])
        
        opt['loss'] = self.loss_func.compute(opt['position'][0], opt['position'][1])
        opt['step'] += 1
        opt['history'].append(opt['position'].copy())
    
    def update_trajectory(self, name):
        if name not in self.optimizers:
            return
        
        opt = self.optimizers[name]
        if len(opt['history']) >= 2:
            p1 = opt['history'][-2]
            p2 = opt['history'][-1]
            self.trajectories[name].append((p1, p2, self.optimizer_colors.get(opt['type'], [1, 1, 1])))
    
    def get_draw_coords(self, x, y, z=None):
        dx = (x - self.x_range[0]) * self.surface_x_scale - self.surface_x_offset
        dy = (y - self.y_range[0]) * self.surface_y_scale - self.surface_y_offset
        
        if z is None:
            z = self.loss_func.compute(x, y)
            z_norm = np.clip((z - self.surface_z_min) / (self.surface_z_max - self.surface_z_min), 0, 1)
            dz = (z_norm - 0.5) * 4.0
        else:
            dz = z
        
        return [dx, dy, dz]
    
    def draw(self, projection, view, wireframe_mode=0):
        GL.glUseProgram(self.surface_shader.render_idx)
        
        modelview = view @ np.eye(4, dtype=np.float32)
        
        self.surface_uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.surface_uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        
        self.surface_lighting.setup_phong(mode=1)
        
        if wireframe_mode == 1:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        elif wireframe_mode == 2:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_POINT)
        
        self.surface_vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLES, len(self.surface_indices), GL.GL_UNSIGNED_INT, None)
        
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        
        for name, opt in self.optimizers.items():
            pos = opt['position']
            dx, dy, dz = self.get_draw_coords(pos[0], pos[1])
            
            color = self.optimizer_colors.get(opt['type'], [1, 0, 1])
            self._draw_sphere(dx, dy, dz, color, projection, view)
            
            if len(opt['history']) >= 2:
                self._draw_trail(opt['history'], opt['type'], projection, view)
    
    def _draw_sphere(self, x, y, z, color, projection, view):
        lat_div, long_div = 12, 12
        radius = 0.05
        
        vertices = []
        normals = []
        colors = []
        
        for i in range(lat_div + 1):
            lat = np.pi * (-0.5 + i / lat_div)
            for j in range(long_div + 1):
                lon = 2 * np.pi * j / long_div
                nx = np.cos(lat) * np.cos(lon)
                ny = np.sin(lat)
                nz = np.cos(lat) * np.sin(lon)
                
                vx = x + radius * nx
                vy = y + radius * ny
                vz = z + radius * nz
                
                vertices.append([vx, vy, vz])
                normals.append([nx, ny, nz])
                colors.append(color)
        
        indices = []
        for i in range(lat_div):
            for j in range(long_div):
                first = i * (long_div + 1) + j
                second = first + long_div + 1
                indices.extend([first, second, first + 1])
                indices.extend([second, second + 1, first + 1])
        
        sphere_vao = VAO()
        sphere_vao.add_vbo(0, np.array(vertices, dtype=np.float32), ncomponents=3, stride=0, offset=None)
        sphere_vao.add_vbo(1, np.array(colors, dtype=np.float32), ncomponents=3, stride=0, offset=None)
        sphere_vao.add_vbo(2, np.array(normals, dtype=np.float32), ncomponents=3, stride=0, offset=None)
        sphere_vao.add_ebo(np.array(indices, dtype=np.int32))
        
        GL.glUseProgram(self.marker_shader.render_idx)
        
        modelview = view @ np.eye(4, dtype=np.float32)
        self.marker_uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.marker_uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        
        self.marker_lighting.setup_phong(mode=1)
        
        sphere_vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLES, len(indices), GL.GL_UNSIGNED_INT, None)
    
    def _draw_trail(self, history, opt_type, projection, view):
        if len(history) < 2:
            return
        
        color = self.optimizer_colors.get(opt_type, [1, 1, 1])
        vertices = []
        colors = []
        
        for i in range(len(history) - 1):
            p1 = history[i]
            p2 = history[i + 1]
            
            x1, y1, z1 = self.get_draw_coords(p1[0], p1[1])
            x2, y2, z2 = self.get_draw_coords(p2[0], p2[1])
            
            vertices.append([x1, y1, z1])
            vertices.append([x2, y2, z2])
            colors.append(color)
            colors.append(color)
        
        if vertices:
            trail_verts = np.array(vertices, dtype=np.float32)
            trail_colors = np.array(colors, dtype=np.float32)
            
            trail_vao = VAO()
            trail_vao.add_vbo(0, trail_verts, ncomponents=3, stride=0, offset=None)
            trail_vao.add_vbo(1, trail_colors, ncomponents=3, stride=0, offset=None)
            
            GL.glUseProgram(self.trail_shader.render_idx)
            
            modelview = view @ np.eye(4, dtype=np.float32)
            self.trail_uma.upload_uniform_matrix4fv(projection, 'projection', True)
            self.trail_uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
            
            trail_vao.activate()
            GL.glDrawArrays(GL.GL_LINES, 0, len(vertices))
    
    def set_loss_function(self, loss_func):
        self.loss_func = loss_func
        self._generate_surface()
        if self.surface_vao:
            self.surface_vao.delete()
            self.setup()
    
    def reset_optimizer(self, name, initial_pos=None):
        if name not in self.optimizers:
            return
        
        if initial_pos is None:
            initial_pos = [
                np.random.uniform(self.x_range[0], self.x_range[1]),
                np.random.uniform(self.y_range[0], self.y_range[1])
            ]
        
        opt = self.optimizers[name]
        opt['position'] = np.array(initial_pos, dtype=np.float32)
        opt['velocity'] = np.zeros(2, dtype=np.float32)
        opt['loss'] = self.loss_func.compute(initial_pos[0], initial_pos[1])
        opt['gradient_mag'] = 0.0
        opt['step'] = 0
        opt['history'] = [initial_pos.copy()]
        opt['momentum_buffer'] = np.zeros(2, dtype=np.float32)
        opt['adam_m'] = np.zeros(2, dtype=np.float32)
        opt['adam_v'] = np.zeros(2, dtype=np.float32)
    
    def get_optimizer_state(self, name):
        if name not in self.optimizers:
            return None
        return self.optimizers[name]
    
    def toggle_optimizer(self, name):
        if name not in self.optimizers:
            return
        self.optimizers[name]['running'] = not self.optimizers[name]['running']
    
    def is_running(self, name):
        if name not in self.optimizers:
            return False
        return self.optimizers[name]['running']
