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
        # loss_func là "hàm mục tiêu" cần tối ưu.
        # Từ loss này, visualizer sẽ:
        # 1. dựng mặt 3D z = f(x, y)
        # 2. thả các optimizer lên mặt đó
        # 3. cập nhật vị trí từng optimizer theo gradient
        # 4. vẽ quỹ đạo di chuyển của chúng
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
        self.base_plane_z = -2.02
        
        # Khởi tạo xong state thì sinh luôn mesh của mặt loss.
        self._generate_surface()
    
    def _generate_surface(self):
        # BƯỚC 1: tạo lưới mẫu đều trên miền x-y.
        x_vals = np.linspace(self.x_range[0], self.x_range[1], self.resolution)
        y_vals = np.linspace(self.y_range[0], self.y_range[1], self.resolution)
        
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.zeros_like(X)
        
        # BƯỚC 2: tính giá trị loss tại từng điểm lưới để thu được độ cao Z.
        for i in range(self.resolution):
            for j in range(self.resolution):
                Z[i, j] = self.loss_func.compute(X[i, j], Y[i, j])
        
        Z = np.nan_to_num(Z, nan=0.0, posinf=100.0, neginf=-100.0)
        Z = np.clip(Z, -50.0, 100.0)
        
        z_min, z_max = Z.min(), Z.max()
        # BƯỚC 3: chuẩn hóa Z về [0, 1] để việc tô màu và scale ổn định hơn.
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
        
        # BƯỚC 4: ánh xạ miền x-y-z thực sang không gian vẽ cỡ [-2, 2].
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
        
        # BƯỚC 5: từ mỗi điểm lưới, sinh:
        # - vertex để vẽ
        # - color để nhìn độ cao trực quan
        # - normal để chiếu sáng
        # - index để tách ô vuông thành 2 tam giác
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
                    # Normal của mặt loss cũng đi theo ý tưởng bề mặt z = f(x, y):
                    # lấy đạo hàm gần đúng theo x và y rồi dựng vector pháp tuyến.
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
                    # Mỗi ô lưới được cắt thành 2 tam giác để GPU dễ render.
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
        # BƯỚC 6: sau khi có dữ liệu CPU, tạo VAO/VBO/EBO và shader để render trên GPU.
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
    
    def add_optimizer(self, name, optimizer_type, initial_pos=None):
        # Mỗi optimizer được lưu như một gói state riêng:
        # vị trí hiện tại, loss hiện tại, độ lớn gradient, history quỹ đạo,
        # buffer momentum, state Adam...
        if initial_pos is None:
            initial_pos = [
                np.random.uniform(self.x_range[0], self.x_range[1]),
                np.random.uniform(self.y_range[0], self.y_range[1])
            ]

        # Tạo seed cố định để nhiễu của SGD / MiniBatch ổn định giữa các lần demo.
        seed = sum(ord(ch) for ch in f"{name}:{optimizer_type}")
        
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
            'rng': np.random.default_rng(seed),
        }
        
        self.trajectories[name] = []

    def _estimate_stochastic_gradient(self, opt, grad, batch_size, mode):
        # Hàm này mô phỏng ý tưởng "stochastic":
        # gradient không còn là gradient đầy đủ chính xác nữa,
        # mà có nhiễu hoặc được lấy trung bình trên mini-batch.
        if mode == 'GD':
            return grad

        batch = max(1, int(batch_size))
        rng = opt['rng']

        if mode == 'SGD':
            # SGD: lấy gradient nhiễu hơn, tương đương nhìn thấy một mẫu nhỏ nên hướng đi dao động.
            scale = 0.18 #độ nhiễu càng lớn càng rung càng nhỏ thì sẽ càng giống GD
            #phân phối chuẩn có trung bình 0 và phương sai 𝜎^2
            noise = rng.normal(0.0, scale, size=grad.shape).astype(np.float32) # f(x) = 1/σ√2π * exp(-x²/2σ²)
            return grad * (1.0 + noise)

        if mode == 'MiniBatch':
            # MiniBatch: lấy trung bình nhiều gradient nhiễu,
            # nên đỡ "giật" hơn SGD thuần nhưng vẫn không mượt như GD toàn phần.
            scale = 0.25 / np.sqrt(batch)
            samples = []
            for _ in range(batch):
                noise = rng.normal(0.0, scale, size=grad.shape).astype(np.float32)
                samples.append(grad * (1.0 + noise))
            return np.mean(samples, axis=0).astype(np.float32)

        return grad
    
    def step_optimizer(self, name, learning_rate, momentum=0.0, batch_size=1):
        if name not in self.optimizers:
            return
        
        opt = self.optimizers[name]
        x, y = opt['position']
        
        # BƯỚC 7: tại vị trí hiện tại, tính gradient của loss.
        # Gradient chính là hướng dốc nhất đi lên,
        # nên optimizer sẽ đi ngược dấu gradient để đi xuống.
        grad = self.loss_func.gradient(x, y).astype(np.float32)
        effective_grad = grad
        
        if opt['type'] == 'GD':
            # đi ngược chiều gradient để tìm điểm tối ưu
            # Gradient Descent chuẩn: dùng đúng full gradient.
            # đạo hàm của 1 hàm là độ dốc của đồ thị tại 1 điểm
            effective_grad = grad
            opt['position'] = opt['position'] - learning_rate * effective_grad
            
        elif opt['type'] == 'SGD':
            # dùng 1 gradient cho toàn bộ dữ liệu
            # SGD: dùng gradient nhiễu để tạo cảm giác "nhảy" khi tối ưu.
            effective_grad = self._estimate_stochastic_gradient(opt, grad, batch_size, 'SGD')
            opt['position'] = opt['position'] - learning_rate * effective_grad
            
        elif opt['type'] == 'MiniBatch':
            # MiniBatch: gradient là trung bình của một nhóm mẫu nhỏ.
            effective_grad = self._estimate_stochastic_gradient(opt, grad, batch_size, 'MiniBatch')
            opt['position'] = opt['position'] - learning_rate * effective_grad
            
        elif opt['type'] == 'Momentum':
            # Momentum cộng "quán tính" vào bước cập nhật:
            # buffer mới = momentum * buffer cũ + gradient hiện tại.
            effective_grad = grad
            #vt = 𝜇 vt-1 + gt
            # Pt+1 = Pt - 𝜂 vt
            opt['momentum_buffer'] = momentum * opt['momentum_buffer'] + effective_grad
            opt['position'] = opt['position'] - learning_rate * opt['momentum_buffer']
            
        elif opt['type'] == 'Nesterov':
            # Nesterov nhìn trước một bước (lookahead), rồi mới tính gradient tại vị trí dự báo đó.
            # Nesterov thì “nhìn trước” xem nếu quán tính kéo mình đi một đoạn thì mình sẽ ở đâu, rồi tính gradient ở đó.
            nesterov_lr = learning_rate * 0.5  # Slow down Nesterov
            lookahead_pos = opt['position'] - momentum * opt['momentum_buffer']
            effective_grad = self.loss_func.gradient(lookahead_pos[0], lookahead_pos[1]).astype(np.float32)
            opt['momentum_buffer'] = momentum * opt['momentum_buffer'] + effective_grad
            opt['position'] = opt['position'] - nesterov_lr * opt['momentum_buffer']
            
        elif opt['type'] == 'Adam':
            # Adam = Adaptive Moment Estimation
            # Mục tiêu: cập nhật tham số để loss giảm dần, ổn định hơn GD thường.
            #
            # Ý tưởng chính:
            # 1) Dùng m để nhớ "xu hướng" gradient gần đây  -> giống momentum
            # 2) Dùng v để nhớ "độ lớn" gradient từng chiều -> tự chỉnh bước đi theo từng trục
            #
            # Ký hiệu:
            # p_t      : vị trí/tham số tại bước t
            # g_t      : gradient tại bước t
            # m_t      : moment bậc 1 (trung bình động của gradient)
            # v_t      : moment bậc 2 (trung bình động của gradient bình phương)
            # beta1    : hệ số nhớ của m, thường ~ 0.9
            # beta2    : hệ số nhớ của v, thường ~ 0.999
            # eta      : learning rate
            # epsilon  : số rất nhỏ để tránh chia cho 0
            #
            # Công thức Adam:
            # g_t = ∇f(p_t)
            #
            # m_t = beta1 * m_(t-1) + (1 - beta1) * g_t
            # v_t = beta2 * v_(t-1) + (1 - beta2) * (g_t)^2
            #
            # Bias correction:
            # m_hat = m_t / (1 - beta1^t)
            # v_hat = v_t / (1 - beta2^t)
            #
            # Update:
            # p_(t+1) = p_t - eta * m_hat / (sqrt(v_hat) + epsilon)
            #
            # Trực giác:
            # - m_hat cho biết nên đi theo hướng nào
            # - sqrt(v_hat) cho biết chiều nào quá dốc thì giảm bước lại
            # => Adam = có quán tính + tự điều chỉnh learning rate theo từng chiều
            t = opt['step'] + 1
            effective_grad = grad
            opt['adam_m'] = opt['beta1'] * opt['adam_m'] + (1 - opt['beta1']) * effective_grad
            opt['adam_v'] = opt['beta2'] * opt['adam_v'] + (1 - opt['beta2']) * (effective_grad ** 2)
            
            m_hat = opt['adam_m'] / (1 - opt['beta1'] ** t)
            v_hat = opt['adam_v'] / (1 - opt['beta2'] ** t)
            
            update = learning_rate * m_hat / (np.sqrt(v_hat) + opt['epsilon'])
            opt['position'] = opt['position'] - update

        opt['gradient_mag'] = float(np.linalg.norm(effective_grad))
        
        # BƯỚC 8: chặn vị trí lại trong miền đang vẽ để optimizer không "bay" ra ngoài scene.
        opt['position'][0] = np.clip(opt['position'][0], self.x_range[0], self.x_range[1])
        opt['position'][1] = np.clip(opt['position'][1], self.y_range[0], self.y_range[1])
        
        # BƯỚC 9: cập nhật loss, số bước và lưu history để lát nữa còn vẽ quỹ đạo.
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
        # Hàm phụ này đổi tọa độ toán học thật (x, y, z của loss)
        # sang tọa độ hiển thị trong scene OpenGL.
        dx = (x - self.x_range[0]) * self.surface_x_scale - self.surface_x_offset
        dy = (y - self.y_range[0]) * self.surface_y_scale - self.surface_y_offset
        
        if z is None:
            z = self.loss_func.compute(x, y)
            z_norm = np.clip((z - self.surface_z_min) / (self.surface_z_max - self.surface_z_min), 0, 1)
            dz = (z_norm - 0.5) * 4.0
        else:
            dz = z
        
        return [dx, dy, dz]
    
    def draw(self, projection, view, wireframe_mode=0, display_mode=0, cam_far=100.0, show_trajectory=True):
        # BƯỚC 10: vẽ toàn bộ cảnh SGD theo thứ tự:
        # 1. vẽ mặt loss
        # 2. vẽ marker của từng optimizer
        # 3. vẽ đường rơi xuống đáy
        # 4. vẽ quỹ đạo nếu được bật
        GL.glUseProgram(self.surface_shader.render_idx)
        
        modelview = view @ np.eye(4, dtype=np.float32)
        
        self.surface_uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.surface_uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        
        self.surface_lighting.setup_phong(mode=1, view_matrix=view, display_mode=display_mode, cam_far=cam_far)
        
        if wireframe_mode == 1:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        elif wireframe_mode == 2:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_POINT)
        
        self.surface_vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLES, len(self.surface_indices), GL.GL_UNSIGNED_INT, None)
        
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        
        for name, opt in self.optimizers.items():
            # Mỗi optimizer là một quả cầu nhỏ đang chạy trên bề mặt loss.
            pos = opt['position']
            dx, dy, dz = self.get_draw_coords(pos[0], pos[1])
            
            color = self.optimizer_colors.get(opt['type'], [1, 0, 1])
            self._draw_drop_line(dx, dy, dz, color, projection, view)
            self._draw_sphere(dx, dy, dz, color, projection, view, wireframe_mode, display_mode, cam_far)
            
            if show_trajectory and len(opt['history']) >= 2:
                self._draw_trail(opt['history'], opt['type'], projection, view)
                self._draw_trail_projection(opt['history'], opt['type'], projection, view)
    
    def _draw_sphere(self, x, y, z, color, projection, view, wireframe_mode=0, display_mode=0, cam_far=100.0):
        # Marker của optimizer được vẽ như một quả cầu nhỏ để dễ nhìn trong không gian 3D.
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
        
        if wireframe_mode == 1:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        elif wireframe_mode == 2:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_POINT)
        else:
            self.marker_lighting.setup_phong(mode=1, view_matrix=view, display_mode=display_mode, cam_far=cam_far)
        
        sphere_vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLES, len(indices), GL.GL_UNSIGNED_INT, None)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
    
    def _draw_trail(self, history, opt_type, projection, view):
        if len(history) < 2:
            return
        
        # Quỹ đạo chính là nối các vị trí lịch sử lại thành nhiều đoạn thẳng liên tiếp.
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

    def _draw_trail_projection(self, history, opt_type, projection, view):
        if len(history) < 2:
            return

        # Đây là bản chiếu của quỹ đạo xuống mặt đáy để người xem dễ theo dõi hướng đi trên mặt phẳng x-y.
        color = np.array(self.optimizer_colors.get(opt_type, [1, 1, 1]), dtype=np.float32) * 0.55
        vertices = []
        colors = []

        for i in range(len(history) - 1):
            p1 = history[i]
            p2 = history[i + 1]

            x1, y1, _ = self.get_draw_coords(p1[0], p1[1])
            x2, y2, _ = self.get_draw_coords(p2[0], p2[1])

            vertices.append([x1, y1, self.base_plane_z])
            vertices.append([x2, y2, self.base_plane_z])
            colors.append(color)
            colors.append(color)

        if vertices:
            trail_vao = VAO()
            trail_vao.add_vbo(0, np.array(vertices, dtype=np.float32), ncomponents=3, stride=0, offset=None)
            trail_vao.add_vbo(1, np.array(colors, dtype=np.float32), ncomponents=3, stride=0, offset=None)

            GL.glUseProgram(self.trail_shader.render_idx)
            modelview = view @ np.eye(4, dtype=np.float32)
            self.trail_uma.upload_uniform_matrix4fv(projection, 'projection', True)
            self.trail_uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

            trail_vao.activate()
            GL.glDrawArrays(GL.GL_LINES, 0, len(vertices))

    def _draw_drop_line(self, x, y, z, color, projection, view):
        # Đường thả đứng giúp người xem nhìn rõ:
        # optimizer đang nằm ở độ cao loss bao nhiêu so với mặt đáy tham chiếu.
        line_color = np.array(color, dtype=np.float32) * 0.7
        vertices = np.array([
            [x, y, self.base_plane_z],
            [x, y, z],
        ], dtype=np.float32)
        colors = np.array([line_color, line_color], dtype=np.float32)

        line_vao = VAO()
        line_vao.add_vbo(0, vertices, ncomponents=3, stride=0, offset=None)
        line_vao.add_vbo(1, colors, ncomponents=3, stride=0, offset=None)

        GL.glUseProgram(self.trail_shader.render_idx)
        modelview = view @ np.eye(4, dtype=np.float32)
        self.trail_uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.trail_uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        line_vao.activate()
        GL.glDrawArrays(GL.GL_LINES, 0, 2)
    
    def reset_optimizer(self, name, initial_pos=None):
        if name not in self.optimizers:
            return
        
        if initial_pos is None:
            initial_pos = [
                np.random.uniform(self.x_range[0], self.x_range[1]),
                np.random.uniform(self.y_range[0], self.y_range[1])
            ]
        
        # Reset nghĩa là đưa toàn bộ state của optimizer về trạng thái ban đầu:
        # vị trí, loss, gradient, history, momentum buffer, Adam moments...
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
