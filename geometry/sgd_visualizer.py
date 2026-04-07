import sys
import os
import numpy as np
import OpenGL.GL as GL

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.shader import Shader
from libs.buffer import VAO, UManager
from libs.lighting import LightingManager


class SGDVisualizer:
    NORMAL_OPTIMIZER_COLORS = {
        'GD': [0.90, 0.28, 0.24],
        'SGD': [0.18, 0.72, 0.44],
        'MiniBatch': [0.20, 0.46, 0.90],
        'Momentum': [0.95, 0.78, 0.20],
        'Nesterov': [0.95, 0.52, 0.16],
        'Adam': [0.72, 0.24, 0.92],
    }
    # Palette phân biệt tốt hơn với người mù màu đỏ-lục (tham chiếu nhóm Okabe-Ito mở rộng).
    COLORBLIND_OPTIMIZER_COLORS = {
        'GD': [0.00, 0.45, 0.70],        # blue
        'SGD': [0.80, 0.47, 0.65],       # reddish purple
        'MiniBatch': [0.00, 0.62, 0.45], # bluish green
        'Momentum': [0.95, 0.90, 0.25],  # yellow
        'Nesterov': [0.90, 0.60, 0.00],  # orange
        'Adam': [0.35, 0.35, 0.35],      # gray
    }

    def __init__(self, loss_func, x_range=(-5, 5), y_range=(-5, 5), resolution=180):
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
        self.optimizer_colors = dict(self.NORMAL_OPTIMIZER_COLORS)
        
        self.surface_vao = None
        self.surface_shader = None
        self.surface_uma = None
        self.surface_vertices = None
        self.surface_colors = None
        self.surface_normals = None
        self.surface_indices = None
        self.surface_display_z = None
        
        self.marker_vao = None
        self.marker_shader = None
        self.marker_uma = None
        self.marker_vertices = None
        
        self.trail_vao = None
        self.trail_shader = None
        self.trail_uma = None
        
        self.display_mode = 'surface'
        self.view_mode = 'combined'
        self.show_contours = True
        self.show_projected_trajectory = True
        self.show_drop_lines = True
        self.show_grid = False
        # OpenGL core profile (đặc biệt macOS) thường chỉ hỗ trợ line width = 1,
        # nên tăng "độ dày" quỹ đạo bằng nhiều line song song lệch nhẹ theo phương vuông góc.
        self.trail_thickness_passes = 5
        self.trail_thickness_offset = 0.008
        self.trail_width_scale = 1.0
        self.contour_levels = 14
        self.surface_z_min = -2.0  # Will be set in _generate_surface
        self.surface_z_max = 2.0   # Will be set in _generate_surface
        self.surface_z_scale = 1.0
        self.surface_x_scale = 1.0
        self.surface_y_scale = 1.0
        self.surface_x_offset = 0.0
        self.surface_y_offset = 0.0
        self.loss_display_mode = "identity"
        self.base_plane_z = -1.90
        self.contour_z = self.base_plane_z + 0.02
        self.x_vals = None
        self.y_vals = None
        self.z_grid = None
        self.z_norm_grid = None
        self.contour_vertices = np.zeros((0, 3), dtype=np.float32)
        self.contour_colors = np.zeros((0, 3), dtype=np.float32)
        self.contour_vao = None
        self.contour_fill_vertices = np.zeros((0, 3), dtype=np.float32)
        self.contour_fill_colors = np.zeros((0, 3), dtype=np.float32)
        self.contour_fill_indices = np.zeros((0,), dtype=np.int32)
        self.contour_fill_vao = None
        
        # Khởi tạo xong state thì sinh luôn mesh của mặt loss.
        self._generate_surface()

    def set_colorblind_palette(self, enabled=False):
        palette = self.COLORBLIND_OPTIMIZER_COLORS if enabled else self.NORMAL_OPTIMIZER_COLORS
        self.optimizer_colors = dict(palette)
    
    def _generate_surface(self):
        # BƯỚC 1: tạo lưới mẫu đều trên miền x-y.
        x_vals = np.linspace(self.x_range[0], self.x_range[1], self.resolution)
        y_vals = np.linspace(self.y_range[0], self.y_range[1], self.resolution)
        self.x_vals = x_vals
        self.y_vals = y_vals
        
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.zeros_like(X)
        
        # BƯỚC 2: tính giá trị loss tại từng điểm lưới để thu được độ cao Z.
        for i in range(self.resolution):
            for j in range(self.resolution):
                Z[i, j] = self.loss_func.compute(X[i, j], Y[i, j])
        
        Z = np.nan_to_num(Z, nan=0.0, posinf=100.0, neginf=-100.0)

        # Himmelblau và các benchmark không âm thường có biên rất lớn ở rìa.
        # Dùng log1p sẽ giữ được dáng "đỉnh - thung lũng" giống plot mẫu hơn tanh,
        # đồng thời tránh chuyện phần đỉnh bị ép phẳng thành một mặt bàn.
        Z_vis = self._transform_loss_array(Z)
        z_min, z_max = Z_vis.min(), Z_vis.max()
        if z_max - z_min > 0:
            Z_norm = (Z_vis - z_min) / (z_max - z_min)
        else:
            Z_norm = np.zeros_like(Z_vis)
        self.z_grid = Z
        self.surface_display_z = Z_vis
        self.z_norm_grid = Z_norm
        
        vertices = []
        colors = []
        normals = []
        indices = []
        
        dx = x_vals[1] - x_vals[0]
        dy = y_vals[1] - y_vals[0]
        
        # BƯỚC 4: ánh xạ miền x-y-z thực sang không gian vẽ cỡ [-2, 2].
        x_scale = 4.0 / (self.x_range[1] - self.x_range[0])
        y_scale = 4.0 / (self.y_range[1] - self.y_range[0])
        z_scale = 2.8 / max(z_max - z_min, 1.0)
        
        # Store for ball positioning
        self.surface_x_scale = x_scale
        self.surface_y_scale = y_scale
        self.surface_x_offset = 2.0
        self.surface_y_offset = 2.0
        self.surface_z_scale = z_scale
        self.surface_z_min = z_min
        self.surface_z_max = z_max
        self.base_plane_z = -1.70
        self.contour_z = self.base_plane_z + 0.01
        self.contour_fill_z = self.base_plane_z - 0.005
        
        # BƯỚC 5: từ mỗi điểm lưới, sinh:
        # - vertex để vẽ
        # - color để nhìn độ cao trực quan
        # - normal để chiếu sáng
        # - index để tách ô vuông thành 2 tam giác
        for i in range(self.resolution):
            for j in range(self.resolution):
                x = (x_vals[j] - self.x_range[0]) * x_scale - 2.0
                y = (y_vals[i] - self.y_range[0]) * y_scale - 2.0
                z = (Z_vis[i, j] - z_min) * z_scale - 0.9
                
                vertices.append([x, y, z])
                
                h = Z_norm[i, j]
                colors.append(self._loss_color(h))
                
                if 0 < i < self.resolution - 1 and 0 < j < self.resolution - 1:
                    # Normal của mặt loss cũng đi theo ý tưởng bề mặt z = f(x, y):
                    # lấy đạo hàm gần đúng theo x và y rồi dựng vector pháp tuyến.
                    dz_dx = (Z_vis[i, j+1] - Z_vis[i, j-1]) / (2 * dx)
                    dz_dy = (Z_vis[i+1, j] - Z_vis[i-1, j]) / (2 * dy)
                    
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
        self.contour_fill_vertices = self.surface_vertices.copy()
        self.contour_fill_vertices[:, 2] = self.contour_fill_z
        self.contour_fill_colors = np.array(colors, dtype=np.float32)
        self.contour_fill_indices = self.surface_indices.copy()
        self._generate_contours()

    def _loss_color(self, value):
        # Dùng bảng màu kiểu jet/parula để gần hơn phong cách biểu đồ minh hoạ tối ưu hóa.
        h = float(np.clip(value, 0.0, 1.0))
        anchors = [
            (0.00, np.array([0.12, 0.20, 0.92], dtype=np.float32)),
            (0.25, np.array([0.03, 0.76, 0.98], dtype=np.float32)),
            (0.50, np.array([0.25, 0.86, 0.58], dtype=np.float32)),
            (0.75, np.array([0.98, 0.92, 0.12], dtype=np.float32)),
            (1.00, np.array([0.92, 0.12, 0.10], dtype=np.float32)),
        ]
        for idx in range(len(anchors) - 1):
            p0, c0 = anchors[idx]
            p1, c1 = anchors[idx + 1]
            if h <= p1:
                t = 0.0 if p1 == p0 else (h - p0) / (p1 - p0)
                return ((1.0 - t) * c0 + t * c1).tolist()
        return anchors[-1][1].tolist()

    def _transform_loss_array(self, Z):
        z_min = float(np.min(Z))
        if z_min >= 0.0:
            self.loss_display_mode = "log1p"
            return np.log1p(Z)

        self.loss_display_mode = "signed_log"
        return np.sign(Z) * np.log1p(np.abs(Z))

    def _transform_single_loss(self, raw_z):
        if self.loss_display_mode == "log1p":
            return float(np.log1p(max(raw_z, 0.0)))
        if self.loss_display_mode == "signed_log":
            return float(np.sign(raw_z) * np.log1p(abs(raw_z)))
        return float(raw_z)

    def _generate_contours(self):
        if self.z_norm_grid is None or self.x_vals is None or self.y_vals is None:
            return

        contour_vertices = []
        contour_colors = []
        levels = np.linspace(0.08, 0.92, self.contour_levels)

        for level in levels:
            gray = 0.78 - 0.32 * level
            line_color = np.array([gray, gray, gray], dtype=np.float32)
            for i in range(self.resolution - 1):
                for j in range(self.resolution - 1):
                    segments = self._march_cell(i, j, level)
                    for p0, p1 in segments:
                        contour_vertices.append([p0[0], p0[1], self.contour_z])
                        contour_vertices.append([p1[0], p1[1], self.contour_z])
                        contour_colors.append(line_color)
                        contour_colors.append(line_color)

        self.contour_vertices = np.array(contour_vertices, dtype=np.float32) if contour_vertices else np.zeros((0, 3), dtype=np.float32)
        self.contour_colors = np.array(contour_colors, dtype=np.float32) if contour_colors else np.zeros((0, 3), dtype=np.float32)

    def _march_cell(self, i, j, level):
        z = self.z_norm_grid
        x0, x1 = self.x_vals[j], self.x_vals[j + 1]
        y0, y1 = self.y_vals[i], self.y_vals[i + 1]

        p00 = np.array(self.get_draw_coords(x0, y0, self.contour_z)[:2], dtype=np.float32)
        p10 = np.array(self.get_draw_coords(x1, y0, self.contour_z)[:2], dtype=np.float32)
        p11 = np.array(self.get_draw_coords(x1, y1, self.contour_z)[:2], dtype=np.float32)
        p01 = np.array(self.get_draw_coords(x0, y1, self.contour_z)[:2], dtype=np.float32)

        v00, v10 = z[i, j], z[i, j + 1]
        v11, v01 = z[i + 1, j + 1], z[i + 1, j]
        corners = [(p00, v00), (p10, v10), (p11, v11), (p01, v01)]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        intersections = []

        for a_idx, b_idx in edges:
            pa, va = corners[a_idx]
            pb, vb = corners[b_idx]
            if (va - level) == 0 and (vb - level) == 0:
                continue
            if (va - level) * (vb - level) < 0.0 or va == level or vb == level:
                intersections.append(self._interp_contour_point(pa, pb, va, vb, level))

        if len(intersections) == 2:
            return [(intersections[0], intersections[1])]
        if len(intersections) == 4:
            return [(intersections[0], intersections[1]), (intersections[2], intersections[3])]
        return []

    @staticmethod
    def _interp_contour_point(pa, pb, va, vb, level):
        if abs(vb - va) < 1e-6:
            return 0.5 * (pa + pb)
        t = (level - va) / (vb - va)
        return pa + t * (pb - pa)
    
    def setup(self):
        # BƯỚC 6: sau khi có dữ liệu CPU, tạo VAO/VBO/EBO và shader để render trên GPU.
        self.surface_vao = VAO()
        self.surface_vao.add_vbo(0, self.surface_vertices, ncomponents=3, stride=0, offset=None)
        self.surface_vao.add_vbo(1, self.surface_colors, ncomponents=3, stride=0, offset=None)
        self.surface_vao.add_vbo(2, self.surface_normals, ncomponents=3, stride=0, offset=None)
        self.surface_vao.add_ebo(self.surface_indices)
        
        # SGD dùng shader riêng để tận dụng normal thật của mặt loss,
        # không dùng lại shader phong chung vốn lấy normal từ position.
        self.surface_shader = Shader("./shaders/sgd_surface.vert", "./shaders/sgd_surface.frag")
        self.surface_uma = UManager(self.surface_shader)
        
        self.marker_shader = Shader("./shaders/phong.vert", "./shaders/phong.frag")
        self.marker_uma = UManager(self.marker_shader)
        self.marker_lighting = LightingManager(self.marker_uma)
        
        self.trail_shader = Shader("./shaders/color_interp.vert", "./shaders/color_interp.frag")
        self.trail_uma = UManager(self.trail_shader)
        if len(self.contour_fill_vertices) > 0 and len(self.contour_fill_indices) > 0:
            self.contour_fill_vao = VAO()
            self.contour_fill_vao.add_vbo(0, self.contour_fill_vertices, ncomponents=3, stride=0, offset=None)
            self.contour_fill_vao.add_vbo(1, self.contour_fill_colors, ncomponents=3, stride=0, offset=None)
            self.contour_fill_vao.add_ebo(self.contour_fill_indices)
        if len(self.contour_vertices) > 0:
            self.contour_vao = VAO()
            self.contour_vao.add_vbo(0, self.contour_vertices, ncomponents=3, stride=0, offset=None)
            self.contour_vao.add_vbo(1, self.contour_colors, ncomponents=3, stride=0, offset=None)
    
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
            'loss_history': [float(self.loss_func.compute(initial_pos[0], initial_pos[1]))],
            'grad_history': [0.0],
            'running': False,
            'momentum_buffer': np.zeros(2, dtype=np.float32),
            'adam_m': np.zeros(2, dtype=np.float32),
            'adam_v': np.zeros(2, dtype=np.float32),
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
            'rng': np.random.default_rng(seed),
            'marker_rotation': np.eye(3, dtype=np.float32),
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
        prev_position = opt['position'].copy()
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
        opt['loss_history'].append(float(opt['loss']))
        opt['grad_history'].append(float(opt['gradient_mag']))
        self._update_marker_rotation(opt, prev_position, opt['position'])
    
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
            raw_z = self.loss_func.compute(x, y)
            vis_z = self._map_loss_to_display_height(raw_z)
            dz = (vis_z - self.surface_z_min) * self.surface_z_scale - 0.9
        else:
            dz = z
        
        return [dx, dy, dz]

    def _map_loss_to_display_height(self, raw_z):
        # Marker/trajectory phải dùng cùng phép biến đổi độ cao với mặt loss,
        # nếu không quả bóng sẽ "bay" lệch khỏi bề mặt.
        return self._transform_single_loss(raw_z)

    @staticmethod
    def _project_to_screen(point, mvp, viewport_rect):
        vec = np.array([point[0], point[1], point[2], 1.0], dtype=np.float32)
        clip = mvp @ vec
        if abs(clip[3]) < 1e-7:
            return None
        ndc = clip[:3] / clip[3]
        if ndc[2] < -1.2 or ndc[2] > 1.2:
            return None

        vx, vy, vw, vh = viewport_rect
        sx = vx + (ndc[0] * 0.5 + 0.5) * vw
        sy = vy + (1.0 - (ndc[1] * 0.5 + 0.5)) * vh
        return np.array([sx, sy], dtype=np.float32)

    def pick_hover_info(self, mouse_x, mouse_y, projection, view, viewport_rect):
        # Lấy điểm gần con trỏ nhất trên lưới loss rồi trả về x, y, z để hiển thị kiểu tooltip.
        if self.surface_vertices is None or self.z_grid is None:
            return None

        model = np.eye(4, dtype=np.float32)
        mvp = projection @ (view @ model)
        mouse = np.array([mouse_x, mouse_y], dtype=np.float32)

        step = max(1, self.resolution // 90)
        best = None
        best_dist = 18.0
        for i in range(0, self.resolution, step):
            for j in range(0, self.resolution, step):
                idx = i * self.resolution + j
                screen = self._project_to_screen(self.surface_vertices[idx], mvp, viewport_rect)
                if screen is None:
                    continue
                dist = float(np.linalg.norm(screen - mouse))
                if dist < best_dist:
                    best_dist = dist
                    best = {
                        'x': float(self.x_vals[j]),
                        'y': float(self.y_vals[i]),
                        'z': float(self.z_grid[i, j]),
                        'screen_x': float(screen[0]),
                        'screen_y': float(screen[1]),
                    }
        return best

    @staticmethod
    def _rotation_matrix(axis, angle):
        axis = np.asarray(axis, dtype=np.float32)
        norm = np.linalg.norm(axis)
        if norm < 1e-7 or abs(angle) < 1e-7:
            return np.eye(3, dtype=np.float32)
        axis = axis / norm
        x, y, z = axis
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        C = 1.0 - c
        return np.array([
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ], dtype=np.float32)

    def _surface_normal_from_loss(self, x, y):
        eps_x = max((self.x_range[1] - self.x_range[0]) / self.resolution, 1e-4)
        eps_y = max((self.y_range[1] - self.y_range[0]) / self.resolution, 1e-4)
        zc = self._map_loss_to_display_height(self.loss_func.compute(x, y))
        zx1 = self._map_loss_to_display_height(self.loss_func.compute(np.clip(x + eps_x, self.x_range[0], self.x_range[1]), y))
        zx0 = self._map_loss_to_display_height(self.loss_func.compute(np.clip(x - eps_x, self.x_range[0], self.x_range[1]), y))
        zy1 = self._map_loss_to_display_height(self.loss_func.compute(x, np.clip(y + eps_y, self.y_range[0], self.y_range[1])))
        zy0 = self._map_loss_to_display_height(self.loss_func.compute(x, np.clip(y - eps_y, self.y_range[0], self.y_range[1])))

        dz_dx = (zx1 - zx0) / (2.0 * eps_x)
        dz_dy = (zy1 - zy0) / (2.0 * eps_y)
        normal = np.array([
            -dz_dx * self.surface_x_scale / max(self.surface_z_scale, 1e-6),
            -dz_dy * self.surface_y_scale / max(self.surface_z_scale, 1e-6),
            1.0,
        ], dtype=np.float32)
        norm = np.linalg.norm(normal)
        return normal / norm if norm > 1e-7 else np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def _update_marker_rotation(self, opt, prev_pos, new_pos, radius=0.05):
        prev_draw = np.array(self.get_draw_coords(prev_pos[0], prev_pos[1]), dtype=np.float32)
        new_draw = np.array(self.get_draw_coords(new_pos[0], new_pos[1]), dtype=np.float32)
        travel = new_draw - prev_draw
        distance = float(np.linalg.norm(travel))
        if distance < 1e-6:
            return

        tangent = travel / distance
        surface_normal = self._surface_normal_from_loss(new_pos[0], new_pos[1])
        axis = np.cross(tangent, surface_normal)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-7:
            return

        angle = distance / max(radius, 1e-6)
        rot = self._rotation_matrix(axis / axis_norm, angle)
        opt['marker_rotation'] = rot @ opt['marker_rotation']
    
    def draw(self, projection, view, wireframe_mode=0, display_mode=0, cam_far=100.0, show_trajectory=True,
             show_projected_trajectory=True, show_drop_lines=True, show_contours=True, view_mode='combined',
             replay_step=None):
        # BƯỚC 10: vẽ toàn bộ cảnh SGD theo thứ tự:
        # 1. vẽ mặt loss
        # 2. vẽ marker của từng optimizer
        # 3. vẽ đường rơi xuống đáy
        # 4. vẽ quỹ đạo nếu được bật
        draw_surface = view_mode in ('surface', 'combined')
        draw_contours = show_contours and view_mode in ('contour', 'combined', 'interactive')

        if draw_surface:
            GL.glUseProgram(self.surface_shader.render_idx)
            
            modelview = view @ np.eye(4, dtype=np.float32)
            
            self.surface_uma.upload_uniform_matrix4fv(projection, 'projection', True)
            self.surface_uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

            # Ánh sáng dịu để ưu tiên colormap hơn là highlight gắt.
            loc = GL.glGetUniformLocation(self.surface_shader.render_idx, "u_light_dir")
            if loc != -1:
                GL.glUniform3f(loc, 0.30, 0.55, 0.78)
            loc = GL.glGetUniformLocation(self.surface_shader.render_idx, "u_view_dir")
            if loc != -1:
                GL.glUniform3f(loc, 0.0, 0.0, 1.0)
            loc = GL.glGetUniformLocation(self.surface_shader.render_idx, "u_ambient_strength")
            if loc != -1:
                GL.glUniform1f(loc, 0.70)
            loc = GL.glGetUniformLocation(self.surface_shader.render_idx, "u_diffuse_strength")
            if loc != -1:
                GL.glUniform1f(loc, 0.30)
            loc = GL.glGetUniformLocation(self.surface_shader.render_idx, "u_specular_strength")
            if loc != -1:
                GL.glUniform1f(loc, 0.03)
            loc = GL.glGetUniformLocation(self.surface_shader.render_idx, "u_shininess")
            if loc != -1:
                GL.glUniform1f(loc, 10.0)
            loc = GL.glGetUniformLocation(self.surface_shader.render_idx, "u_tint_mix")
            if loc != -1:
                GL.glUniform1f(loc, 0.0)
            loc = GL.glGetUniformLocation(self.surface_shader.render_idx, "u_tint_color")
            if loc != -1:
                GL.glUniform3f(loc, 0.0, 0.0, 0.0)
            
            if wireframe_mode == 1:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            elif wireframe_mode == 2:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_POINT)
            
            self.surface_vao.activate()
            GL.glDrawElements(GL.GL_TRIANGLES, len(self.surface_indices), GL.GL_UNSIGNED_INT, None)
            self.surface_vao.deactivate()

            # Vẽ thêm lưới đen mảnh trên chính mặt loss để gần phong cách textbook hơn.
            GL.glEnable(GL.GL_POLYGON_OFFSET_LINE)
            GL.glPolygonOffset(-1.0, -1.0)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            loc = GL.glGetUniformLocation(self.surface_shader.render_idx, "u_tint_mix")
            if loc != -1:
                GL.glUniform1f(loc, 1.0)
            loc = GL.glGetUniformLocation(self.surface_shader.render_idx, "u_tint_color")
            if loc != -1:
                GL.glUniform3f(loc, 0.08, 0.08, 0.08)
            self.surface_vao.activate()
            GL.glDrawElements(GL.GL_TRIANGLES, len(self.surface_indices), GL.GL_UNSIGNED_INT, None)
            self.surface_vao.deactivate()
            GL.glDisable(GL.GL_POLYGON_OFFSET_LINE)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

        if draw_contours:
            self._draw_contour_fill(projection, view)

        if draw_contours:
            self._draw_contours(projection, view)
        
        replay_step_idx = None if replay_step is None else max(int(replay_step), 0)

        for name, opt in self.optimizers.items():
            # Mỗi optimizer là một quả cầu nhỏ đang chạy trên bề mặt loss.
            full_history = opt.get('history', [])
            if len(full_history) == 0:
                continue

            if replay_step_idx is None:
                draw_history = full_history
                pos = opt['position']
            else:
                idx = min(replay_step_idx, len(full_history) - 1)
                draw_history = full_history[:idx + 1]
                pos = np.array(draw_history[-1], dtype=np.float32)

            dx, dy, dz = self.get_draw_coords(pos[0], pos[1])
            
            color = self.optimizer_colors.get(opt['type'], [1, 0, 1])
            if view_mode in ('contour', 'interactive'):
                dz = self.contour_z + 0.03
            if show_drop_lines and view_mode not in ('contour', 'interactive'):
                self._draw_drop_line(dx, dy, dz, color, projection, view)
            self._draw_sphere(dx, dy, dz, color, projection, view, wireframe_mode, display_mode, cam_far, opt)
            
            if show_trajectory and len(draw_history) >= 2:
                if view_mode not in ('contour', 'interactive'):
                    self._draw_trail(draw_history, opt['type'], projection, view)
                if show_projected_trajectory or view_mode in ('contour', 'interactive'):
                    self._draw_trail_projection(draw_history, opt['type'], projection, view)

    def _draw_contours(self, projection, view):
        if self.contour_vao is None or len(self.contour_vertices) == 0:
            return

        GL.glUseProgram(self.trail_shader.render_idx)
        modelview = view @ np.eye(4, dtype=np.float32)
        self.trail_uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.trail_uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        # Trên macOS OpenGL core profile, line width > 1.0 thường báo invalid value.
        # Giữ 1.0 để app chạy ổn định, còn độ nổi bật của contour sẽ dựa vào màu sắc.
        GL.glLineWidth(1.0)
        self.contour_vao.activate()
        GL.glDrawArrays(GL.GL_LINES, 0, len(self.contour_vertices))
        self.contour_vao.deactivate()
        GL.glLineWidth(1.0)

    def _draw_contour_fill(self, projection, view):
        if self.contour_fill_vao is None or len(self.contour_fill_indices) == 0:
            return

        GL.glUseProgram(self.trail_shader.render_idx)
        modelview = view @ np.eye(4, dtype=np.float32)
        self.trail_uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.trail_uma.upload_uniform_matrix4fv(modelview, 'modelview', True)

        # Tô nền màu ngay trên mặt phẳng contour để mode này giống heatmap hơn.
        self.contour_fill_vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLES, len(self.contour_fill_indices), GL.GL_UNSIGNED_INT, None)
        self.contour_fill_vao.deactivate()
    
    def _draw_sphere(self, x, y, z, color, projection, view, wireframe_mode=0, display_mode=0, cam_far=100.0, opt=None):
        # Marker của optimizer được vẽ như một quả cầu nhỏ để dễ nhìn trong không gian 3D.
        lat_div, long_div = 12, 12
        radius = 0.05
        rotation = np.eye(3, dtype=np.float32) if opt is None else opt.get('marker_rotation', np.eye(3, dtype=np.float32))
        
        vertices = []
        normals = []
        colors = []
        base_color = np.array(color, dtype=np.float32)
        stripe_dark = np.clip(base_color * 0.18, 0.0, 1.0)
        stripe_light = np.clip(0.35 + base_color * 0.85, 0.0, 1.0)
        
        for i in range(lat_div + 1):
            lat = np.pi * (-0.5 + i / lat_div)
            for j in range(long_div + 1):
                lon = 2 * np.pi * j / long_div
                local_n = np.array([
                    np.cos(lat) * np.cos(lon),
                    np.sin(lat),
                    np.cos(lat) * np.sin(lon)
                ], dtype=np.float32)
                rotated_n = rotation @ local_n
                nx, ny, nz = rotated_n
                
                vx = x + radius * nx
                vy = y + radius * ny
                vz = z + radius * nz
                
                vertices.append([vx, vy, vz])
                normals.append([nx, ny, nz])
                
                # Thêm các vệt tối/sáng để mắt người nhìn ra bóng đang quay chứ không bị "trượt".
                meridian = abs(local_n[0]) < 0.16
                equator = abs(local_n[1]) < 0.12
                cap = np.linalg.norm(local_n - np.array([0.0, 0.0, 1.0], dtype=np.float32)) < 0.42
                if meridian or equator:
                    colors.append(stripe_dark)
                elif cap:
                    colors.append(stripe_light)
                else:
                    colors.append(base_color)
        
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
        color = np.array(self.optimizer_colors.get(opt_type, [1.0, 0.0, 1.0]), dtype=np.float32)
        vertices = []
        colors = []
        
        for i in range(len(history) - 1):
            p1 = history[i]
            p2 = history[i + 1]
            
            x1, y1, z1 = self.get_draw_coords(p1[0], p1[1])
            x2, y2, z2 = self.get_draw_coords(p2[0], p2[1])

            dir_xy = np.array([x2 - x1, y2 - y1], dtype=np.float32)
            norm_xy = float(np.linalg.norm(dir_xy))
            if norm_xy > 1e-6:
                perp = np.array([-dir_xy[1], dir_xy[0]], dtype=np.float32) / norm_xy
            else:
                perp = np.array([0.0, 1.0], dtype=np.float32)

            pass_half = self.trail_thickness_passes // 2
            for k in range(-pass_half, pass_half + 1):
                offset_xy = perp * (k * self.trail_thickness_offset * self.trail_width_scale)
                vertices.append([x1 + offset_xy[0], y1 + offset_xy[1], z1])
                vertices.append([x2 + offset_xy[0], y2 + offset_xy[1], z2])
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
            GL.glLineWidth(1.0)
            GL.glDrawArrays(GL.GL_LINES, 0, len(vertices))

    def _draw_trail_projection(self, history, opt_type, projection, view):
        if len(history) < 2:
            return

        # Đây là bản chiếu của quỹ đạo xuống mặt đáy để người xem dễ theo dõi hướng đi trên mặt phẳng x-y.
        base_color = np.array(self.optimizer_colors.get(opt_type, [1.0, 0.0, 1.0]), dtype=np.float32)
        color = np.clip(base_color * 0.82, 0.0, 1.0)
        vertices = []
        colors = []

        for i in range(len(history) - 1):
            p1 = history[i]
            p2 = history[i + 1]

            x1, y1, _ = self.get_draw_coords(p1[0], p1[1])
            x2, y2, _ = self.get_draw_coords(p2[0], p2[1])

            dir_xy = np.array([x2 - x1, y2 - y1], dtype=np.float32)
            norm_xy = float(np.linalg.norm(dir_xy))
            if norm_xy > 1e-6:
                perp = np.array([-dir_xy[1], dir_xy[0]], dtype=np.float32) / norm_xy
            else:
                perp = np.array([0.0, 1.0], dtype=np.float32)

            pass_half = self.trail_thickness_passes // 2
            for k in range(-pass_half, pass_half + 1):
                offset_xy = perp * (k * self.trail_thickness_offset * self.trail_width_scale)
                vertices.append([x1 + offset_xy[0], y1 + offset_xy[1], self.base_plane_z])
                vertices.append([x2 + offset_xy[0], y2 + offset_xy[1], self.base_plane_z])
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
            GL.glLineWidth(1.0)
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
        opt['loss_history'] = [float(opt['loss'])]
        opt['grad_history'] = [0.0]
        opt['momentum_buffer'] = np.zeros(2, dtype=np.float32)
        opt['adam_m'] = np.zeros(2, dtype=np.float32)
        opt['adam_v'] = np.zeros(2, dtype=np.float32)
