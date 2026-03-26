import numpy as np
import OpenGL.GL as GL
import importlib
from libs.buffer import VAO, UManager
from libs.shader import Shader
from geometry.base_shape import BaseShape

class TransformGizmo(BaseShape):
    def __init__(self):
        super().__init__() # Initialize transform from BaseShape
        
        # Đường dẫn shader cơ bản để vẽ màu đơn sắc
        vert_shader_basic = "./shaders/color_interp.vert"
        frag_shader_basic = "./shaders/color_interp.frag"
        
        # --- THAY ĐỔI 1: TÁI SỬ DỤNG LẠI CLASS CONE3D CỦA BẠN ---
        # Dùng importlib để lách luật thư mục tên '3d' bắt đầu bằng số
        cone_module = importlib.import_module("geometry.3d.cone3d")
        Cone = cone_module.Cone
        
        # Khởi tạo 3 khối Cone riêng biệt (vô hiệu hóa ánh sáng 'lighting_enabled=False')
        # để nó vẽ ra màu đơn sắc sáng rõ
        self.cone_x = Cone(vert_shader_basic, frag_shader_basic, lighting_enabled=False)
        self.cone_y = Cone(vert_shader_basic, frag_shader_basic, lighting_enabled=False)
        self.cone_z = Cone(vert_shader_basic, frag_shader_basic, lighting_enabled=False)
        
        # Gán Setup cho 3 khối nón
        self.cone_x.setup()
        self.cone_y.setup()
        self.cone_z.setup()
        
        # Gán MÀU ĐƠN SẮC RGB cho 3 khối nón (Dùng hàm set_solid_color vừa thêm)
        self.cone_x.set_solid_color([1.0, 0.0, 0.0]) # Red
        self.cone_y.set_solid_color([0.0, 1.0, 0.0]) # Green
        self.cone_z.set_solid_color([0.0, 0.0, 1.0]) # Blue

        # --- THAY ĐỔI 2: CHỈ CÒN TẠO HÌNH HỌC CHO 3 ĐƯỜNG KẺ (SHAFTS) ---
        # Dữ liệu 3 đường thẳng
        self.vertices_lines = np.array([
            0.0, 0.0, 0.0,   2.0, 0.0, 0.0,  # X axis
            0.0, 0.0, 0.0,   0.0, 2.0, 0.0,  # Y axis
            0.0, 0.0, 0.0,   0.0, 0.0, 2.0   # Z axis
        ], dtype=np.float32)

        # Dữ liệu màu sắc cho 3 đường thẳng (dạng RGBA)
        self.colors_lines = np.array([
            1.0, 0.0, 0.0, 1.0,   1.0, 0.0, 0.0, 1.0, # Red
            0.0, 1.0, 0.0, 1.0,   0.0, 1.0, 0.0, 1.0, # Green
            0.0, 0.0, 1.0, 1.0,   0.0, 0.0, 1.0, 1.0  # Blue
        ], dtype=np.float32)

        self.vao_lines = VAO()
        # VAO 0: Vertices
        self.vao_lines.add_vbo(0, self.vertices_lines, ncomponents=3, stride=0, offset=None)
        # VAO 1: Colors (4 thành phần RGBA)
        self.vao_lines.add_vbo(1, self.colors_lines, ncomponents=4, stride=0, offset=None)
        
        # Dùng lại shader cơ bản của bạn
        self.shader = Shader(vert_shader_basic, frag_shader_basic)
        self.uma = UManager(self.shader)

        # Interaction state
        self.selected_axis = None  # 'x', 'y', 'z', None
        self.drag_start_pos = None
        self.drag_start_value = None

    def project_to_screen(self, point_3d, view_matrix, proj_matrix, win_size):
        """Hàm chiếu 1 điểm từ thế giới 3D lên tọa độ Pixel 2D trên màn hình"""
        p = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0], dtype=np.float32)
        p = view_matrix @ p
        p = proj_matrix @ p
        if p[3] != 0:
            p = p / p[3] # Đưa về [-1, 1]
            
        # Chuyển sang Pixel: O(0,0) nằm ở góc trên bên trái
        screen_x = (p[0] + 1.0) * 0.5 * win_size[0]
        screen_y = (1.0 - p[1]) * 0.5 * win_size[1] 
        return np.array([screen_x, screen_y])

    def check_axis_selection(self, mouse_pos, view, projection, win_size, target_pos):
        # 1. Tìm điểm tâm 2D trên màn hình
        origin_2d = self.project_to_screen(target_pos, view, projection, win_size)
        
        # 2. Tìm điểm ngọn của 3 mũi tên 2D trên màn hình
        x_tip = self.project_to_screen([target_pos[0] + 2.0, target_pos[1], target_pos[2]], view, projection, win_size)
        y_tip = self.project_to_screen([target_pos[0], target_pos[1] + 2.0, target_pos[2]], view, projection, win_size)
        z_tip = self.project_to_screen([target_pos[0], target_pos[1], target_pos[2] + 2.0], view, projection, win_size)
        
        mouse = np.array(mouse_pos)
        
        # Hàm tính khoảng cách từ điểm (chuột) tới đoạn thẳng (mũi tên)
        def dist_to_segment(p, v, w):
            l2 = np.sum((v - w)**2)
            if l2 == 0: return np.linalg.norm(p - v)
            t = max(0, min(1, np.dot(p - v, w - v) / l2))
            proj = v + t * (w - v)
            return np.linalg.norm(p - proj)

        dist_x = dist_to_segment(mouse, origin_2d, x_tip)
        dist_y = dist_to_segment(mouse, origin_2d, y_tip)
        dist_z = dist_to_segment(mouse, origin_2d, z_tip)
        
        min_dist = min(dist_x, dist_y, dist_z)
        
        # Trúng nếu cách đường kẻ dưới 15 pixels
        if min_dist < 15.0:
            if min_dist == dist_x: return 'x'
            elif min_dist == dist_y: return 'y'
            else: return 'z'
        return None

    def handle_mouse_press(self, mouse_pos, target_pos, current_tool, view, proj, win_size):
        if current_tool in ['move', 'rotate', 'scale']:
            self.selected_axis = self.check_axis_selection(mouse_pos, view, proj, win_size, target_pos)
            if self.selected_axis:
                self.drag_start_pos = mouse_pos
                self.drag_start_value = {'x': target_pos[0], 'y': target_pos[1], 'z': target_pos[2]}
                if current_tool == 'scale':
                    self.drag_start_value = {'x': target_pos[0], 'y': target_pos[1], 'z': target_pos[2]}
                print(f"[GIZMO] Selected axis: {self.selected_axis}")

    def handle_mouse_drag(self, mouse_pos, target_pos, current_tool, view, proj, win_size):
        if self.selected_axis and self.drag_start_pos:
            # 1. Chiếu trục đang chọn lên màn hình 2D
            origin_2d = self.project_to_screen(target_pos, view, proj, win_size)
            if self.selected_axis == 'x': tip_3d = [target_pos[0] + 1.0, target_pos[1], target_pos[2]]
            elif self.selected_axis == 'y': tip_3d = [target_pos[0], target_pos[1] + 1.0, target_pos[2]]
            else: tip_3d = [target_pos[0], target_pos[1], target_pos[2] + 1.0]
                
            tip_2d = self.project_to_screen(tip_3d, view, proj, win_size)
            
            # 2. Vector hướng của mũi tên trên màn hình
            axis_dir_2d = tip_2d - origin_2d
            length = np.linalg.norm(axis_dir_2d)
            if length > 0: axis_dir_2d = axis_dir_2d / length
            
            # 3. Vector rê chuột
            mouse_delta = np.array(mouse_pos) - np.array(self.drag_start_pos)
            
            # 4. Tích vô hướng (Rê chuột đúng hướng mũi tên thì đi nhanh, lệch hướng thì đi chậm/không đi)
            move_amount = np.dot(mouse_delta, axis_dir_2d)
            speed = 0.02 # Độ nhạy
            
            if current_tool == 'move':
                if self.selected_axis == 'x': target_pos[0] += move_amount * speed
                elif self.selected_axis == 'y': target_pos[1] += move_amount * speed
                elif self.selected_axis == 'z': target_pos[2] += move_amount * speed
            
            # Cập nhật vị trí bắt đầu để kéo mượt
            self.drag_start_pos = mouse_pos

    def handle_mouse_release(self):
        """Handle mouse release"""
        self.selected_axis = None
        self.drag_start_pos = None
        self.drag_start_value = None

    def draw(self, projection, view, position):
        GL.glUseProgram(self.shader.render_idx)
        
        # --- BƯỚC 1: VẼ 3 ĐƯỜNG THẲNG XYZ (LINE 1.0) ---
        # Lấy ma trận Transform của Gizmo từ BaseShape
        gizmo_transform = self.get_transform_matrix()
        # Dịch chuyển tới vị trí của Object
        T_base = np.identity(4, dtype=np.float32)
        T_base[0,3], T_base[1,3], T_base[2,3] = position
        
        # Kết hợp model matrix cho đường kẻ
        final_lines_model = gizmo_transform @ T_base
        modelview_lines = view @ final_lines_model
        
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview_lines, 'modelview', True)
        
        GL.glDisable(GL.GL_DEPTH_TEST) # Gizmo luôn hiện trên cùng
        GL.glLineWidth(1.0) # An toàn cho macOS

        self.vao_lines.activate()
        GL.glDrawArrays(GL.GL_LINES, 0, 6) # Vẽ 3 đường thẳng

        # --- BƯỚC 2: TÁI SỬ DỤNG HÀM DRAW CỦA CÁC KHỐI NÓN CỦA BẠN ---
        # Cần tính toán Ma trận Model cho từng khối nón để nó gắn vào cuối đường kẻ

        # 1. Mũi tên xanh lá (Y) - Gắn vào cuối đường kẻ Y (độ dài 2.0)
        # Hướng Y chuẩn, không cần xoay
        T_y = np.identity(4, dtype=np.float32)
        T_y[0,3], T_y[1,3], T_y[2,3] = 0.0, 2.0, 0.0 # Tọa độ tương đối so với tâm Gizmo
        # Kết hợp model matrix
        final_cone_y_model = gizmo_transform @ T_base @ T_y
        self.cone_y.draw(projection, view, final_cone_y_model)

        # 2. Mũi tên đỏ (X) - Gắn vào cuối đường kẻ X (độ dài 2.0)
        # Cần xoay sang phải (-90 độ quanh trục Z)
        T_x = np.identity(4, dtype=np.float32)
        T_x[0,3], T_x[1,3], T_x[2,3] = 2.0, 0.0, 0.0 # Tọa độ tương đối
        R_x = np.array([
            [ 0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [ 0.0, 0.0, 1.0, 0.0],
            [ 0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        final_cone_x_model = gizmo_transform @ T_base @ T_x @ R_x
        self.cone_x.draw(projection, view, final_cone_x_model)

        # 3. Mũi tên xanh dương (Z) - Gắn vào cuối đường kẻ Z (độ dài 2.0)
        # Cần xoay ra trước (+90 độ quanh trục X) và xoay 180 độ để đúng chiều
        T_z = np.identity(4, dtype=np.float32)
        T_z[0,3], T_z[1,3], T_z[2,3] = 0.0, 0.0, 2.0 # Tọa độ tương đối
        R_z = np.array([
            [ 1.0,  0.0,  0.0, 0.0],
            [ 0.0,  0.0, -1.0, 0.0], # cos(-90)=0, -sin(-90)=1 (Sửa dấu ở đây)
            [ 0.0,  1.0,  0.0, 0.0], # sin(-90)=-1, cos(-90)=0 (Sửa dấu ở đây)
            [ 0.0,  0.0,  0.0, 1.0]
        ], dtype=np.float32)
        final_cone_z_model = gizmo_transform @ T_base @ T_z @ R_z
        self.cone_z.draw(projection, view, final_cone_z_model)

        # Trả lại trạng thái depth cũ
        GL.glEnable(GL.GL_DEPTH_TEST)

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'vao_lines'):
            self.vao_lines.delete()
        if hasattr(self, 'shader'):
            self.shader.delete()
        
        # Cleanup các khối nón
        self.cone_x.cleanup()
        self.cone_y.cleanup()
        self.cone_z.cleanup()