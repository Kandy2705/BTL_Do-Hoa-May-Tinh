import numpy as np
import OpenGL.GL as GL
import importlib
import math
from libs.buffer import VAO, UManager
from libs.shader import Shader
from geometry.base_shape import BaseShape

class TransformGizmo(BaseShape):
    def __init__(self):
        super().__init__() # Initialize transform from BaseShape
        
        vert_shader_basic = "./shaders/color_interp.vert"
        frag_shader_basic = "./shaders/color_interp.frag"
        
        # --- IMPORT CONE (Cho Move Tool) ---
        cone_module = importlib.import_module("geometry.3d.cone3d")
        Cone = cone_module.Cone
        self.cone_x = Cone(vert_shader_basic, frag_shader_basic, radius=0.1, height=0.2, sectors=8)
        self.cone_y = Cone(vert_shader_basic, frag_shader_basic, radius=0.1, height=0.2, sectors=8)
        self.cone_z = Cone(vert_shader_basic, frag_shader_basic, radius=0.1, height=0.2, sectors=8)
        
        self.cone_x.setup(); self.cone_x.set_color([1.0, 0.0, 0.0]) # Red
        self.cone_y.setup(); self.cone_y.set_color([0.0, 1.0, 0.0]) # Green
        self.cone_z.setup(); self.cone_z.set_color([0.0, 0.0, 1.0]) # Blue

        # --- TẠO ĐƯỜNG KẺ ---
        self.vertices_lines = np.array([
            0.0, 0.0, 0.0,   2.0, 0.0, 0.0,
            0.0, 0.0, 0.0,   0.0, 2.0, 0.0,
            0.0, 0.0, 0.0,   0.0, 0.0, 2.0
        ], dtype=np.float32)

        self.colors_lines = np.array([
            1.0, 0.0, 0.0, 1.0,   1.0, 0.0, 0.0, 1.0,
            0.0, 1.0, 0.0, 1.0,   0.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 1.0,   0.0, 0.0, 1.0, 1.0
        ], dtype=np.float32)

        # --- IMPORT CUBE CHO SCALE TOOL ---
        cube_module = importlib.import_module("geometry.3d.cube3d")
        Cube = cube_module.Cube
        
        cube_size = 0.05
        cube_offset = 2.0
        self.cube_x_vertices = self._create_cube_vertices([cube_offset, 0, 0], cube_size)
        self.cube_x_colors = np.array([[1.0, 0.0, 0.0]] * 36, dtype=np.float32)  
        self.cube_y_vertices = self._create_cube_vertices([0, cube_offset, 0], cube_size)
        self.cube_y_colors = np.array([[0.0, 1.0, 0.0]] * 36, dtype=np.float32)
        self.cube_z_vertices = self._create_cube_vertices([0, 0, cube_offset], cube_size)
        self.cube_z_colors = np.array([[0.0, 0.0, 1.0]] * 36, dtype=np.float32)
        
        # Khởi tạo 3 cube objects
        self.cube_x = Cube(vert_shader_basic, frag_shader_basic)
        self.cube_y = Cube(vert_shader_basic, frag_shader_basic)
        self.cube_z = Cube(vert_shader_basic, frag_shader_basic)
        
        self.cube_x.setup(); self.cube_x.set_color([1.0, 0.0, 0.0]) # Red
        self.cube_y.setup(); self.cube_y.set_color([0.0, 1.0, 0.0]) # Green
        self.cube_z.setup(); self.cube_z.set_color([0.0, 0.0, 1.0]) # Blue

        # --- TẠO VÒNG XOAY CHO ROTATE TOOL ---
        self.rotation_circles = self._create_rotation_circles(radius=2.0)

        # KHỞI TẠO VAOs
        self.vao_lines = VAO()
        self.vao_lines.add_vbo(0, self.vertices_lines, ncomponents=3, stride=0, offset=None)
        self.vao_lines.add_vbo(1, self.colors_lines, ncomponents=4, stride=0, offset=None)
        
        self.shader = Shader(vert_shader_basic, frag_shader_basic)
        self.uma = UManager(self.shader)

        self.vao_rotation = VAO()
        all_circle_vertices = np.vstack([self.rotation_circles['x_vertices'], 
                                        self.rotation_circles['y_vertices'], 
                                        self.rotation_circles['z_vertices']])
        all_circle_colors = np.vstack([self.rotation_circles['x_colors'], 
                                       self.rotation_circles['y_colors'], 
                                       self.rotation_circles['z_colors']])
        self.vao_rotation.add_vbo(0, all_circle_vertices, ncomponents=3, stride=0, offset=None)
        self.vao_rotation.add_vbo(1, all_circle_colors, ncomponents=4, stride=0, offset=None)

        # Interaction states
        self.selected_axis = None
        self.drag_start_pos = None
        self.start_angle = 0.0

    def _create_cube_vertices(self, center, size):
        x, y, z = center
        s = size / 2
        vertices = np.array([
            [x-s, y-s, z+s], [x+s, y-s, z+s], [x+s, y+s, z+s], [x-s, y+s, z+s],
            [x-s, y-s, z-s], [x-s, y+s, z-s], [x+s, y+s, z-s], [x+s, y-s, z-s],
            [x-s, y+s, z-s], [x-s, y+s, z+s], [x+s, y+s, z+s], [x+s, y+s, z-s],
            [x-s, y-s, z-s], [x+s, y-s, z-s], [x+s, y-s, z+s], [x-s, y-s, z+s],
            [x+s, y-s, z-s], [x+s, y+s, z-s], [x+s, y+s, z+s], [x+s, y-s, z+s],
            [x-s, y-s, z-s], [x-s, y-s, z+s], [x-s, y+s, z+s], [x-s, y+s, z-s]
        ], dtype=np.float32)
        indices = np.array([
            0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11,
            12, 13, 14, 12, 14, 15, 16, 17, 18, 16, 18, 19, 20, 21, 22, 20, 22, 23
        ], dtype=np.uint32)
        return vertices[indices]

    def _create_rotation_circles(self, radius=2.0, segments=64):
        circles = {}
        theta = np.linspace(0, 2*np.pi, segments, endpoint=False)
        
        x_v, y_v, z_v = [], [], []
        x_c, y_c, z_c = [], [], []
        
        for i in range(segments):
            y = radius * np.cos(theta[i])
            z = radius * np.sin(theta[i])
            x_v.append([0.0, y, z])
            x_c.append([1.0, 0.0, 0.0, 1.0])
            
            x = radius * np.cos(theta[i])
            z = radius * np.sin(theta[i])
            y_v.append([x, 0.0, z])
            y_c.append([0.0, 1.0, 0.0, 1.0])
            
            x = radius * np.cos(theta[i])
            y = radius * np.sin(theta[i])
            z_v.append([x, y, 0.0])
            z_c.append([0.0, 0.0, 1.0, 1.0])
            
        circles['x_vertices'] = np.array(x_v, dtype=np.float32)
        circles['x_colors'] = np.array(x_c, dtype=np.float32)
        circles['y_vertices'] = np.array(y_v, dtype=np.float32)
        circles['y_colors'] = np.array(y_c, dtype=np.float32)
        circles['z_vertices'] = np.array(z_v, dtype=np.float32)
        circles['z_colors'] = np.array(z_c, dtype=np.float32)
        return circles

    def project_to_screen(self, point_3d, view_matrix, proj_matrix, win_size):
        p = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0], dtype=np.float32)
        p = view_matrix @ p
        p = proj_matrix @ p
        if p[3] != 0: p = p / p[3] 
        screen_x = (p[0] + 1.0) * 0.5 * win_size[0]
        screen_y = (1.0 - p[1]) * 0.5 * win_size[1] 
        return np.array([screen_x, screen_y])

    def check_axis_selection(self, mouse_pos, view, projection, win_size, target_pos, current_tool):
        mouse = np.array(mouse_pos)
        def dist_to_segment(p, v, w):
            l2 = np.sum((v - w)**2)
            if l2 == 0: return np.linalg.norm(p - v)
            t = max(0, min(1, np.dot(p - v, w - v) / l2))
            proj = v + t * (w - v)
            return np.linalg.norm(p - proj)

        # --- KIỂM TRA CHUỘT CHO ROTATE TOOL ---
        if current_tool == 'rotate':
            min_dist = float('inf')
            selected = None
            
            def check_ring(ring_v, axis_name):
                nonlocal min_dist, selected
                for i in range(len(ring_v)):
                    p1_3d = np.array(target_pos) + ring_v[i]
                    p2_3d = np.array(target_pos) + ring_v[(i+1) % len(ring_v)]
                    
                    p1_2d = self.project_to_screen(p1_3d, view, projection, win_size)
                    p2_2d = self.project_to_screen(p2_3d, view, projection, win_size)
                    
                    d = dist_to_segment(mouse, p1_2d, p2_2d)
                    if d < min_dist:
                        min_dist = d
                        selected = axis_name

            check_ring(self.rotation_circles['x_vertices'], 'x')
            check_ring(self.rotation_circles['y_vertices'], 'y')
            check_ring(self.rotation_circles['z_vertices'], 'z')
            
            if min_dist < 15.0: return selected
            return None

        # --- KIỂM TRA CHUỘT CHO MOVE / SCALE TOOL ---
        else:
            origin_2d = self.project_to_screen(target_pos, view, projection, win_size)
            x_tip = self.project_to_screen([target_pos[0] + 2.0, target_pos[1], target_pos[2]], view, projection, win_size)
            y_tip = self.project_to_screen([target_pos[0], target_pos[1] + 2.0, target_pos[2]], view, projection, win_size)
            z_tip = self.project_to_screen([target_pos[0], target_pos[1], target_pos[2] + 2.0], view, projection, win_size)
            
            dist_x = dist_to_segment(mouse, origin_2d, x_tip)
            dist_y = dist_to_segment(mouse, origin_2d, y_tip)
            dist_z = dist_to_segment(mouse, origin_2d, z_tip)
            
            min_dist = min(dist_x, dist_y, dist_z)
            if min_dist < 15.0:
                if min_dist == dist_x: return 'x'
                elif min_dist == dist_y: return 'y'
                else: return 'z'
            return None

    def handle_mouse_press(self, mouse_pos, target_pos, current_tool, view, proj, win_size):
        if current_tool in ['move', 'rotate', 'scale']:
            self.selected_axis = self.check_axis_selection(mouse_pos, view, proj, win_size, target_pos, current_tool)
            
            if self.selected_axis:
                self.drag_start_pos = mouse_pos
                # Tính góc xoay ban đầu nếu là Rotate Tool
                if current_tool == 'rotate':
                    origin_2d = self.project_to_screen(target_pos, view, proj, win_size)
                    v = np.array(mouse_pos) - origin_2d
                    self.start_angle = math.atan2(v[1], v[0])
                    
                print(f"[GIZMO] Selected axis: {self.selected_axis} for {current_tool}")

    def handle_mouse_drag(self, mouse_pos, target_obj, current_tool, view, proj, win_size):
        if self.selected_axis and self.drag_start_pos:
            
            # --- LOGIC KÉO CHUỘT CHO ROTATE TOOL (Dùng Góc) ---
            if current_tool == 'rotate':
                origin_2d = self.project_to_screen(target_obj.position, view, proj, win_size)
                v = np.array(mouse_pos) - origin_2d
                
                # Tính góc xoay hiện tại của chuột
                curr_angle = math.atan2(v[1], v[0])
                delta_rad = curr_angle - self.start_angle
                
                # Fix lỗi nhảy số khi vượt qua góc Pi/-Pi
                if delta_rad > math.pi: delta_rad -= 2*math.pi
                elif delta_rad < -math.pi: delta_rad += 2*math.pi
                
                delta_deg = math.degrees(delta_rad)
                
                # Cộng dồn Rotation. Tùy trục mà lật dấu cho thuận mắt
                if self.selected_axis == 'x': target_obj.rotation[0] -= delta_deg
                elif self.selected_axis == 'y': target_obj.rotation[1] += delta_deg 
                elif self.selected_axis == 'z': target_obj.rotation[2] -= delta_deg
                
                self.start_angle = curr_angle
                self.drag_start_pos = mouse_pos

            # --- LOGIC KÉO CHUỘT CHO MOVE / SCALE TOOL (Dùng Khoảng Cách) ---
            else:
                target_pos = target_obj.position
                origin_2d = self.project_to_screen(target_pos, view, proj, win_size)
                
                if self.selected_axis == 'x': tip_3d = [target_pos[0] + 1.0, target_pos[1], target_pos[2]]
                elif self.selected_axis == 'y': tip_3d = [target_pos[0], target_pos[1] + 1.0, target_pos[2]]
                else: tip_3d = [target_pos[0], target_pos[1], target_pos[2] + 1.0]
                    
                tip_2d = self.project_to_screen(tip_3d, view, proj, win_size)
                axis_dir_2d = tip_2d - origin_2d
                length = np.linalg.norm(axis_dir_2d)
                if length > 0: axis_dir_2d = axis_dir_2d / length
                
                mouse_delta = np.array(mouse_pos) - np.array(self.drag_start_pos)
                move_amount = np.dot(mouse_delta, axis_dir_2d)
                speed = 0.02 
                
                if current_tool == 'move':
                    if self.selected_axis == 'x': target_obj.position[0] += move_amount * speed
                    elif self.selected_axis == 'y': target_obj.position[1] += move_amount * speed
                    elif self.selected_axis == 'z': target_obj.position[2] += move_amount * speed
                elif current_tool == 'scale':
                    scale_factor = 1.0 + move_amount * speed * 0.1
                    if self.selected_axis == 'x': target_obj.scale[0] *= scale_factor
                    elif self.selected_axis == 'y': target_obj.scale[1] *= scale_factor
                    elif self.selected_axis == 'z': target_obj.scale[2] *= scale_factor
                
                self.drag_start_pos = mouse_pos

    def handle_mouse_release(self):
        self.selected_axis = None
        self.drag_start_pos = None

    def draw(self, projection, view, position, current_tool="move"):
        GL.glUseProgram(self.shader.render_idx)
        gizmo_transform = self.get_transform_matrix()
        T_base = np.identity(4, dtype=np.float32)
        T_base[0,3], T_base[1,3], T_base[2,3] = position
        
        final_model = gizmo_transform @ T_base
        modelview = view @ final_model
        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        
        GL.glDisable(GL.GL_DEPTH_TEST) 
        GL.glLineWidth(1.0) 

        # ======= NẾU LÀ ROTATE TOOL -> VẼ 3 VÒNG TRÒN =======
        if current_tool == 'rotate':
            self.vao_rotation.activate()
            # Vẽ 3 vòng (64 segments mỗi vòng) bằng LINE_LOOP
            GL.glDrawArrays(GL.GL_LINE_LOOP, 0, 64)   # Vòng X
            GL.glDrawArrays(GL.GL_LINE_LOOP, 64, 64)  # Vòng Y
            GL.glDrawArrays(GL.GL_LINE_LOOP, 128, 64) # Vòng Z
            
        # ======= NẾU LÀ MOVE/SCALE -> VẼ ĐƯỜNG KẺ + NÓN/CUBE =======
        else:
            self.vao_lines.activate()
            GL.glDrawArrays(GL.GL_LINES, 0, 6) 

            if current_tool == 'move':
                T_y = np.identity(4, dtype=np.float32); T_y[1,3] = 2.0
                T_x = np.identity(4, dtype=np.float32); T_x[0,3] = 2.0
                T_z = np.identity(4, dtype=np.float32); T_z[2,3] = 2.0
                
                self.cone_y.draw(projection, view, gizmo_transform @ T_base @ T_y)
                R_x = np.array([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
                self.cone_x.draw(projection, view, gizmo_transform @ T_base @ T_x @ R_x)
                R_z = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]], dtype=np.float32)
                self.cone_z.draw(projection, view, gizmo_transform @ T_base @ T_z @ R_z)
                
            elif current_tool == 'scale':
                T_y = np.identity(4, dtype=np.float32); T_y[1,3] = 2.0
                T_x = np.identity(4, dtype=np.float32); T_x[0,3] = 2.0
                T_z = np.identity(4, dtype=np.float32); T_z[2,3] = 2.0
                
                S_cube = np.diag([0.1, 0.1, 0.1, 1.0]).astype(np.float32)
                self.cube_y.draw(projection, view, gizmo_transform @ T_base @ T_y @ S_cube)
                self.cube_x.draw(projection, view, gizmo_transform @ T_base @ T_x @ S_cube)
                self.cube_z.draw(projection, view, gizmo_transform @ T_base @ T_z @ S_cube)

        GL.glEnable(GL.GL_DEPTH_TEST)

    def cleanup(self):
        if hasattr(self, 'vao_lines'): self.vao_lines.delete()
        if hasattr(self, 'vao_rotation'): self.vao_rotation.delete()
        if hasattr(self, 'shader'): self.shader.delete()
        self.cone_x.cleanup(); self.cone_y.cleanup(); self.cone_z.cleanup()
        self.cube_x.cleanup(); self.cube_y.cleanup(); self.cube_z.cleanup()