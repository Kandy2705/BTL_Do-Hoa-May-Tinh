# TÀI LIỆU BÁO CÁO BÀI TẬP LỚN
## Môn: Đồ họa Máy tính
### Đề tài: Mini Game Engine với OpenGL và Python

---

## Mục lục
1. [Bức tranh toàn cảnh (Architecture Overview)](#1-bức-tranh-toàn-cảnh-architecture-overview)
2. [OpenGL Pipeline](#2-opengl-pipeline)
3. [Chi tiết Phần 1 - Game Engine 3D](#3-chi-tiết-phần-1---game-engine-3d)
4. [Chi tiết Phần 2 - SGD Visualization](#4-chi-tiết-phần-2---sgd-visualization)
5. [Bộ câu hỏi phòng thủ](#5-bộ-câu-hỏi-phòng-thủ-hội-đồng)

---

## 1. Bức tranh toàn cảnh (Architecture Overview)

### 1.1 Kiến trúc MVC

Engine được xây dựng theo mô hình **MVC (Model-View-Controller)**:

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py                                   │
│                     (Entry Point)                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AppController                                  │
│                 (controller.py)                                   │
│  - Xử lý input từ bàn phím/chuột                               │
│  - Điều phối Model và View                                       │
│  - Main Loop                                                     │
└─────────────────────────────────────────────────────────────────┘
         │                                    │
         ▼                                    ▼
┌─────────────────────┐          ┌─────────────────────┐
│    AppModel          │          │      Viewer         │
│   (model.py)         │          │   (viewer.py)       │
│  - Quản lý dữ liệu  │          │  - Render graphics  │
│  - Scene graph       │          │  - GLFW window     │
│  - Game objects      │          │  - ImGui UI        │
└─────────────────────┘          └─────────────────────┘
```

### 1.2 Luồng Main Loop

Main Loop là trái tim của engine, chạy 60 FPS:

```python
# controller.py - Dòng 542-654
def run(self) -> None:
    while not self.view.should_close():
        self.view.poll_events()          # 1. Xử lý sự kiện GLFW
        self.view.begin_frame()          # 2. Bắt đầu frame ImGui
        
        ui_actions = self.view.draw_ui() # 3. Vẽ UI, nhận actions
        self._process_ui_actions(...)    # 4. Xử lý UI actions
        
        # 5. SGD Simulation Step (nếu đang ở mode SGD)
        if self.model.selected_category == 4:
            if self.model.sgd_simulation_running:
                for _ in range(self.model.sgd_simulation_speed):
                    self.model.sgd_step()
        
        # 6. Tính ma trận View/Projection
        view = self.view.trackball().view_matrix()
        projection = self.view.trackball().projection_matrix(...)
        
        # 7. Render
        self.view.draw_coordinate_system(...)
        self.view.draw_drawables(...)
        
        self.view.end_frame()             # 8. Kết thúc frame
```

### 1.3 Cấu trúc thư mục

```
BTL_DHMT/
├── main.py                    # Entry point
├── controller.py             # Điều phối chính
├── model.py                  # Dữ liệu và Scene
├── viewer.py                 # Render và UI
│
├── components/               # UI Components (ImGui)
│   ├── main_menu.py
│   ├── hierarchy_panel.py
│   ├── inspector_panel.py
│   └── sgd_panel.py
│
├── geometry/                 # Hình học
│   ├── base_shape.py        # Base class cho mọi shape
│   ├── 2d/                  # Hình 2D (triangle, circle, ...)
│   ├── 3d/                  # Hình 3D (cube, sphere, ...)
│   ├── math_surface3d.py    # Surface z = f(x,y)
│   ├── model_loader3d.py   # Load .obj/.ply
│   └── sgd_visualizer.py   # SGD Visualization
│
├── libs/                     # Thư viện
│   ├── transform.py         # Ma trận 4x4, Quaternion, Trackball
│   ├── shader.py           # Shader compilation
│   ├── buffer.py           # VAO, VBO, EBO
│   ├── lighting.py         # Lighting Manager
│   ├── gizmo.py            # Transform Gizmo
│   ├── loss_functions.py   # Loss functions cho SGD
│   └── coordinate_system.py # Hệ trục tọa độ
│
├── core/
│   └── GameObject.py       # Base GameObject class
│
└── shaders/                 # GLSL Shaders
    ├── phong.vert / .frag
    ├── gouraud.vert / .frag
    └── color_interp.vert / .frag
```

---

## 2. OpenGL Pipeline

### 2.1 Sơ đồ tổng quát

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Vertex Data  │────▶│  Vertex      │────▶│  Primitive   │────▶│ Rasterization │
│ (Attributes) │     │  Shader      │     │  Assembly    │     │              │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                        │
                                                                        ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Output      │◀────│ Per-Fragment│◀────│  Fragment    │◀────│   Output     │
│   Merging     │     │  Operations  │     │  Shader      │     │   Merger     │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

### 2.2 Bước 1: Vertex Data (Dữ liệu đỉnh)

#### Lý thuyết:
Dữ liệu đỉnh bao gồm: vị trí (position), màu (color), pháp tuyến (normal), UV coordinates. Được lưu trong VBO (Vertex Buffer Object).

#### Code trong buffer.py:

```python
# buffer.py - Dòng 19-29
def add_vbo(self, location, data,
           ncomponents=3, dtype=GL.GL_FLOAT, normalized=False, stride=0, offset=None):
    self.activate() # VAO
    buffer_idx = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, buffer_idx)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, data, GL.GL_STATIC_DRAW)  # Nạp dữ liệu vào GPU
    GL.glVertexAttribPointer(location, ncomponents, dtype, normalized, stride, offset)
    GL.glEnableVertexAttribArray(location)
    self.vbo[location] = buffer_idx
    self.deactivate()
```

#### Ví dụ thực tế - Tạo hình lập phương:

```python
# geometry/3d/cube3d.py
vertices = np.array([
    # Mỗi đỉnh có: x, y, z, r, g, b (6 thuộc tính)
    -1,-1,-1,  1,0,0,  # đỉnh 0 + màu đỏ
    -1, 1,-1,  0,1,0,  # đỉnh 1 + màu xanh lá
    ...
], dtype=np.float32)

# Nạp vào VAO tại location 0 (position) và 1 (color)
cube_vao.add_vbo(0, vertices[:, :3], ncomponents=3)  # Vị trí
cube_vao.add_vbo(1, vertices[:, 3:], ncomponents=3)  # Màu
```

### 2.3 Bước 2: Vertex Shader (Phép biến đổi đỉnh)

#### Lý thuyết:
Vertex shader nhận từng đỉnh, áp dụng các phép biến đổi ma trận MVP (Model-View-Projection) để chuyển từ tọa độ local sang clip coordinates.

**Ma trận MVP = Projection × View × Model**

#### Ma trận biến đổi trong transform.py:

```python
# transform.py - Dòng 74-78: Ma trận Translate
def translate(x=0.0, y=0.0, z=0.0):
    matrix = np.identity(4, 'f')
    matrix[:3, 3] = vec(x, y, z)
    return matrix

# transform.py - Dòng 81-85: Ma trận Scale
def scale(x, y=None, z=None):
    x, y, z = (x, y, z) if isinstance(x, Number) else (x[0], x[1], x[2])
    y, z = (x, x) if y is None or z is None else (y, z)
    return np.diag((x, y, z, 1))

# transform.py - Dòng 94-102: Ma trận Rotate
def rotate(axis=(1., 0., 0.), angle=0.0, radians=None):
    x, y, z = normalized(vec(axis))
    s, c = sincos(angle, radians)
    nc = 1 - c
    return np.array([[x*x*nc + c,   x*y*nc - z*s, x*z*nc + y*s, 0],
                     [y*x*nc + z*s, y*y*nc + c,   y*z*nc - x*s, 0],
                     [x*z*nc - y*s, y*z*nc + x*s, z*z*nc + c,   0],
                     [0,            0,            0,            1]], 'f')
```

#### Ma trận View (Camera) trong transform.py:

```python
# transform.py - Dòng 105-114
def lookat(eye, target, up):
    """ Computes 4x4 view matrix from 3d point 'eye' to 'target' """
    view = normalized(vec(target)[:3] - vec(eye)[:3])  # Z axis (forward)
    up = normalized(vec(up)[:3])                        # Y axis (up)
    right = np.cross(view, up)                          # X axis (right)
    up = np.cross(right, view)                        # Recompute up
    rotation = np.identity(4)
    rotation[:3, :3] = np.vstack([right, up, -view])  # Rotation matrix
    return rotation @ translate(-eye)                  # Rotate then translate
```

#### Ma trận Projection (Perspective):

```python
# transform.py - Dòng 48-57
def perspective(fovy, aspect, near, far):
    _scale = 1.0/math.tan(math.radians(fovy)/2.0)
    sx, sy = _scale / aspect, _scale
    zz = (far + near) / (near - far)
    zw = 2 * far * near/(near - far)
    return np.array([[sx, 0,  0,  0],
                     [0,  sy, 0,  0],
                     [0,  0, zz, zw],
                     [0,  0, -1,  0]], 'f')
```

#### Vertex Shader trong phong.vert:

```glsl
# shaders/phong.vert
uniform mat4 projection, modelview;  // Ma trận từ CPU

void main(){
    // Tính vị trí vertex trong không gian view
    vec4 vertPos4 = modelview * vec4(position, 1.0);
    vertPos = vec3(vertPos4) / vertPos4.w;
    
    // Tính pháp tuyến (cho lighting)
    mat4 normal_matrix = transpose(inverse(modelview));
    normal_interp = vec3(normal_matrix * vec4(normalize(position), 0.0));
    
    // Output clip coordinates
    gl_Position = projection * vertPos4;
}
```

### 2.4 Bước 3: Primitive Assembly & Rasterization

#### Lý thuyết:
Sau khi vertex shader xử lý xong, OpenGL tự động:
1. **Assembly**: Nối các đỉnh thành primitive (tam giác, đường, điểm)
2. **Clipping**: Cắt primitive nằm ngoài view frustum
3. **Rasterization**: Chuyển primitive thành các fragment (pixel)

```python
# Ví dụ: Tạo chỉ số cho lập phương (6 mặt × 2 tam giác × 3 đỉnh = 36 indices)
indices = np.array([
    0, 1, 2, 0, 2, 3,    # Mặt trước
    4, 5, 6, 4, 6, 7,    # Mặt sau
    ...
], dtype=np.uint32)

# buffer.py - Dòng 31-36: Nạp EBO
def add_ebo(self, indices):
    self.activate()
    self.ebo = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
    GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices, GL.GL_STATIC_DRAW)
    self.deactivate()
```

### 2.5 Bước 4: Fragment Shader (Tính màu pixel)

#### Lý thuyết:
Fragment shader nhận dữ liệu đã được nội suy (interpolate) từ vertex shader, tính toán màu cuối cùng cho mỗi pixel.

**So sánh Gouraud vs Phong Shading:**

| Đặc điểm | Gouraud | Phong |
|-----------|---------|-------|
| Lighting tính ở đâu | Vertex Shader | Fragment Shader |
| Chất lượng | Thấp hơn | Cao hơn |
| Hiệu năng | Nhanh hơn | Chậm hơn |
| Specular highlight | Có thể bị miss | Luôn chính xác |

#### Phong Shading (shaders/phong.frag):

```glsl
# shaders/phong.frag
precision mediump float;

in vec3 normal_interp;  // Pháp tuyến đã nội suy
in vec3 vertPos;        // Vị trí đã nội suy
in vec3 colorInterp;    // Màu đỉnh đã nội suy

uniform mat3 K_materials;   // [diffuse, specular, ambient] colors
uniform mat3 I_light;       // [diffuse, specular, ambient] intensities
uniform vec3 light_pos;     // Vị trí nguồn sáng
uniform float shininess;    // Độ bóng

void main() {
    vec3 N = normalize(normal_interp);        // Pháp tuyến
    vec3 L = normalize(light_pos - vertPos); // Vector tới ánh sáng
    vec3 R = reflect(-L, N);                 // Vector phản xạ
    vec3 V = normalize(-vertPos);             // Vector tới camera
    
    // ====== PHONG LIGHTING MODEL ======
    
    // 1. Diffuse: Cường độ phụ thuộc góc giữa L và N
    float NdotL = max(dot(L, N), 0.0);
    
    // 2. Specular: Độ bóng phản xạ
    float specAngle = max(dot(R, V), 0.0);
    float specular = pow(specAngle, shininess);
    
    // 3. Ambient: Ánh sáng nền
    vec3 g = vec3(NdotL, specular, 1.0);  // [diffuse, specular, ambient]
    
    // 4. Kết hợp: I = K_materials × I_light × g
    vec3 rgb = 0.5 * matrixCompMult(K_materials, I_light) * g + 0.5 * colorInterp;
    
    fragColor = vec4(rgb, 1.0);
}
```

#### Gouraud Shading (shaders/gouraud.vert):

```glsl
# shaders/gouraud.vert - Tính lighting TẠI VERTEX
void main(){
    vec3 N = normalize(normal_interp);
    vec3 L = normalize(light_pos - vertPos);
    vec3 R = reflect(-L, N);
    vec3 V = normalize(-vertPos);
    
    // Tính lighting tại đỉnh
    float NdotL = max(dot(N, L), 0.0);
    float specAngle = max(dot(R, V), 0.0);
    float specular = pow(specAngle, shininess);
    vec3 g = vec3(NdotL, specular, 1.0);
    vec3 lighting = matrixCompMult(K_materials, I_light) * g;
    
    // Output màu đã tính, fragment shader chỉ nội suy
    colorInterp = 0.5 * lighting + 0.5 * color;
    gl_Position = projection * vertPos4;
}
```

---

## 3. Chi tiết Phần 1 - Game Engine 3D

### 3.1 Camera Trackball

#### Nguyên lý toán học:

Trackball cho phép xoay camera quanh một đối tượng bằng cách kéo chuột.

**1. Quaternion (Rotation Representation):**

```python
# transform.py - Dòng 118-121: Định nghĩa Quaternion
def quaternion(x=vec(0., 0., 0.), y=0.0, z=0.0, w=1.0):
    return np.array((w, x, y, z), 'f')  # [x, y, z, w]

# transform.py - Dòng 139-144: Nhân quaternion
def quaternion_mul(q1, q2):
    return np.dot(np.array([[q1[0], -q1[1], -q1[2], -q1[3]],
                          [q1[1],  q1[0], -q1[3],  q1[2]],
                          [q1[2],  q1[3],  q1[0], -q1[1]],
                          [q1[3], -q1[2],  q1[1],  q1[0]]]), q2)
```

**2. Tạo Quaternion từ Axis-Angle:**

```python
# transform.py - Dòng 124-127
def quaternion_from_axis_angle(axis, degrees=0.0, radians=None):
    sin, cos = sincos(radians=radians*0.5) if radians else sincos(degrees*0.5)
    return quaternion(normalized(vec(axis))*sin, w=cos)
```

**3. Chuyển Quaternion sang Ma trận 4x4:**

```python
# transform.py - Dòng 147-156
def quaternion_matrix(q):
    q = normalized(q)
    nxx, nyy, nzz = -q[1]*q[1], -q[2]*q[2], -q[3]*q[3]
    qwx, qwy, qwz = q[0]*q[1], q[0]*q[2], q[0]*q[3]
    qxy, qxz, qyz = q[1]*q[2], q[1]*q[3], q[2]*q[3]
    return np.array([[2*(nyy + nzz)+1, 2*(qxy - qwz),   2*(qxz + qwy),   0],
                     [2*(qxy + qwz), 2*(nxx + nzz) + 1, 2*(qyz - qwx), 0],
                     [2*(qxz - qwy), 2*(qyz + qwx), 2*(nxx + nyy) + 1, 0],
                     [0, 0, 0, 1]], 'f')
```

#### Trackball Class trong transform.py:

```python
# transform.py - Dòng 176-226
class Trackball:
    def __init__(self, yaw=0., roll=0., pitch=0., distance=3., radians=None):
        self.rotation = quaternion_from_euler(yaw, roll, pitch, radians)
        self.distance = max(distance, 0.001)  # Khoảng cách tới target
        self.pos2d = vec(0.0, 0.0)           # Pan offset
        
    def drag(self, old, new, winsize):
        """Xử lý kéo chuột: tạo quaternion mới từ 2 điểm trên màn hình"""
        old, new = ((2*vec(pos) - winsize) / winsize for pos in (old, new))
        self.rotation = quaternion_mul(self._rotate(old, new), self.rotation)
        
    def zoom(self, delta, size):
        """Phóng to/thu nhỏ bằng scroll"""
        self.distance = max(0.001, self.distance * (1 - 50*delta/size))
        
    def pan(self, dx, dy):
        """Di chuyển camera theo phương ngang"""
        self.pos2d += vec(dx, dy) * 0.001 * self.distance
        
    def view_matrix(self):
        """Ma trận View: Translate → Rotate"""
        return translate(*self.pos2d, -self.distance) @ self.matrix()
```

### 3.2 Hệ thống Ánh sáng (Lighting)

#### Nguyên lý 3 thành phần:

**1. Ambient (Ánh sáng nền):** Môi trường luôn sáng đều, không phụ thuộc vị trí.

**2. Diffuse (Ánh sáng khuếch tán):** Phụ thuộc góc giữa tia sáng và pháp tuyến mặt.
```
I_diffuse = k_d × I_light × max(0, N·L)
```
- `N`: Pháp tuyến mặt
- `L`: Vector từ điểm tới nguồn sáng
- `k_d`: Hệ số diffuse của vật liệu

**3. Specular (Ánh sáng phản xạ):** Tạo vùng sáng trên bề mặt bóng.
```
I_specular = k_s × I_light × pow(max(0, R·V), shininess)
```
- `R`: Vector phản xạ = reflect(-L, N)
- `V`: Vector từ điểm tới camera
- `shininess`: Độ bóng (hệ số dùng trong lũy thừa)

#### Cài đặt trong lighting.py:

```python
# libs/lighting.py - Dòng 98-132
def setup_phong(self, light=None, material=None, mode=1):
    light = light or self.DEFAULT_LIGHT
    material = material or self.DEFAULT_MATERIAL
    
    # Đóng gói thành ma trận 3x3: [diffuse, specular, ambient]
    I_light = np.array([
        light.diffuse,
        light.specular,
        light.ambient
    ], dtype=np.float32)
    
    K_materials = np.array([
        material.diffuse,
        material.specular,
        material.ambient
    ], dtype=np.float32)
    
    # Upload lên GPU
    self.uma.upload_uniform_matrix3fv(I_light, 'I_light', False)
    self.uma.upload_uniform_vector3fv(light.position, 'light_pos')
    self.uma.upload_uniform_matrix3fv(K_materials, 'K_materials', False)
    self.uma.upload_uniform_scalar1f(material.shininess, 'shininess')
```

### 3.3 Transform Gizmo (Move/Rotate/Scale)

#### Nguyên lý:

Gizmo là công cụ transform cho phép thao tác trực tiếp trên đối tượng 3D.

**1. Chiếu 3D → 2D (Screen Projection):**

```python
# libs/gizmo.py - Dòng 136-143
def project_to_screen(self, point_3d, view_matrix, proj_matrix, win_size):
    p = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0], dtype=np.float32)
    p = view_matrix @ p              # Transform to view space
    p = proj_matrix @ p               # Transform to clip space
    if p[3] != 0: p = p / p[3]      # Perspective divide
    
    # Convert to screen coordinates
    screen_x = (p[0] + 1.0) * 0.5 * win_size[0]
    screen_y = (1.0 - p[1]) * 0.5 * win_size[1]
    return np.array([screen_x, screen_y])
```

**2. Phát hiện trục được chọn:**

```python
# libs/gizmo.py - Dòng 145-196
def check_axis_selection(self, mouse_pos, view, projection, win_size, target_pos, current_tool):
    # Tính khoảng cách từ chuột tới mỗi trục (X, Y, Z)
    origin_2d = self.project_to_screen(target_pos, view, projection, win_size)
    
    # Tính vị trí 2D của điểm cách origin 2 đơn vị theo mỗi trục
    x_tip_2d = self.project_to_screen([target_pos[0] + 2.0, ...], ...)
    y_tip_2d = self.project_to_screen([target_pos[0], target_pos[1] + 2.0, ...], ...)
    z_tip_2d = self.project_to_screen([target_pos[0], target_pos[1], target_pos[2] + 2.0], ...)
    
    # Tính khoảng cách chuột tới mỗi đoạn thẳng (origin → tip)
    dist_x = dist_to_segment(mouse, origin_2d, x_tip_2d)
    dist_y = dist_to_segment(mouse, origin_2d, y_tip_2d)
    dist_z = dist_to_segment(mouse, origin_2d, z_tip_2d)
    
    # Chọn trục gần nhất
    min_dist = min(dist_x, dist_y, dist_z)
    if min_dist < 15.0:  # Threshold
        return 'x' if min_dist == dist_x else 'y' if min_dist == dist_y else 'z'
```

**3. Xử lý kéo chuột:**

```python
# libs/gizmo.py - Dòng 212-266
def handle_mouse_drag(self, mouse_pos, target_obj, current_tool, ...):
    if current_tool == 'move':
        # Tính hướng di chuyển 2D
        mouse_delta = mouse_pos - self.drag_start_pos
        move_amount = np.dot(mouse_delta, axis_dir_2d)
        
        # Cập nhật position
        if self.selected_axis == 'x': target_obj.position[0] += move_amount * speed
        elif self.selected_axis == 'y': target_obj.position[1] += move_amount * speed
        elif self.selected_axis == 'z': target_obj.position[2] += move_amount * speed
        
    elif current_tool == 'rotate':
        # Tính góc xoay từ vị trí chuột
        delta_rad = curr_angle - self.start_angle
        delta_deg = math.degrees(delta_rad)
        
        # Cộng dồn vào rotation
        if self.selected_axis == 'x': target_obj.rotation[0] -= delta_deg
        elif self.selected_axis == 'y': target_obj.rotation[1] += delta_deg
        elif self.selected_axis == 'z': target_obj.rotation[2] -= delta_deg
```

### 3.4 Load Model (.obj/.ply)

#### Nguyên lý:

File OBJ/PLY chứa danh sách vertices, normals, UVs và faces (indices).

```python
# geometry/model_loader3d.py
def load_obj(filename):
    vertices, normals, texcoords, indices = [], [], [], []
    
    for line in open(filename):
        if line.startswith('v '):      # Vertex
            v = list(map(float, line.split()[1:]))
            vertices.extend(v)
        elif line.startswith('vn '):    # Vertex Normal
            vn = list(map(float, line.split()[1:]))
            normals.extend(vn)
        elif line.startswith('vt '):    # Vertex Texcoord
            vt = list(map(float, line.split()[1:]))
            texcoords.extend(vt)
        elif line.startswith('f '):      # Face
            face = line.split()[1:]
            for f in face:
                # OBJ format: v/vt/vn hoặc v//vn hoặc v
                v_idx = int(f.split('/')[0]) - 1  # 1-indexed → 0-indexed
                indices.append(v_idx)
    
    return np.array(vertices), np.array(normals), np.array(texcoords), np.array(indices)
```

### 3.5 Texture Mapping

#### Nguyên lý:

Texture mapping gán ảnh 2D lên bề mặt 3D thông qua UV coordinates.

```python
# libs/buffer.py - Dòng 80-106
def setup_texture(self, sampler_name, image_file):
    rgb_image = UManager.load_texture(image_file)  # Load ảnh bằng OpenCV
    
    texture_idx = GL.glGenTextures(1)
    GL.glActiveTexture(GL.GL_TEXTURE0 + binding_loc)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture_idx)
    
    # Upload ảnh lên GPU
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB,
                    rgb_image.shape[1], rgb_image.shape[0],
                    0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, rgb_image)
    
    # Filter: LINEAR = smooth interpolation
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
```

---

## 4. Chi tiết Phần 2 - SGD Visualization

### 4.1 Tổng quan

Phần 2 tái sử dụng engine 3D để trực quan hóa các thuật toán tối ưu hóa AI.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Loss Function z = f(x, y)                     │
│              (Himmelblau, Rosenbrock, Booth, Quadratic)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               SGDVisualizer (geometry/sgd_visualizer.py)          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Surface Mesh │  │  Optimizer  │  │  Trajectory │            │
│  │  (Vertices)  │  │  (Spheres)  │  │   (Lines)   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Loss Functions (Hàm mất mát)

```python
# libs/loss_functions.py - Dòng 20-30
class Himmelblau(OptimizationFunction):
    """Hàm Himmelblau: f(x,y) = (x² + y - 11)² + (x + y² - 7)²"""
    
    def compute(self, x, y):
        return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    
    def gradient(self, x, y):
        # Đạo hàm riêng: ∂f/∂x và ∂f/∂y
        dx = 4 * x * (x**2 + y - 11) + 2 * (x + y**2 - 7)
        dy = 2 * (x**2 + y - 11) + 4 * y * (x + y**2 - 7)
        return np.array([dx, dy])
```

### 4.3 Gradient Descent Algorithm

#### Nguyên lý:

Gradient descent tìm điểm minimum bằng cách di chuyển ngược hướng gradient:

```
position_new = position_old - learning_rate × gradient
```

```python
# geometry/sgd_visualizer.py - Dòng 176-213
def step_optimizer(self, name, learning_rate, momentum=0.0, batch_size=1):
    opt = self.optimizers[name]
    x, y = opt['position']
    
    # Tính gradient tại vị trí hiện tại
    grad = self.loss_func.gradient(x, y)
    
    # GD thuần
    if opt['type'] == 'GD':
        opt['position'] = opt['position'] - learning_rate * grad
    
    # Momentum: Thêm "quán tính"
    elif opt['type'] == 'Momentum':
        opt['momentum_buffer'] = momentum * opt['momentum_buffer'] + grad
        opt['position'] = opt['position'] - learning_rate * opt['momentum_buffer']
    
    # Adam: Adaptive Moment Estimation
    elif opt['type'] == 'Adam':
        t = opt['step'] + 1
        opt['adam_m'] = opt['beta1'] * opt['adam_m'] + (1 - opt['beta1']) * grad
        opt['adam_v'] = opt['beta2'] * opt['adam_v'] + (1 - opt['beta2']) * (grad ** 2)
        
        # Bias correction
        m_hat = opt['adam_m'] / (1 - opt['beta1'] ** t)
        v_hat = opt['adam_v'] / (1 - opt['beta2'] ** t)
        
        update = learning_rate * m_hat / (np.sqrt(v_hat) + opt['epsilon'])
        opt['position'] = opt['position'] - update
    
    opt['loss'] = self.loss_func.compute(opt['position'][0], opt['position'][1])
    opt['step'] += 1
    opt['history'].append(opt['position'].copy())
```

### 4.4 So sánh các thuật toán

| Thuật toán | Công thức cập nhật | Đặc điểm |
|------------|---------------------|-----------|
| **GD** | `x = x - lr × ∇f(x)` | Chậm, ổn định |
| **SGD** | `x = x - lr × ∇f(x)` | Giống GD (batch=1) |
| **MiniBatch** | `x = x - lr × ∇f(x)` | Trung gian |
| **Momentum** | `v = μv + lr×∇f`; `x = x - v` | Nhanh hơn, có quán tính |
| **Nesterov** | `v = μv + lr×∇f(x - μv)` | Momentum nhìn trước |
| **Adam** | adaptive lr từ m, v | Nhanh, ổn định nhất |

### 4.5 Vẽ Surface từ hàm toán học

```python
# geometry/sgd_visualizer.py - Dòng 53-122
def _generate_surface(self):
    # Tạo lưới điểm trong domain
    x_vals = np.linspace(self.x_range[0], self.x_range[1], self.resolution)
    y_vals = np.linspace(self.y_range[0], self.y_range[1], self.resolution)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Tính Z = f(x, y) cho mỗi điểm
    Z = np.zeros_like(X)
    for i in range(self.resolution):
        for j in range(self.resolution):
            Z[i, j] = self.loss_func.compute(X[i, j], Y[i, j])
    
    # Chuẩn hóa Z về khoảng [-2, 2] để vẽ đẹp
    z_scale = 4.0 / max(z_max - z_min, 1.0)
    
    # Tạo vertices với màu theo độ cao
    for i in range(self.resolution):
        for j in range(self.resolution):
            x = (x_vals[j] - self.x_range[0]) * x_scale - 2.0
            y = (y_vals[i] - self.y_range[0]) * y_scale - 2.0
            z = (Z[i, j] - z_min) * z_scale - 2.0
            
            # Màu: cao = đỏ, thấp = xanh
            h = Z_norm[i, j]
            color = [0.2 + 0.6 * h, 0.4 * (1 - abs(h - 0.5) * 2), 0.6 + 0.4 * (1 - h)]
```

---

## 5. Bộ câu hỏi phòng thủ Hội đồng

### Câu 1: Tại sao dùng Quaternion thay vì Euler Angles cho Camera?

**Câu trả lời:**

Quaternion tránh được **Gimbal Lock** (khóa gimbal) - khi 2 trục quay trùng nhau, ta mất 1 bậc tự do.

```python
# transform.py - Dòng 147-156: Quaternion → Matrix
def quaternion_matrix(q):
    q = normalized(q)
    # Tránh gimbal lock vì quaternion không có singularity
    return np.array([[2*(nyy + nzz)+1, ...], ...])
```

### Câu 2: Sự khác biệt giữa Vertex Shader và Fragment Shader trong tính lighting?

**Câu trả lời:**

- **Vertex Shader (Gouraud):** Tính lighting tại các đỉnh, sau đó nội suy tuyến tính cho fragment. **Nhược điểm:** Specular highlight có thể bị miss nếu nằm giữa 2 đỉnh.

- **Fragment Shader (Phong):** Chỉ nội suy pháp tuyến và vị trí, tính lighting tại **mỗi pixel**. **Ưu điểm:** Chính xác hơn, specular luôn đúng.

```glsl
// Phong: Tính tại fragment (mỗi pixel)
vec3 N = normalize(normal_interp);
vec3 L = normalize(light_pos - vertPos);
float specular = pow(max(dot(reflect(-L, N), V), 0.0), shininess);
```

### Câu 3: Tại sao cần Ma trận Normal (transpose(inverse(modelview)))?

**Câu trả lời:**

Khi apply transform (đặc biệt là non-uniform scale), pháp tuyến không biến đổi đúng theo world matrix. Phải dùng **Normal Matrix** để đảm bảo pháp tuyến luôn vuông góc với mặt.

```glsl
// phong.vert - Dòng 18
mat4 normal_matrix = transpose(inverse(modelview));
normal_interp = vec3(normal_matrix * vec4(normalize(position), 0.0));
```

### Câu 4: Gradient Descent hội tụ về đâu?

**Câu trả lời:**

Gradient descent tìm **điểm cực tiểu (local minimum)** của hàm mất mát. Không đảm bảo tìm được global minimum (trừ khi hàm lồi).

- **Himmelblau:** Có 4 điểm minimum tại `(3,2)`, `(-2.8,3.1)`, `(-3.8,-3.3)`, `(3.6,-1.8)` với giá trị `f = 0`
- **Rosenbrock:** Minimum tại `(1,1)` với `f = 0`

### Câu 5: Tại sao Adam thường hội tụ nhanh hơn GD thuần?

**Câu trả lời:**

Adam sử dụng **adaptive learning rate**:
- `m` (first moment): Ước lượng gradient trung bình
- `v` (second moment): Ước lượng bình phương gradient

```python
# geometry/sgd_visualizer.py - Dòng 207-215
m_hat = adam_m / (1 - beta1 ** t)   # Bias-corrected
v_hat = adam_v / (1 - beta2 ** t)    # Bias-corrected
update = learning_rate * m_hat / (sqrt(v_hat) + epsilon)
position = position - update
```

Ưu điểm:
1. Tự điều chỉnh learning rate theo từng tham số
2. Momentum giúp vượt qua local minima nhỏ
3. Bias correction giúp ổn định ở các bước đầu

### Câu 6: Gizmo phát hiện trục được chọn bằng cách nào?

**Câu trả lời:**

1. **Project 3D → 2D:** Chiếu điểm gốc và điểm tip của mỗi trục lên màn hình
2. **Tính khoảng cách:** Khoảng cách từ vị trí chuột tới đoạn thẳng (origin → tip)
3. **Chọn trục gần nhất:** Nếu khoảng cách < threshold (15 pixels), chọn trục đó

```python
# libs/gizmo.py - Dòng 187-196
dist_x = dist_to_segment(mouse, origin_2d, x_tip_2d)
dist_y = dist_to_segment(mouse, origin_2d, y_tip_2d)
dist_z = dist_to_segment(mouse, origin_2d, z_tip_2d)

if min(dist_x, dist_y, dist_z) < 15.0:
    return 'x' if dist_x < dist_y and dist_x < dist_z else ...
```

### Câu 7: Lỗi "Gimbal Lock" xảy ra khi nào và cách tránh?

**Câu trả lời:**

Gimbal Lock xảy ra khi 2 trong 3 gimbal (trục xoay) trùng nhau, mất 1 bậc tự do.

**Khi nào xảy ra:**
- Dùng Euler Angles (pitch = ±90°)
- Xoay object 90° quanh trục Y → X và Z trùng nhau

**Cách tránh trong code:**
- Dùng **Quaternion** thay vì Euler Angles
- Code trong `transform.py` dùng `quaternion_from_euler()` và `quaternion_matrix()`

```python
# transform.py - Dòng 130-136
def quaternion_from_euler(yaw=0.0, pitch=0.0, roll=0.0, radians=None):
    # Quaternion không có gimbal lock vì 4 chiều
    siy, coy = sincos(yaw * 0.5, ...)
    sir, cor = sincos(roll * 0.5, ...)
    sip, cop = sincos(pitch * 0.5, ...)
    return quaternion(x=coy*sir*cop - siy*cor*sip, ...)
```

---

## Tài liệu tham khảo

1. **OpenGL Pipeline:** https://learnopengl.com/
2. **Quaternion Mathematics:** https://eater.net/quaternions
3. **Gradient Descent:** https://ruder.io/optimizing-gradient-descent/
4. **Phong vs Gouraud Shading:** https://en.wikipedia.org/wiki/Gouraud_shading

---

*Tài liệu được viết bởi sinh viên cho mục đích báo cáo BTL Đồ họa Máy tính*