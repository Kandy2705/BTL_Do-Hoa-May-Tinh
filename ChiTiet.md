# Chi Tiết Cấu Trúc Project Đồ Họa Máy Tính

## Tổng Quan Về Project

Project được xây dựng theo kiến trúc MVC (Model-View-Controller) để quản lý và hiển thị các đối tượng đồ họa 2D, 3D, và mathematical surfaces.

---

## 1. CẤU TRÚC MVC

### 1.1 Model (`model.py`)

**Vai trò:** Quản lý dữ liệu và trạng thái của ứng dụng

**Thuộc tính chính:**
- `selected_idx`: Index của hình được chọn trong category
- `selected_category`: Category hiện tại (0: 2D, 1: 3D, 2: Math Surface, 3: Model from file, 4: SGD)
- `selected_shader`: Shader đang được sử dụng
- `active_drawable`: Đối tượng đồ họa đang active
- `drawables`: Danh sách các đối tượng có thể vẽ
- `math_function`: Chuỗi hàm toán học cho mathematical surface
- `model_filename`: Tên file model .obj/.ply

**Phương thức quan trọng:**
- `load_active_drawable()`: Tải đối tượng đồ họa dựa trên lựa chọn
- `set_selected()`: Chọn hình cụ thể
- `set_category()`: Chuyển đổi category
- `set_shader()`: Đổi shader
- `_shape_factories()`: Trả về danh sách các class hình có thể tạo

**Cách hoạt động:**
1. Khi user chọn category và hình, model sẽ import module tương ứng
2. Tạo instance của class hình với shader được chọn
3. Load hình vào `active_drawable` để vẽ

### 1.2 View (`viewer.py`)

**Vai trò:** Quản lý window, OpenGL context, và rendering

**Thành phần chính:**
- **Window Management:** Khởi tạo GLFW window, OpenGL context
- **ImGui Interface:** Tạo UI cho việc chọn hình, shader, parameter
- **Trackball:** Quản lý camera (zoom, rotate, pan)
- **Rendering Loop:** Main loop để vẽ các đối tượng

**Phương thức quan trọng:**
- `__init__()`: Khởi tạo window, OpenGL, ImGui
- `cycle_polygon_mode()`: Chuyển đổi giữa fill/wireframe/point mode
- Callback functions cho mouse/keyboard input

### 1.3 Controller (`controller.py`)

**Vai trò:** Điều phối giữa Model và View, xử lý user input

**Thành phần chính:**
- **Input Handling:** Xử lý keyboard và mouse events
- **Coordinate System:** Quản lý hệ trục tọa độ
- **Shader Management:** Đổi shader runtime

**Phương thức quan trọng:**
- `on_scroll()`: Zoom in/out
- `on_mouse_move()`: Rotate camera
- `on_key()`: Xử lý phím (W: wireframe, S: shader, G: grid toggle)
- `_setup_coordinate_system()`: Setup hệ trục tọa độ

---

## 2. HỆ THỐNG TỌA ĐỘ (`libs/coordinate_system.py`)

**Vai trò:** Vẽ hệ trục tọa độ 3D và lưới tham chiếu

### 2.1 Khởi Tạo
```python
CoordinateSystem(axis_length=20.0, grid_size=1.0)
```

### 2.2 Thành Phần

#### Axes (Trục tọa độ)
- **Trục X:** Màu đỏ, từ (0,0,0) đến (axis_length,0,0)
- **Trục Y:** Màu xanh lá, từ (0,0,0) đến (0,axis_length,0)  
- **Trục Z:** Màu xanh dương, từ (0,0,0) đến (0,0,axis_length)

#### Grid (Lưới)
- Lưới trên mặt phẳng XY
- Màu xám nhạt
- Khoảng cách giữa các đường = `grid_size`
- Phạm vi từ `-axis_length` đến `+axis_length`

### 2.3 Phương Thức Quan Trọng
- `_generate_axes()`: Tạo vertices và colors cho 3 trục
- `_generate_grid()`: Tạo vertices và colors cho lưới
- `setup()`: Thiết lập VAO để render
- `draw()`: Vẽ hệ trục và lưới
- `toggle_visibility()`: Bật/tắt hiển thị

### 2.4 Cách Render
1. Upload projection và modelview matrices
2. Vẽ grid trước (GL_LINES)
3. Vẽ axes sau (GL_LINES) để hiển thị trên cùng

---

## 3. THƯ MỤC GEOMETRY

### 3.1 Cấu Trúc Chung

Mỗi file trong thư mục `geometry/` theo cùng một cấu trúc:

```python
class ShapeName:
    def __init__(self, vert_shader, frag_shader, **kwargs):
        # Khởi tạo vertices, colors, normals
        # Khởi tạo shader, VAO, UManager, LightingManager
    
    def setup(self):
        # Setup VBOs, EBOs, uniforms
        # Upload data lên GPU
    
    def draw(self, projection, view, model):
        # Upload matrices
        # Setup lighting
        # Vẽ đối tượng
```

### 3.2 Các Loại Hình

#### 3.2.1 2D Shapes
**File:** `triangle2d.py`, `rectangle2d.py`, `circle2d.py`, `ellipse2d.py`, `pentagon2d.py`, `hexagon2d.py`, `trapezoid2d.py`, `star2d.py`, `arrow2d.py`

**Đặc điểm chung:**
- Vertices trong mặt phẳng Z=0
- Sử dụng `GL_TRIANGLES` hoặc `GL_TRIANGLE_FAN`
- Màu sắc được gán cho mỗi vertex
- Projection: Orthographic (`T.ortho(-1, 1, -1, 1, -1, 1)`)

**Ví dụ: Triangle2D**
```python
self.vertices = np.array([
    [-1, -1, 0],  # Đỉnh 1
    [+1, -1, 0],  # Đỉnh 2  
    [ 0, +1, 0]   # Đỉnh 3
], dtype=np.float32)

self.colors = np.array([
    [1.0, 0.0, 0.0],  # Đỏ
    [0.0, 1.0, 0.0],  # Xanh lá
    [0.0, 0.0, 1.0]   # Xanh dương
], dtype=np.float32)
```

#### 3.2.2 3D Shapes
**File:** `cube3d.py`, `sphere_tetrahedron3d.py`, `sphere_grid3d.py`, `sphere_latlong3d.py`, `cylinder3d.py`, `cone3d.py`, `truncated_cone3d.py`, `tetrahedron3d.py`, `torus3d.py`, `prism3d.py`

**Đặc điểm chung:**
- Vertices trong không gian 3D
- Sử dụng `GL_TRIANGLE_STRIP` hoặc `GL_TRIANGLES`
- Có normals cho lighting
- Projection: Perspective
- Có indices để tối ưu rendering

**Ví dụ: Cube3D**
```python
# 8 vertices của lập phương
self.vertices = np.array([
    [-1, -1, +1],  # A
    [+1, -1, +1],  # B  
    [+1, -1, -1],  # C
    [-1, -1, -1],  # D
    [-1, +1, +1],  # E
    [+1, +1, +1],  # F
    [+1, +1, -1],  # G
    [-1, +1, -1],  # H
], dtype=np.float32)

# Indices cho triangle strip
self.indices = np.array([
    0, 4, 1, 5, 2, 6, 3, 7, 0, 4, ...
], dtype=np.int32)

# Normals = normalized vertices
self.normals = self.vertices / np.linalg.norm(self.vertices, axis=1, keepdims=True)
```

#### 3.2.3 Mathematical Surface
**File:** `math_surface3d.py`

**Đặc điểm:**
- Tạo mesh từ hàm z = f(x,y)
- Nhận function string từ user
- Parse và eval function an toàn
- Tạo grid vertices và normals

**Cách hoạt động:**
1. Parse function string: `"(x**2 + y - 11)**2 + (x + y**2 - 7)**2"`
2. Tạo grid points (x,y)
3. Tính z = f(x,y) cho mỗi point
4. Tạo normals bằng gradient
5. Tạo indices cho triangle mesh

#### 3.2.4 Model Loader
**File:** `model_loader3d.py`

**Đặc điểm:**
- Load file .obj/.ply
- Parse vertices, faces, normals
- Tự động tính normals nếu thiếu
- Hỗ trợ materials và textures

### 3.3 Lighting System

**File:** `libs/lighting.py`

**Components:**
- **LightingManager:** Quản lý lighting parameters
- **Gouraud Shading:** Tính màu per-vertex
- **Phong Shading:** Tính màu per-fragment

**Setup:**
```python
# Gouraud
self.lighting.setup_gouraud()

# Phong  
self.lighting.setup_phong(mode=1)
```

---

## 4. SHADER SYSTEM

### 4.1 Shader Types
1. **Color Interpolation:** Interpolate màu giữa vertices
2. **Gouraud:** Per-vertex lighting
3. **Phong:** Per-fragment lighting

### 4.2 Shader Files
- **Vertex Shaders:** `shaders/*.vert`
- **Fragment Shaders:** `shaders/*.frag`

### 4.3 Shader Switching
User có thể switch shader runtime bằng phím 'S':
```python
def on_key(self, window, key, scancode, action, mods):
    if key == glfw.KEY_S:
        self.model.set_shader((self.model.selected_shader + 1) % 3)
```

---

## 5. BUFFER MANAGEMENT

### 5.1 VAO (Vertex Array Object)
- Quản lý state của vertex attributes
- Chứa VBOs và EBOs

### 5.2 VBO (Vertex Buffer Object)  
- Chứa vertex data (positions, colors, normals)
- Upload lên GPU một lần trong `setup()`

### 5.3 EBO (Element Buffer Object)
- Chứa indices để tránh duplicate vertices
- Tối ưu cho complex meshes

### 5.4 UManager (Uniform Manager)
- Quản lý uniform variables
- Upload matrices và lighting parameters

---

## 6. MAIN FLOW

### 6.1 Startup Sequence
1. `main.py` tạo `AppController`
2. `AppController` tạo `AppModel` và `Viewer`
3. `Viewer` khởi tạo OpenGL window và ImGui
4. `AppController` setup coordinate system
5. `AppModel` load default drawable

### 6.2 Render Loop
1. Clear framebuffer
2. Update ImGui
3. Upload projection/view matrices
4. Vẽ coordinate system
5. Vẽ active drawable
6. Render ImGui
7. Swap buffers

### 6.3 User Interaction
1. ImGui: Chọn category, shape, shader
2. Mouse: Rotate camera (drag), Zoom (scroll)
3. Keyboard: 
   - W: Toggle wireframe
   - S: Switch shader
   - G: Toggle grid
   - Q: Quit

---

## 7. EXTENDING THE SYSTEM

### 7.1 Adding New Shapes
1. Tạo file mới trong `geometry/`
2. Implement class với `__init__`, `setup`, `draw`
3. Add to `model.py` `_shape_factories()`
4. Add to `model.py` `menu_options`

### 7.2 Adding New Shaders
1. Tạo `.vert` và `.frag` files
2. Add paths to `model.py` `_shader_paths()`
3. Add name to `model.py` `shader_names`

### 7.3 Adding New Categories
1. Update `category_options` in `model.py`
2. Add handling in `selected_category` logic
3. Update `_shape_factories()` for new category

---

## 8. TROUBLESHOOTING

### 8.1 Common Issues
- **Shader compilation:** Check shader syntax
- **Missing normals:** Ensure normals are normalized
- **Projection issues:** Check matrix dimensions
- **Lighting not working:** Verify uniform locations

### 8.2 Debug Tips
- Use `print()` in `setup()` and `draw()`
- Check OpenGL errors with `glGetError()`
- Verify VAO/VBO binding states
- Test with simple shapes first

---

## 9. PERFORMANCE CONSIDERATIONS

### 9.1 Optimizations
- Use VBOs/EBOs to reduce memory bandwidth
- Batch similar draw calls
- Use frustum culling for complex scenes
- Implement LOD for high-poly meshes

### 9.2 Memory Management
- Delete unused VAOs/VBOs
- Use buffer orphaning for dynamic data
- Minimize state changes

---

## 10. FUTURE ENHANCEMENTS

### 10.1 Features to Add
- Texture mapping
- Shadow mapping  
- Post-processing effects
- Animation system
- Physics simulation

### 10.2 Improvements
- Better error handling
- More lighting models
- Material system
- Scene graph
- Asset management pipeline
