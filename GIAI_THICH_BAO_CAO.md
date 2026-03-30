# Giải thích đồ án để báo cáo

## 1. Nhìn nhanh dự án này đang làm gì

Đây là một mini engine đồ họa viết bằng Python, dùng:

- `GLFW` để tạo cửa sổ và nhận input
- `PyOpenGL` để gọi OpenGL
- `PyImGui` để dùng giao diện editor kiểu Unity
- `NumPy` để xử lý ma trận, vector, hình học

Dự án thực hiện 2 nhóm chức năng lớn:

1. Hiển thị và thao tác đối tượng 2D/3D, bề mặt toán học, model `.obj/.ply`, ánh sáng, camera, texture, gizmo.
2. Trực quan hóa các thuật toán tối ưu `GD`, `SGD`, `MiniBatch`, `Momentum`, `Nesterov`, `Adam` trên mặt loss 3D.

Nói để nhớ nhỏ nhất khi báo cáo:

- Đây không chỉ là chương trình "vẽ hình".
- Đây là một hệ editor nhỏ: có `Scene`, `Hierarchy`, `Inspector`, camera, light, object transform, shader, texture và gizmo.

---

## 2. Chương trình bắt đầu từ đâu

File bắt đầu là [main.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/main.py).

### `main()` trong `main.py`

- `main()` khởi tạo `AppController()` rồi gọi `controller.run()`.
- `try/except/finally` dùng để:
  - bắt `KeyboardInterrupt`
  - in traceback nếu lỗi
  - báo ứng dụng đã đóng

Ý nghĩa:

- `main.py` rất mỏng.
- Toàn bộ logic thực sự nằm ở `controller.py`.

---

## 3. Kiến trúc tổng thể

Dự án theo mô hình MVC:

- `Model`: [model.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/model.py)
- `View`: [viewer.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/viewer.py)
- `Controller`: [controller.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/controller.py)

Luồng tổng quát:

1. `main.py` tạo `AppController`
2. `AppController` tạo `Viewer` và `AppModel`
3. `Viewer` tạo cửa sổ OpenGL + ImGui
4. `AppModel` giữ toàn bộ state hiện tại
5. Trong `run()`:
   - đọc input
   - vẽ UI
   - xử lý action từ UI
   - cập nhật scene/SGD
   - render grid + object + gizmo

---

## 4. Dự án này có 2 kiểu sử dụng khác nhau

Đây là điểm rất quan trọng khi báo cáo, vì nhìn code sẽ thấy hơi "lạ".

### Kiểu 1: Preview theo category

State nằm trong `AppModel.selected_category`:

- `0`: 2D Shapes
- `1`: 3D Shapes
- `2`: Mathematical Surface
- `3`: Model from file
- `4`: SGD Visualization

Lúc này dự án dùng `self.model.drawables` để hiển thị "một đối tượng đang được chọn".

### Kiểu 2: Normal Mode / Scene Editor

- Mặc định lúc mở app là `selected_category = 5`
- Đây là mode scene editor
- Lúc này người dùng tạo object vào `scene.objects`
- `Hierarchy` + `Inspector` + `Transform Gizmo` hoạt động mạnh nhất trong mode này.

Nói để trình bày:

- Một nhánh dùng để demo từng shape/shader
- Một nhánh dùng để lập scene giống editor nhỏ của Unity

---

## 4.1. Đối chiếu nhanh với đề BTL 1 phần 1 và phần 2

Sau khi đối chiếu với `cgAssignments_vi.pdf`, có thể chốt ngắn gọn như sau:

### Phần 1 - Vẽ các hình cơ bản

Đã có:

- Đủ 9 hình 2D bắt buộc.
- Đủ nhóm 3D bắt buộc.
- Mặt cầu được mô hình bằng 3 cách:
  - chia nhỏ tứ diện rồi chuẩn hóa
  - chia lưới rồi chiếu lên mặt cầu
  - kinh-vĩ độ
- Mặt toán học `z = f(x, y)`.
- Nạp model `.obj/.ply`.
- GUI, zoom, pan, rotate.
- Nhiều đèn và bật/tắt đèn.
- Nhiều camera và chuyển camera.
- RGB / depth map.
- Texture mapping.
- Wireframe.

Điểm mình đã chỉnh thêm để bám đề hơn:

- Sửa phím đổi shader để chạy đủ 4 mode thay vì bị kẹt ở 3 mode.
- Bật đúng mode "Rainbow / color interpolation" trong preview để bạn demo phần nội suy màu dễ hơn.
- Sửa lại `Flat Shading` để không còn chỉ "tô màu rồi return", mà có chiếu sáng theo từng mặt đúng nghĩa flat shading.
- Gắn thêm `Phong shading` cho mode nội suy màu để vừa có chuyển màu đẹp vừa có khối sáng tối.
- Sửa `MathematicalSurface` để các lựa chọn trong Inspector như shader, texture, màu, flat shading hoạt động đồng bộ hơn.
- Sửa loader `.obj` để tự lần theo `.mtl` và `map_Kd` texture trong cùng thư mục model khi dữ liệu vật liệu có sẵn.

### Phần 2 - Minh họa SGD và các biến thể

Đã có sẵn:

- 4 hàm loss bắt buộc.
- Hiển thị surface 3D.
- Chạy song song nhiều optimizer.
- Hiển thị trajectory, loss, độ lớn gradient, số bước.
- Start / pause / reset.
- Điều chỉnh learning rate, momentum, batch size, speed.

Ghi chú khi báo cáo:

- Hiện tại phần trực quan hóa SGD đang dùng `surface 3D`; chưa có một chế độ `contour` tách riêng.
- Tuy nhiên trong đề ghi "surface 3D hoặc contour", nên phần hiện thực hiện tại vẫn bám yêu cầu bắt buộc.

Điểm mình đã sửa và bổ sung:

- Thêm điều khiển `Max Steps` trong GUI để khớp yêu cầu "số bước lặp tối đa".
- Sửa `SGD` và `MiniBatch` để không còn đi y hệt `GD`.
- Bật/tắt trajectory bằng GUI đúng với state trong model.
- Thêm projected trail ở mặt đáy và đường thả từ optimizer xuống mặt đáy để phần demo trực quan và "ngầu" hơn.

Kết luận ngắn để nói với thầy:

- Phần 1 và phần 2 hiện tại đã phủ được gần như đầy đủ các ý bắt buộc của BTL 1.
- Những điểm mình vừa vá là để phần SGD sát đề hơn và phần trực quan hóa rõ ràng hơn khi demo.

---

## 5. Giải thích file chính

## 5.1. `controller.py`

File: [controller.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/controller.py)

`AppController` là bộ não điều phối.

### `__init__()` - dòng 13

Làm các việc:

- tạo `Viewer`
- tạo `AppModel`
- đưa reference của `model` vào `viewer`
- tạo `CoordinateSystem`
- gán callback input của `Viewer` sang hàm trong `Controller`

Ý nghĩa:

- `Viewer` chỉ biết nhận sự kiện
- `Controller` mới quyết định "sự kiện đó nghĩa là gì"

### `on_scroll()` - dòng 28

- Lấy kích thước cửa sổ
- gọi `trackball.zoom()`

Lý thuyết:

- camera được điều khiển bởi trackball
- zoom thực chất là thay đổi khoảng cách camera-đến-scene

### `on_mouse_move()` - dòng 32

Hàm này biến chuyển động chuột thành hành vi camera:

- nếu ImGui đang giữ chuột thì bỏ qua
- nếu giữ chuột trái:
  - `hand`: pan camera
  - `rotate` hoặc `select`: xoay camera bằng `trackball.drag`
  - `move`: vừa pan theo X/Y, vừa thay đổi `distance` theo Z
  - `scale`: chưa làm ở đây, để cho gizmo

Lưu ý:

- Phần thao tác object bằng gizmo lại được xử lý chủ yếu ở `viewer.py`
- nên `controller.py` đang đóng vai trò điều khiển camera là chính

### `on_mouse_button()` - dòng 82

- hiện tại để trống
- comment ghi rõ: phần gizmo do `viewer` xử lý

### `on_key()` - dòng 87

Đây là bảng phím tắt lớn của dự án:

- `W`: đổi wireframe/fill/point
- `Q`: thoát
- `S`: đổi shader
- `G`: ẩn/hiện grid
- mũi tên: pan camera
- `1`, `2`, `3`: tắt/bật từng đèn
- `C`: đổi góc nhìn giữa scene camera và game camera
- `Space`, `R`, `T`: điều khiển SGD

### `_setup_coordinate_system()` - dòng 164

- tạo shader màu đơn giản cho grid/trục
- tạo `VAO`, `Shader`, `UManager`
- gọi `coord_system.setup(...)`

### `_process_ui_actions()` - dòng 175

Đây là hàm cực kỳ quan trọng.
Nó nhận `actions` từ ImGui rồi đổi thành thay đổi thật trong model/scene.

Nhóm action chính:

- đổi category, đổi shape, đổi shader
- đổi hàm toán học
- chọn model file, texture file
- thêm `Light`, `Camera`
- cập nhật transform, mesh color, shader, camera, light
- chọn object trong hierarchy
- xóa object
- đồng bộ giá trị Inspector vào object
- bấm `Apply` để tạo lại math surface
- đổi tool (`hand`, `move`, `rotate`, `scale`)
- bật `Flat Color`
- đổi `RGB / Depth Map`

Ý nghĩa kiến trúc:

- UI không sửa trực tiếp hệ thống lớn
- UI chỉ trả ra action dictionary
- `Controller` là nơi giải mã action

### `_browse_texture_file()`, `_browse_model_file()`, `_browse_texture_for_specific_object()`, `_browse_model_for_specific_object()`

- dùng `osascript` để mở hộp thoại chọn file native trên macOS
- khi chọn xong sẽ cập nhật vào `model` hoặc object cụ thể

### `run()` - dòng 574

Đây là main loop của chương trình.

Mỗi frame:

1. `poll_events()`
2. `begin_frame()`
3. `draw_ui(...)` để lấy `ui_actions`
4. xử lý đổi category/shape
5. gọi `_process_ui_actions(ui_actions)`
6. nếu đang ở SGD và đang run thì gọi `sgd_step()`
7. đồng bộ camera object với `trackball`
8. tính `view`, `projection`
9. vẽ coordinate system
10. vẽ object
11. `end_frame()`

Nói để nói với thầy:

- `Controller.run()` là tim của ứng dụng
- nó nối input, UI, cập nhật dữ liệu, và render vào chung một chu kỳ frame

---

## 5.2. `viewer.py`

File: [viewer.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/viewer.py)

`Viewer` phụ trách:

- tạo window
- tạo OpenGL context
- tạo ImGui context
- giữ trackball/camera đang hoạt động
- vẽ object và UI

### `__init__()` - dòng 17

Làm các việc:

- `glfw.init()`
- xin context OpenGL 3.3 Core
- tạo cửa sổ
- bật `DEPTH_TEST`
- tạo ImGui renderer
- tạo `default_trackball`
- tạo `TransformGizmo`
- load icon texture cho toolbar
- đăng ký callback chuột/phím

### `set_model_reference()` - dòng 60

- `Viewer` cần `model` để biết object nào đang được chọn, tool nào đang bật

### property `trackball` - dòng 64

Hàm này cực hay:

- nếu `active_camera_idx == 0` thì dùng `default_trackball`
- nếu `> 0` thì tìm `GameObjectCamera` trong scene và trả về `trackball` của camera đó

Tức là:

- dự án có 2 lớp camera:
  - scene camera để điều khiển tự do
  - game camera là object trong scene

### `_apply_unity_style()` - dòng 83

- đổi màu giao diện, rounding, spacing
- tạo cảm giác editor giống Unity Dark

### `_on_scroll()`, `_on_key()` - dòng 106, 167

- đây là lớp wrapper
- `Viewer` nhận event từ GLFW rồi chuyển tiếp sang callback thật của controller nếu ImGui không đang chiếm input

### `on_mouse_move()` - dòng 109

Phần này xử lý:

- chuột phải: xoay camera
- chuột trái + `hand`: pan
- chuột trái + object được chọn + tool move/rotate/scale: kéo gizmo

### `on_mouse_button()` - dòng 146

- lúc nhấn chuột trái, nếu có 1 object được chọn và tool là move/rotate/scale:
  - tính `view`, `projection`, `window size`
  - gọi `gizmo.handle_mouse_press(...)`
- lúc nhả chuột:
  - gọi `gizmo.handle_mouse_release()`

### `load_texture()` - dòng 172

- mở ảnh bằng PIL
- lật ảnh theo trục dọc cho đúng quy ước OpenGL
- tạo texture OpenGL 2D

### `begin_frame()` / `end_frame()` - dòng 195, 199

- `begin_frame`: clear color + depth, tạo frame ImGui mới
- `end_frame`: render ImGui rồi `swap_buffers`

### `draw_drawables()` - dòng 204

Đây là hàm render scene chính.

Nó làm 3 việc:

1. tính `view`, `projection`
2. nếu đang ở category SGD thì gọi `sgd_visualizer.draw(...)`
3. nếu là scene bình thường:
   - duyệt qua `scene_objects`
   - cập nhật transform vào `drawable`
   - đưa danh sách đèn vào `drawable.scene_lights`
   - tự set các uniform global:
     - `u_display_mode`
     - `u_cam_far`
     - `u_shininess`
     - `u_light_range`
   - gọi `obj.drawable.draw(...)`

Cuối cùng:

- nếu chỉ chọn 1 object và tool là move/rotate/scale thì vẽ gizmo
### `cycle_polygon_mode()` - dong 272

- luân phiên `GL_FILL -> GL_LINE -> GL_POINT`

### `draw_coordinate_system()` - dòng 275

- vẽ grid + trục nếu `visible`

### `draw_ui()` - dòng 280

Dùng các component:

- `MainMenu`
- viewport toolbar
- transform toolbar
- `HierarchyPanel`
- `InspectorPanel`
- `Project`
- `SGDPanel` nếu category = 4

Hàm này không sửa dữ liệu trực tiếp.
Nó gom tất cả hành động vào `actions`.

---

## 5.3. `model.py`

File: [model.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/model.py)

`AppModel` là nơi giữ state.

### `__init__()` - dòng 22

State chính:

- shape đang chọn
- category đang chọn
- shader đang chọn
- file model, file texture
- màu object
- tool đang dùng
- SGD state
- `display_mode`
- `Scene`
- `hierarchy_objects`
- `mesh_components`

Nói ngắn gọn:

- bất kỳ thông tin nào cần tồn tại qua nhiều frame đều nằm trong `AppModel`

### `menu_options` - dòng 89

- trả ra danh sách shape theo `selected_category`

### `shader_names` - dòng 127

- danh sách tên để hiển thị trên Inspector/UI

### `_shape_factories()` - dòng 131

- map category hiện tại sang `(module_name, class_name)`
- đây là cơ chế dynamic import

### `_shader_paths()` - dòng 168

- map `selected_shader` thành file shader
- `0`: `color_interp`
- `1`: `gouraud`
- `2`: `phong`
- còn lại: `standard`

### `load_active_drawable()` - dòng 177

Hàm quan trọng nhất của preview mode.

Nó làm:

1. xóa `active_drawable` cũ
2. kiểm tra shape có hợp lệ không
3. import class shape động
4. chọn shader
5. nếu là `MathematicalSurface`:
   - dùng `exec()` tạo hàm `f(x, y)` từ chuỗi
6. nếu là `ModelLoader`:
   - truyền `filename`
7. `drawable.setup()`
8. đưa vào `self.drawables`

Ý nghĩa:

- Model không code cứng từng shape
- mà nạp class theo tên module

### `set_selected()`, `set_category()`, `set_shader()`

- cập nhật state
- gọi `load_active_drawable()` để tạo lại object preview

### `set_math_function()`, `set_model_filename()`, `set_texture_filename()`

- lưu input người dùng
- riêng `set_texture_filename()` còn gọi `active_drawable.set_texture(...)` nếu có

### `add_hierarchy_object()` - dòng 271

Hàm này dùng trong scene editor.

Tùy theo `obj_type`:

- `3d`: tạo `GameObjectOBJ` + drawable 3D
- `2d`: tạo `GameObjectOBJ` + drawable 2D
- `math`: tạo `GameObjectMath` + `MathematicalSurface`
- `custom_model`: tạo `GameObjectOBJ` + `ModelLoader`
- `light`: tạo `GameObjectLight`
- `camera`: tạo `GameObjectCamera` + thêm `Trackball`

Sau đó:

- thêm vào `scene.objects`
- chọn object vừa tạo
- thêm metadata vào `hierarchy_objects`

### `select_hierarchy_object()`, `get_selected_hierarchy_object()`

- phục vụ UI cũ

### `update_object_data()` - dòng 427

- tìm object theo `id`
- nếu object có drawable thì cập nhật một số state đặc biệt:
  - `u_use_flat_color`
  - `u_enable_lighting`
  - `shader`

### `update_selected_object_data()` - dòng 454

- cập nhật dữ liệu lồng nhau trong `hierarchy_objects`
- nếu object đang chọn là `math` và đổi script:
  - tạo lại surface
- nếu là `custom_model` và đổi filename:
  - tạo lại model loader

### `set_color()` - dòng 502

- đổi màu chung
- nếu `active_drawable` có `set_color` thì đổi màu trên GPU

### `_reload_hierarchy_object()` - dòng 509

Dùng khi:

- đổi hàm toán học
- đổi file model

Nó tìm `scene_obj` theo `id`, rồi tạo lại `drawable` mới.

### `reload_current_shape()` - dòng 585

- gọi lại `load_active_drawable()`

### Cụm SGD - dòng 591 trở đi

#### `init_sgd_visualizer()`

- chọn loss function từ `LOSS_FUNCTIONS`
- tạo `SGDVisualizer`
- add các optimizer đang bật
- setup VAO/shader cho visualizer

#### `set_sgd_loss_function()`

- đổi hàm loss và khởi tạo lại

#### `sgd_step()`

- mỗi frame sẽ cho từng optimizer đi 1 bước
- sau đó update quãng đường

#### `reset_sgd()`

- đưa các optimizer về vị trí ban đầu

#### `get_sgd_stats()`

- lấy thống kê để đổ vào panel

#### `toggle_optimizer_enabled()`

- tắt/bật từng thuật toán

---

## 5.4. `core/GameObject.py` và `components/scene.py`

File: [core/GameObject.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/core/GameObject.py)

`GameObject` là lớp dữ liệu cơ bản của scene:

- `id`
- `name`
- `type`
- `shader`, `color`, `texture_filename`
- `visible`
- `position`, `rotation`, `scale`

Lớp con:

- `GameObjectMath`: thêm `math_script`
- `GameObjectOBJ`: thêm `obj_ply_file`
- `GameObjectLight`: thêm `light_color`, `light_intensity`
- `GameObjectCamera`: thêm `camera_fov`, `camera_near`, `camera_far`

File: [components/scene.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/components/scene.py)

`Scene` rất gọn:

- `objects`
- `selected_objects`
- `add_object`
- `remove_object`
- `select_object`
- `clear_selection`

Tức là scene graph hiện tại chưa có cha-con.
Nó đang là danh sách object phẳng.

---

## 5.5. UI Components

### `components/main_menu.py`

- tạo menu `File`, `BTL 1`, `BTL 2`
- menu `BTL 1` đổi sang các category chính
- trả lại `actions`

### `components/hierarchy_panel.py`

- vẽ cây `MainScene`
- normal mode mới cho thêm object bằng popup context
- click object → trả action `select_object`
- right click → `delete_object`
- click vào khoảng trong → `clear_selection`

### `components/inspector_panel.py`

Inspector hiển thị theo type của object đang chọn:

- `Transform`
- `Mesh Renderer`
- `Math Script`
- `Camera Settings`
- `Light Settings`

Action quan trọng:

- `update_attr`
- `browse_texture_for_object`
- `apply_math`

### `components/sgd_panel.py`

Panel này dùng để:

- chọn loss function
- đổi learning rate, momentum, batch size, speed
- tắt/bật optimizer
- đổi initial position
- start/stop/reset
- hiển statistics
- đổi render mode của mặt surface

---

## 6. Thư viện nền

## 6.1. `libs/transform.py`

File: [libs/transform.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/libs/transform.py)

Đây là file lý thuyết nhất của project.

### Nhóm hàm vector/ma trận

- `vec(...)`: tạo vector `numpy`
- `normalized(v)`: chuẩn hóa vector
- `lerp(a, b, t)`: nội suy tuyến tính
- `identity()`: ma trận đơn vị
- `ortho(...)`: phép chiếu trực giao
- `perspective(...)`: phép chiếu phối cảnh
- `frustum(...)`: phối cảnh tổng quát
- `translate(...)`: ma trận tịnh tiến
- `scale(...)`: ma trận co giãn
- `sincos(...)`: trả sin/cos
- `rotate(axis, angle)`: ma trận quay theo công thức Rodrigues
- `lookat(eye, target, up)`: ma trận View

### Nhóm quaternion

- `quaternion(...)`
- `quaternion_from_axis_angle(...)`
- `quaternion_from_euler(...)`
- `quaternion_mul(q1, q2)`
- `quaternion_matrix(q)`
- `quaternion_slerp(q0, q1, t)`

Lý do dùng quaternion:

- xoay mượt
- tránh gimbal lock tốt hơn Euler khi thao tác trackball

### `Trackball`

- `__init__`: giữ `rotation`, `distance`, `pos2d`
- `drag`: đổi chuột thành quaternion quay
- `zoom`: đổi `distance`
- `pan`: đổi `pos2d`
- `view_matrix`: tạo ma trận camera
- `projection_matrix`: tạo perspective matrix từ `fov`, `near`, `far`
- `_project3d` + `_rotate`: biến điểm chuột 2D thành thao tác quay trên "quả cầu ảo"

Nói để báo cáo:

- camera trong project không là camera FPS
- mà là camera kiểu trackball orbit editor

## 6.2. `libs/shader.py`

- `Shader.__init__`: compile vertex + fragment shader, link program
- `_compile_shader`: đọc file hoặc nhận source string, compile, in log nếu lỗi
- `__del__`: xóa program khi object hủy

## 6.3. `libs/buffer.py`

### `VAO`

- giữ `vao`, `vbo`, `ebo`
- `add_vbo`: nạp attribute vào GPU
- `add_ebo`: nạp index buffer
- `activate` / `deactivate`

Lý thuyết:

- `VBO`: dữ liệu đỉnh
- `EBO`: chỉ số nối đỉnh thành tam giác
- `VAO`: "bản mô tả" layout attribute

### `UManager`

- quản lý upload uniform và texture
- `setup_texture(...)`
- `upload_uniform_matrix4fv(...)`
- `upload_uniform_matrix3fv(...)`
- `upload_uniform_vector4fv(...)`
- `upload_uniform_vector3fv(...)`
- `upload_uniform_scalar1f(...)`
- `upload_uniform_scalar1i(...)`

## 6.4. `libs/coordinate_system.py`

- `_generate_axes()`: tạo trục X/Y/Z màu đỏ-xanh lá-xanh dương
- `_generate_grid()`: tạo lưới XY hoặc XZ tùy mode
- `setup(...)`: nạp dữ liệu vào VAO
- `draw(...)`: vẽ lưới và trục bằng `GL_LINES`
- `toggle_visibility()`: ẩn/hiện
- `set_mode(is_3d)`: đổi mặt phẳng lưới

## 6.5. `libs/gizmo.py`

`TransformGizmo` là một class nhỏ nhưng rất quan trọng.

### `__init__`

- giữ `trackball`, `window_size`
- tạo `shader` và `VAO` cho các trục

### `draw()`

- chỉ vẽ khi có object được chọn và tool là move/rotate/scale
- vẽ 3 trục X/Y/Z với màu riêng

### `handle_mouse_press()` / `handle_mouse_move()` / `handle_mouse_release()`

- `press`: lưu mouse start, tìm axis gần nhất
- `move`: tính delta, cập nhật transform object
- `release`: kết thúc drag

### `_project_to_screen()` / `_unproject_from_screen()`

- biến đổi giữa world space và screen space
- dùng để detect axis selection

---

## 6.6. `libs/lighting.py`

`LightingManager` cũ (đã không còn dùng trong geometry mới).

---

## 7. Shader System

## 7.1. File shaders (`shaders/`)

### `color_interp.vert/.frag`

- vertex: truyền position và color
- fragment: nội suy màu

### `gouraud.vert/.frag`

- vertex: tính lighting ở vertex shader
- fragment: chỉ nội suy màu đã tính

### `standard.vert/.frag`

- vertex: truyền position, normal, UV
- fragment: full lighting pipeline:
  - ambient
  - diffuse (Lambert)
  - specular (Phong)
  - multi-lighting (tối đa 4 đèn)
  - texture mapping
  - flat color override

### Uniform naming convention

- `projection`, `modelview`, `view`: ma trận biến đổi
- `u_use_flat_color`, `u_flat_color`: màu phẳng
- `u_use_texture`, `u_texture`: texture
- `u_render_mode`: chế độ render (0=None, 1=Gouraud, 2=Phong)
- `u_*_light[]`: array lighting cho multi-light

---

## 8. Geometry System

## 8.1. Base Shape (`geometry/base_shape.py`)

`BaseShape` là lớp cha của tất cả geometry.

### Transform

- `position`, `rotation`, `scale`: vector3
- `get_transform_matrix()`: tính ma trận 4x4

### Common interface

- `setup()`: tạo VAO/VBO
- `draw(projection, view, model)`: render
- `set_color(color)`: đổi màu
- `set_texture(filepath)`: nạp texture
- `cleanup()`: dọn dẹp GPU

---

## 8.2. 2D Shapes (`geometry/2d/`)

Đặc điểm chung:

- Z = 0 cho tất cả vertices
- Normals = (0, 0, 1)
- Render mode mặc định = 0 (solid color)

Các shape:

- `arrow2d.py`: mũi tên 2D
- `circle2d.py`: hình tròn
- `ellipse2d.py`: hình elip
- `hexagon2d.py`: lục giác
- `pentagon2d.py`: ngũ giác
- `rectangle2d.py`: hình chữ nhật
- `star2d.py`: ngôi sao
- `trapezoid2d.py`: hình thang
- `triangle2d.py`: tam giác

---

## 8.3. 3D Shapes (`geometry/3d/`)

Đặc điểm chung:

- Có normals thực sự
- Render mode mặc định = 2 (Phong shading)
- Hỗ trợ texture mapping

Các shape:

- `cone3d.py`: hình nón
- `cube3d.py`: hình lập phương
- `cylinder3d.py`: hình trụ
- `prism3d.py`: hình lăng trụ
- `sphere_grid3d.py`: cầu dùng lưới
- `sphere_latlong3d.py`: cầu dùng kinh-vĩ độ
- `sphere_tetrahedron3d.py`: cầu từ tứ diện
- `tetrahedron3d.py`: tứ diện
- `torus3d.py`: hình xuyến
- `truncated_cone3d.py`: hình nón cụt

---

## 8.4. Special Geometry

### `model_loader3d.py`

- Load file `.obj` và `.ply`
- Tự động tính normals nếu thiếu
- Hỗ trợ texture UV

### `math_surface3d.py`

- Tạo bề mặt từ hàm toán học f(x,y)
- Màu nội suy theo chiều cao
- Giữ lại màu đẹp khi đồng bộ shader mới

---

## 9. SGD Visualization

## 9.1. Loss Functions

Các hàm loss có sẵn:

- `Sphere`: f(x,y) = x² + y²
- `Rosenbrock`: f(x,y) = (1-x)² + 100(y-x²)²
- `Rastrigin`: f(x,y) = 20 + x² + y² - 10(cos(2πx) + cos(2πy))
- `Ackley`: hàm phức tạp với nhiều local minima

## 9.2. Optimizers

6 thuật toán được implement:

1. **Gradient Descent (GD)**: đi theo gradient chính xác
2. **Stochastic GD (SGD)**: thêm noise vào gradient
3. **Mini-Batch GD**: trung bình nhiều gradient có noise
4. **Momentum**: tích lũy velocity để vượt local minima
5. **Nesterov**: momentum có lookahead
6. **Adam**: adaptive learning rate với moment estimates

## 9.3. Visualization Features

- Surface 3D của loss function
- Trajectory của mỗi optimizer
- Drop lines từ optimizer xuống mặt đáy
- Projected trails trên mặt đáy
- Real-time animation
- Toggle trajectory on/off

---

## 10. Công nghệ và Dependencies

### Core Libraries

- **Python 3.8+**: ngôn ngữ chính
- **NumPy**: toán học, ma trận, vector
- **PyOpenGL**: binding OpenGL
- **GLFW**: window management và input
- **PyImGui**: UI framework

### Additional Libraries

- **PIL (Pillow)**: image processing cho texture
- **PyObjC**: macOS native dialogs

### OpenGL Version

- **Context**: OpenGL 3.3 Core Profile
- **Shaders**: GLSL 330+

---

## 11. Điểm nổi bật khi báo cáo

### 11.1. Kiến trúc kỹ thuật

- **MVC Pattern**: tách biệt rõ ràng giữa logic, view và data
- **Dynamic Loading**: nạp geometry class theo tên module
- **Component System**: UI components độc lập, dễ mở rộng
- **Event-driven**: UI trả actions, controller xử lý

### 11.2. Graphics Features

- **Multi-lighting**: 4 nguồn sáng đồng thời
- **Shader Pipeline**: vertex/fragment shader hoàn chỉnh
- **Texture Mapping**: UV mapping và filtering
- **Transform Gizmo**: 3D manipulation tool

### 11.3. Educational Value

- **SGD Visualization**: trực quan hóa 6 thuật toán tối ưu
- **Real-time Animation**: thấy được hành vi của từng optimizer
- **Interactive Controls**: điều chỉnh parameters real-time

### 11.4. Code Quality

- **Vietnamese Comments**: toàn bộ code có chú thích tiếng Việt
- **Modular Design**: chia nhỏ thành modules độc lập
- **Error Handling**: try/catch và validation

---

## 12. Kết luận

Đây là một mini engine đồ họa hoàn chỉnh với:

### Graphics Engine
- Full rendering pipeline với OpenGL 3.3
- Multi-lighting system với 4 đèn
- Texture mapping và shader management
- Transform gizmo 3D

### Editor Interface
- UI/UX giống Unity với ImGui
- Hierarchy và Inspector panels
- Real-time object manipulation
- Multi-camera system

### Educational Tools
- SGD visualization với 6 thuật toán
- 3D surface rendering
- Interactive parameter controls

### Technical Achievement
- 2000+ lines code Python
- 20+ geometry classes
- 6 shader programs
- Complete MVC architecture

Đáng để trình bày như một dự án đồ họa máy tính toàn diện, kết hợp giữa kỹ thuật đồ họa và ứng dụng giáo dục!

---

## 13. Tóm tắt các file chính

1. `main.py` - entry point
2. `controller.py` - điều phối chính
3. `viewer.py` - OpenGL và UI
4. `model.py` - quản lý state
5. `libs/transform.py` - toán học đồ họa
6. `geometry/*/` - các lớp hình học
7. `shaders/` - GLSL programs
8. `components/` - UI panels
9. `core/` - GameObject system
10. `libs/shader.py` - shader wrapper
11. `libs/buffer.py` - VAO/VBO management
12. `libs/coordinate_system.py` - grid và axes
13. `libs/gizmo.py` - 3D manipulation
14. `geometry/sgd_visualizer.py` - SGD visualization
15. `standard.vert/.frag` là shader scene chính.
16. `Trackball` điều khiển camera.
17. `SGDVisualizer` vẽ mặt loss và đường đi của tối ưu.nsformGizmo` ke thua `BaseShape`.

Y tuong:

- move tool: ve 3 truc va 3 hinh non
- rotate tool: ve 3 vong tron
- scale tool: ve 3 truc va 3 cube nho

Ham quan trong:

- `_create_cube_vertices(...)`: tao mesh cube nho cho scale handle
- `_create_rotation_circles(...)`: tao 3 vong quay X/Y/Z
- `project_to_screen(...)`: bien diem 3D thanh toa do man hinh 2D
- `check_axis_selection(...)`: xem chuot dang gan truc nao
- `handle_mouse_press(...)`: ghi nhan truc dang chon
- `handle_mouse_drag(...)`:
  - rotate: doi goc chuot thanh `delta_deg`
  - move: chieu vector chuot len truc dang chon
  - scale: doi scale theo truc
- `handle_mouse_release()`: reset state
- `draw(...)`: ve hinh gizmo len scene

Noi de bao cao:

- ban khong dung ray casting 3D day du
- ma dung phep chieu 3D->2D roi do khoang cach tu chuot den doan thang/vong tron

## 6.6. `libs/lighting.py`

- `Light`: diffuse/specular/ambient/position
- `Material`: diffuse/specular/ambient/shininess
- `LightingManager`:
  - `setup_phong(...)`
  - `setup_gouraud(...)`

Trong project hien tai, nhom shape chu yeu set uniform thu cong theo `standard.vert/.frag`.
`LightingManager` duoc dung ro hon trong phan `SGDVisualizer`.

## 6.7. `libs/loss_functions.py`

Truu tuong:

- `OptimizationFunction`: interface `compute()` va `gradient()`

Ham cu the:

- `Himmelblau`
- `Rosenbrock`
- `Booth`
- `Quadratic`

Moi ham co:

- ten
- chuoi cong thuc de hien thi/ve
- mien gia tri hop ly
- `compute(x, y)`
- `gradient(x, y)`

---

## 7. Geometry

## 7.1. `geometry/base_shape.py`

File: [geometry/base_shape.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/geometry/base_shape.py)

Moi shape deu co:

- `position`
- `rotation`
- `scale`

Ham:

- `set_transform(...)`: nhan transform tu `GameObject`
- `set_color(...)`: de lop con override
- `get_transform_matrix()`: ghep `T * Rx * Ry * Rz * S`

Y nghia:

- bat ky mesh nao cung co the bien doi theo mot giao dien chung

## 7.2. Mau chung cua cac file shape

Gan nhu tat ca file trong `geometry/2d` va `geometry/3d` deu co cung bo khung:

1. `__init__`
   - luu shader
   - tao geometry
   - tao state:
     - `use_flat_color`
     - `flat_color`
     - `use_texture`
     - `texture_id`
     - `render_mode`
   - tao `VAO`, `Shader`, `UManager`
2. `setup`
   - bind VBO vao layout 0/1/2/3
3. `set_texture`
   - load anh bang PIL
   - tao OpenGL texture
4. `draw`
   - tinh `modelview`
   - upload `projection`, `modelview`, `view`
   - upload `u_use_flat_color`, `u_flat_color`
   - upload `u_use_texture`
   - upload `u_render_mode`
   - upload danh sach den `u_light_*`
   - ve bang `glDrawElements` hoac `glDrawArrays`
5. `set_color`
   - doi mau CPU
   - cap nhat lai color VBO
6. `set_solid_color`
   - bat che do override 1 mau
7. `cleanup`
   - giai phong tai nguyen GPU

Noi de bao cao:

- toan bo shape chia se mot giao ke render chung
- diem khac nhau chu yeu nam o "ham sinh geometry"

## 7.3. Nhom 2D

Thu muc: [geometry/2d](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/geometry/2d)

### `triangle2d.py`

- `_generate_triangle_geometry()` tao 3 dinh, normal `(0,0,1)`, UV tam giac
- ve bang `GL_TRIANGLES`

### `rectangle2d.py`

- `_generate_rectangle_geometry()` tao 4 dinh + `indices` de chia thanh 2 tam giac
- day la vi du indexed drawing

### `pentagon2d.py`

- tao 1 dinh tam + vong 5 dinh tren duong tron
- ve kieu `TRIANGLE_FAN`

### `hexagon2d.py`

- giong pentagon nhung 6 canh

### `circle2d.py`

- dung `segments`
- tao tam roi mau tren duong tron
- cang nhieu `segments` thi hinh cang tron

### `ellipse2d.py`

- giong circle nhung:
  - `x = a*cos(theta)`
  - `y = b*sin(theta)`

### `trapezoid2d.py`

- `_generate_trapezoid()` tao 4 dinh bat doi xung
- dung `indices` de chia 2 tam giac

### `star2d.py`

- xen ke dinh ngoai va dinh trong
- ve bang `TRIANGLE_FAN`

### `arrow2d.py`

- tao 7 dinh theo hinh mui ten
- cung ve bang `TRIANGLE_FAN`

Luu y:

- cac shape 2D van co normal va light uniform
- nhung ban chat nam tren mat phang Z=0

## 7.4. Nhom 3D

Thu muc: [geometry/3d](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/geometry/3d)

### `cube3d.py`

- 24 dinh, vi moi mat can normal rieng
- 6 mat x 4 dinh/mat
- `indices` chia moi mat thanh 2 tam giac
- UV lap lai cho 6 mat

### `tetrahedron3d.py`

- 4 dinh cua tu dien
- `_generate_tetrahedron_geometry()` tao 4 tam giac
- `_generate_texcoords()` dung spherical mapping

### `sphere_tetrahedron3d.py`

Y tuong:

- bat dau tu tetrahedron deu
- de quy chia moi tam giac thanh 4 tam giac nho
- chuan hoa diem ra mat cau

Ham quan trong:

- `normalize(v)`
- `_subdivide_with_colors(a, b, c, ..., n)`

Day la ky thuat xap xi sphere bang subdivision.

### `sphere_grid3d.py`

Y tuong:

- bat dau tu 6 mat cua cube
- chia moi mat thanh luoi
- chuan hoa moi diem ra mat cau

Uu diem:

- phan bo diem deu hon lat-long o vung cuc

### `sphere_latlong3d.py`

Y tuong:

- dung kinh do/vi do
- tao dinh theo:
  - `x = cos(lat) cos(lon)`
  - `y = sin(lat)`
  - `z = cos(lat) sin(lon)`

Uu diem:

- de hieu, de lap UV

Nhuoc diem:

- dinh tap trung o gan cuc

### `cylinder3d.py`

- tao vong dinh day va dinh tren
- them tam day, tam nap
- `indices` tao:
  - mat ben
  - day duoi
  - day tren

### `cone3d.py`

- 1 dinh chop
- 1 vong day
- 1 tam day
- normal mat ben duoc tinh theo do doc sinh non

### `truncated_cone3d.py`

- giong cone nhung co ban kinh tren khac 0
- `_add_cap(...)` tao day tren/day duoi
- ve theo triangle list da "trai phang"

### `torus3d.py`

Thong so hoa xuyen:

- vong lon ban kinh `R`
- vong nho ban kinh `r`
- `theta` di quanh vong lon
- `phi` di quanh ong nho

### `prism3d.py`

- thuc chat la "lang tru da giac deu"
- `sides` quyet dinh so canh day
- co than ben + nap tren + nap duoi

## 7.5. `math_surface3d.py`

File: [geometry/math_surface3d.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/geometry/math_surface3d.py)

Day la phan hay de bao cao vi noi ly thuyet toan voi do hoa.

### `__init__`

- nhan `func(x, y)`
- nhan mien `x_range`, `y_range`
- nhan `resolution`

### `_generate_surface()`

Lam cac buoc:

1. tao luoi gia tri `X_real`, `Y_real`
2. tinh `Z_real = f(X_real, Y_real)`
3. loai `nan`, `inf`, cat nguong
4. chuan hoa `X`, `Y`, `Z` ve khoang hien thi `[-2, 2]`
5. tao vertex
6. tinh normal bang xap xi dao ham rieng:
   - `dz/dx`
   - `dz/dy`
   - normal ~ `(-dz/dx, -dz/dy, 1)`
7. to mau theo do cao Z
8. tao `indices` de noi luoi thanh tam giac

Day la cho de noi:

- mat phang do cao `z = f(x, y)` duoc bien thanh triangle mesh

### `setup()`

- nap `vertices`, `colors`, `normals`, `indices`

### `draw()`

- giong shape 3D thuong

### `set_color()`

- neu `use_custom_color` thi dung 1 mau
- neu khong thi quay lai mau theo do cao ban dau

### `set_color_mode()`

- chuyen giua auto-color va custom-color

### `set_solid_color()`

- flat override

## 7.6. `model_loader3d.py`

File: [geometry/model_loader3d.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/geometry/model_loader3d.py)

### `__init__`

- neu co `filename` thi `load_model(filename)`
- neu khong thi tao cube mac dinh

### `_create_default_cube()`

- fallback khi file loi/khong ton tai

### `load_model()`

- xac dinh duoi file:
  - `.obj`
  - `.ply`

### `_load_obj()`

Doc theo dong:

- `v`: dinh
- `vn`: normal
- `f`: face

Sau do:

- doi face thanh danh sach `indices`
- neu khong co mau thi tu sinh mau trang
- tu sinh normal
- normalize model ve kich thuoc hop ly

### `_load_ply()`

Ho tro:

- `ascii`
- `binary_little_endian`/`binary_big_endian`

No doc header truoc de biet:

- so vertex
- so face
- property cua vertex

Sau do moi doc body.

### `_generate_normals()`

- duyet tung tam giac
- tinh `face_normal = cross(edge1, edge2)`
- cong don vao 3 dinh
- chuan hoa lai

Day la "vertex normal trung binh tu cac mat ke".

### `_generate_colors()`

- mac dinh toan bo mau trang

### `_generate_texcoords()`

- auto UV bang spherical mapping

### `_normalize_model()`

- dua tam model ve goc
- scale de fit vao vung nhin

Noi de bao cao:

- model loader khong phai viewer file thu dong
- no da parse va bien file thanh mesh OpenGL co the render ngay

## 7.7. `sgd_visualizer.py`

File: [geometry/sgd_visualizer.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/geometry/sgd_visualizer.py)

Day la cau noi giua AI optimization va do hoa.

### `__init__`

- nhan `loss_func`
- tao dict `optimizers`
- tao `trajectories`
- chon mau rieng cho tung optimizer
- goi `_generate_surface()`

### `_generate_surface()`

Lam 2 viec:

1. lay mau ham loss tren luoi `(x, y)`
2. bien no thanh mat mesh 3D de ve

Co them:

- chuan hoa truc ve khoang `[-2, 2]`
- luu `surface_x_scale`, `surface_y_scale`, `surface_z_scale`
- de sau nay doi diem optimizer tu khong gian toan hoc sang khong gian ve

### `setup()`

- tao VAO/shader cho:
  - mat surface
  - marker cau nho
  - trail line

### `add_optimizer()`

Luu state cho moi thuat toan:

- `position`
- `velocity`
- `loss`
- `gradient_mag`
- `step`
- `history`
- buffer cho `Momentum` va `Adam`

### `step_optimizer()`

Tinh gradient roi cap nhat vi tri tuy theo loai:

- `GD`: `x = x - lr * grad`
- `SGD`: dang de giong GD
- `MiniBatch`: dang de giong GD
- `Momentum`: them bo dem dong luong
- `Nesterov`: nhin truoc roi tinh gradient
- `Adam`: dung `m`, `v`, bias correction

Sau do:

- cat vi tri vao mien hop le
- tinh loss moi
- tang step
- them vao history

### `update_trajectory()`

- lay 2 diem cuoi trong history roi noi thanh doan

### `get_draw_coords()`

- doi `(x, y, loss)` tu he toa do toan hoc sang he toa do mat ve

### `draw()`

1. ve mat loss 3D
2. ve marker cau nho cho tung optimizer
3. ve trail lich su

### `_draw_sphere()`

- tao tam thoi 1 sphere nho tai vi tri hien tai cua optimizer

### `_draw_trail()`

- tao line segments noi cac diem lich su

### `reset_optimizer()`

- dua optimizer ve vi tri ban dau
- xoa state trung gian

---

## 8. Shader

Thu muc: [shaders](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/shaders)

### `color_interp.vert/.frag`

Dung cho:

- coordinate system
- cac mode color co ban
- trail line

Co che:

- vertex shader chi truyen mau dinh sang fragment shader
- fragment shader xuat mau noi suy

### `gouraud.vert/.frag`

Gouraud shading:

- tinh anh sang o vertex shader
- fragment shader chi xuat lai mau da noi suy

### `phong.vert/.frag`

Phong shading:

- vertex shader truyen `normal_interp`, `vertPos`, `colorInterp`
- fragment shader moi tinh diffuse/specular theo tung pixel
- ho tro them `depth map mode`

### `standard.vert/.frag`

Day moi la cap shader chinh duoc scene editor dung nhieu nhat.

Ho tro:

- color thuong
- texture
- flat color override
- Gouraud (`u_render_mode == 1`)
- Phong (`u_render_mode == 2`)
- rainbow interpolation (`u_render_mode == 3`)
- nhieu den cung luc
- attenuation theo khoang cach
- depth map mode

Noi de noi:

- `standard` la "super shader"
- con `phong/gouraud/color_interp` la shader rieng cho cac use case cu the

---

## 9. Noi ly thuyet voi code

### MVP

- `Model`: `BaseShape.get_transform_matrix()`
- `View`: `Trackball.view_matrix()`
- `Projection`: `Trackball.projection_matrix()`

Trong `draw()` cua shape:

- `final_model = object_transform @ model`
- `modelview = view @ final_model`
- `gl_Position = projection * modelview * vec4(position, 1.0)`

### Normal

- can de tinh anh sang
- xuat hien o:
  - tu sinh trong shape 3D
  - tinh bang dao ham trong `math_surface3d.py`
  - tinh bang cross product trong `model_loader3d.py`

### UV

- de dan texture
- shape don gian tu sinh UV truc tiep
- nhieu shape 3D dung spherical mapping

### VAO/VBO/EBO

- `setup()` cua moi shape la noi day du lieu len GPU

### Depth Test

- bat trong `Viewer.__init__`
- giup vat gan che vat xa

### Wireframe / Fill / Point

- doi bang `glPolygonMode`

### RGB / Depth Map

- `Viewer` dua `u_display_mode` va `u_cam_far` vao shader
- shader doi mau thanh thang xam theo do sau

### Flat Color

- bo qua lighting/texture
- tra thang 1 mau duy nhat

---

## 10. Goi y cach trinh bay voi thay

Neu can noi trong 5-7 phut, co the theo thu tu nay:

1. Muc tieu de tai:
   - mini engine do hoa + SGD visualizer
2. Kien truc:
   - `main -> controller -> model/view`
3. Main loop:
   - input -> UI -> update -> render
4. Scene system:
   - `GameObject`, `Scene`, `Hierarchy`, `Inspector`
5. Render pipeline:
   - geometry -> VBO/VAO -> shader -> MVP -> rasterization
6. Shader:
   - solid, Gouraud, Phong, texture, multi-light, depth map
7. Hinh hoc:
   - 2D co ban, 3D parametric, math surface, model loader
8. Tuong tac:
   - trackball, camera, gizmo
9. SGD:
   - ve mat loss + marker + trajectory cua tung optimizer

---

## 11. Mot so diem de phong thu neu bi hoi sau

- Tai sao dung `selected_category = 5`?
  - do day la normal scene editor mode, tach khoi preview mode.

- Tai sao co nhieu shader?
  - vi moi shader phuc vu mot muc dich:
    - don gian
    - lighting per-vertex
    - lighting per-fragment
    - super shader cho scene editor

- Tai sao nhieu shape co code giong nhau?
  - vi chung chia se giao ke render chung, chi khac cach sinh geometry.

- Tai sao can normal?
  - de tinh huong be mat so voi huong sang va huong nhin.

- Tai sao can UV?
  - de anh 2D biet map vao diem nao tren be mat 3D.

- Toi uu va do hoa lien quan gi?
  - do hoa dung de bien ham mat mat 2 bien thanh be mat 3D, sau do ve quy dao hoi tu cua optimizer.

---

## 12. Diem nen tu hoc gap toi nay

Neu chi co it thoi gian, uu tien nho 10 y nay:

1. `main.py` chi khoi dong, `controller.run()` moi la trung tam.
2. `Viewer` lo window + render + UI.
3. `Model` giu state + tao drawable.
4. `Scene` la danh sach `GameObject`.
5. Moi shape co `setup()` va `draw()`.
6. `setup()` = nap du lieu len GPU, `draw()` = truyen uniform va goi lenh ve.
7. MVP = `Projection * View * Model`.
8. `standard.vert/.frag` la shader scene chinh.
9. `Trackball` dieu khien camera.
10. `SGDVisualizer` ve mat loss va duong di cua toi uu.
