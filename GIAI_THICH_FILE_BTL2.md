# Giải Thích Các File Mới Của BTL 2

Tài liệu này chỉ tập trung vào **các file mình mới thêm hoặc mới nối vào** để làm phần `BTL 2`.
Mục tiêu là để bạn nhìn repo đỡ rối và biết:

- file nào là lõi thật sự
- file nào là file hỗ trợ chạy / kiểm tra / demo
- file nào chỉ là tài liệu
- file nào có thể bỏ

## 1. Nhóm file thực sự đang dùng khi bấm `BTL 2` trong app

### [components/btl2_panel.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/components/btl2_panel.py)
- Đây là panel mới xuất hiện khi bạn vào menu `BTL 2`.
- Nó cho bạn chọn:
  - dùng scene hiện tại của BTL 1 hay dùng scene procedural demo
  - config path
  - output path
  - số frame
  - seed
- Nó không tự generate gì cả.
- Nó chỉ dựng UI và trả `actions` về cho `controller`.

### [controller.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/controller.py)
- Đây không phải file mới hoàn toàn, nhưng mình đã **nối thêm logic BTL 2** ở đây.
- Nó bắt các action như:
  - `btl2_sync_config`
  - `btl2_refresh_scene`
  - `btl2_generate`
- Đây là nơi quyết định:
  - nếu đang ở chế độ `current_scene` thì xuất từ scene BTL 1
  - nếu đang ở chế độ `procedural_demo` thì chạy generator procedural

### [model.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/model.py)
- Đây cũng là file cũ nhưng mình đã thêm state cầu nối cho BTL 2.
- Những biến mới quan trọng:
  - `btl2_config_path`
  - `btl2_output_dir`
  - `btl2_num_frames`
  - `btl2_seed`
  - `btl2_source_mode`
  - `btl2_last_status`
- Những hàm mới quan trọng:
  - `sync_btl2_config()`
  - `refresh_btl2_scene_summary()`
  - `run_btl2_generator()`
  - `run_btl2_from_current_scene()`

### [viewer.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/viewer.py)
- File cũ nhưng đã được nối để khi `selected_category == 6` thì mở `BTL2Panel`.
- Nó không generate dataset.
- Nó chỉ hiển thị panel và truyền action về controller.

### [components/main_menu.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/components/main_menu.py)
- File cũ nhưng đã sửa menu `BTL 2`.
- Bây giờ menu này không còn trỏ nhầm sang `SGD` nữa.
- Nó chuyển đúng sang mode `BTL 2`.

## 2. Nhóm file lõi của pipeline BTL 2

### [btl2/app.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/app.py)
- Đây là file lõi nhất của BTL 2.
- Nó làm vai trò “điều phối pipeline export dataset”.
- Những việc nó làm:
  - build scene
  - render RGB
  - render depth
  - render segmentation
  - tính bbox
  - ghi file output
  - ghi YOLO / COCO / metadata
- Nó cũng có nhánh bridge từ scene BTL 1:
  - `generate_from_btl1_scene(...)`

### [btl2/main.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/main.py)
- Đây là CLI riêng cho BTL 2 nếu bạn muốn chạy độc lập, không qua app BTL 1.
- Ví dụ:
  - `python -m btl2.main --config configs/btl2/default.yaml generate`
- Nếu bạn chỉ dùng app chính `main.py` ở gốc repo thì file này không bắt buộc mỗi lần chạy, nhưng **không phải file thừa**.

## 3. Nhóm renderer OpenGL của BTL 2

### [btl2/renderer/window.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/renderer/window.py)
- Tạo hidden GLFW window khi BTL 2 chạy độc lập.
- Nếu BTL 2 được gọi từ BTL 1 thì giờ nó sẽ tái sử dụng context hiện có.

### [btl2/renderer/camera.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/renderer/camera.py)
- Chuyển `CameraState` thành:
  - `view matrix`
  - `projection matrix`
- Đây là cầu nối giữa camera logic và camera dùng để render.

### [btl2/renderer/shader.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/renderer/shader.py)
- Compile shader GLSL và upload uniform.
- Đây là lớp wrapper gọn cho shader program.

### [btl2/renderer/mesh.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/renderer/mesh.py)
- Upload mesh từ CPU sang GPU thành VAO/VBO/EBO.
- Khi render pass cần vẽ object thì gọi `mesh.draw()`.

### [btl2/renderer/material.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/renderer/material.py)
- Chứa màu cơ bản và màu segmentation của object.
- File nhỏ nhưng vẫn có ý nghĩa.

### [btl2/renderer/framebuffer.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/renderer/framebuffer.py)
- Đây là file rất quan trọng.
- Nó tạo offscreen framebuffer để render không cần hiện lên màn hình.
- BTL 2 dùng nó để xuất:
  - RGB
  - segmentation
  - depth

### [btl2/renderer/render_pass_rgb.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/renderer/render_pass_rgb.py)
- Render ảnh màu bình thường.

### [btl2/renderer/render_pass_depth.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/renderer/render_pass_depth.py)
- Render depth pass.

### [btl2/renderer/render_pass_segmentation.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/renderer/render_pass_segmentation.py)
- Render segmentation bằng màu riêng cho từng instance.

## 4. Nhóm scene generation

### [btl2/scene/scene.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/scene/scene.py)
- Chứa dữ liệu tổng của một scene:
  - camera
  - light
  - list object

### [btl2/scene/scene_object.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/scene/scene_object.py)
- Chứa dữ liệu của một object trong scene BTL 2:
  - transform
  - class
  - instance id
  - màu
  - aabb

### [btl2/scene/randomizer.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/scene/randomizer.py)
- Wrapper nhỏ cho random generator.
- Mục đích là giữ deterministic bằng seed.

### [btl2/scene/lighting.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/scene/lighting.py)
- Sinh ánh sáng ngẫu nhiên theo config.

### [btl2/scene/camera_rig.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/scene/camera_rig.py)
- Sinh dashcam camera procedural.
- Khi chạy theo scene BTL 1 thì file này ít tham gia hơn.

### [btl2/scene/object_loader.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/scene/object_loader.py)
- Load `.obj`
- load `.ply` ASCII tối thiểu
- nếu thiếu asset thì fallback về primitive
- Đây là file rất quan trọng vì đề yêu cầu external model loading.

### [btl2/scene/road_scene_builder.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/scene/road_scene_builder.py)
- Dùng khi chạy procedural demo.
- Nó tự đặt road, xe, người, biển báo, đèn giao thông.
- Nếu bạn chạy từ scene BTL 1 thì file này không phải đường chính, nhưng **vẫn là file có ích** vì nó là chế độ fallback/demo của BTL 2.

## 5. Nhóm annotations

### [btl2/annotations/bbox.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/annotations/bbox.py)
- Tính bounding box 2D từ AABB 3D bằng phép chiếu camera.

### [btl2/annotations/occlusion.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/annotations/occlusion.py)
- Ước lượng mức occlusion đơn giản từ segmentation mask.

### [btl2/annotations/segmentation.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/annotations/segmentation.py)
- Lập bảng map giữa màu trong mask và `instance_id/class`.

### [btl2/annotations/depth_export.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/annotations/depth_export.py)
- Linearize depth buffer và ghi depth PNG / NPY.

### [btl2/annotations/yolo_export.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/annotations/yolo_export.py)
- Ghi label YOLO.
- Ghi luôn `dataset.yaml`.

### [btl2/annotations/coco_export.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/annotations/coco_export.py)
- Ghi file COCO train/val.

### [btl2/annotations/metadata_export.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/annotations/metadata_export.py)
- Ghi metadata JSON cho từng frame.

### [btl2/annotations/validators.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/annotations/validators.py)
- Kiểm tra dataset:
  - đủ file hay chưa
  - YOLO hợp lệ chưa
  - COCO hợp lệ chưa
  - mask có foreground chưa

## 6. Nhóm utility

### [btl2/utils/io.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/utils/io.py)
- Đọc YAML config
- tạo output tree
- ghi JSON
- có fallback parser để không bị crash khi máy thiếu `PyYAML`

### [btl2/utils/image.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/utils/image.py)
- Ghi RGB / grayscale / mask ra file ảnh

### [btl2/utils/math3d.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/utils/math3d.py)
- Chứa toán 3D:
  - perspective
  - look_at
  - compose transform
  - project points
  - AABB

### [btl2/utils/colors.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/utils/colors.py)
- Định nghĩa màu class và màu instance segmentation.

### [btl2/utils/constants.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/utils/constants.py)
- Hằng số tên lớp, folder output, extension.

### [btl2/utils/logging_utils.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/btl2/utils/logging_utils.py)
- Chỉ dùng cho CLI/script.
- Không ảnh hưởng đến app chính.

## 7. Nhóm config

### [configs/btl2/default.yaml](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/configs/btl2/default.yaml)
- Config chuẩn của BTL 2.
- App BTL 1 đang đọc file này khi bạn vào tab `BTL 2`.

### [configs/btl2/demo_small.yaml](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/configs/btl2/demo_small.yaml)
- Config nhẹ hơn để test nhanh.
- Chủ yếu dùng cho script demo.

## 8. Nhóm shader

### [shaders/rgb.vert](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/shaders/rgb.vert)
### [shaders/rgb.frag](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/shaders/rgb.frag)
- Shader cho RGB pass.

### [shaders/depth.vert](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/shaders/depth.vert)
### [shaders/depth.frag](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/shaders/depth.frag)
- Shader cho depth pass.

### [shaders/seg.vert](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/shaders/seg.vert)
### [shaders/seg.frag](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/shaders/seg.frag)
- Shader cho segmentation pass.

## 9. Nhóm script chạy ngoài

### [scripts/generate_demo_dataset.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/scripts/generate_demo_dataset.py)
- Script chạy nhanh để sinh dataset demo bằng config nhẹ.

### [scripts/validate_dataset.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/scripts/validate_dataset.py)
- Script kiểm tra dataset sau khi generate.

### [scripts/visualize_annotations.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/scripts/visualize_annotations.py)
- Script vẽ bbox preview và contact sheet.

## 10. Nhóm file tài liệu / placeholder

### [assets/placeholders/README.md](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/assets/placeholders/README.md)
- Chỉ là file ghi chú hướng dẫn thay asset thật.
- Không tham gia chạy code.
- Nhưng vẫn có ích cho repo GitHub.

### [README.md](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/README.md)
- Mình đã sửa để giải thích mối liên hệ giữa BTL 1 và BTL 2.

### [requirements.txt](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/requirements.txt)
- Danh sách thư viện cần thiết cho phần repo mới.

### [.gitignore](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/.gitignore)
- Giữ cho repo không commit rác như output lớn, cache, pyc.

## 11. File mình đã xóa bớt

### `outputs/btl2/demo_dataset/README.md`
- File này chỉ là placeholder để giữ cấu trúc thư mục.
- Code không dùng tới.
- Mình đã xóa để repo bớt rối.

## 12. Kết luận ngắn

Nếu bạn chỉ quan tâm phần **chạy thật**, hãy tập trung vào 4 cụm:

1. `components/btl2_panel.py`, `controller.py`, `model.py`, `viewer.py`
   Đây là cầu nối giữa app BTL 1 và BTL 2.

2. `btl2/app.py`
   Đây là file điều phối chính của BTL 2.

3. `btl2/renderer/*`, `shaders/*`
   Đây là phần render OpenGL đa pass.

4. `btl2/annotations/*`
   Đây là phần xuất dữ liệu huấn luyện.

Phần còn lại chủ yếu là:
- config
- helper
- script chạy ngoài
- file tài liệu
