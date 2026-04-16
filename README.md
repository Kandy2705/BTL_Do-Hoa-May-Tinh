# BTL_DHMT - Đồ họa máy tính: BTL 1 + BTL 2

## 1. Giới thiệu dự án
Dự án này gộp **hai bài tập lớn** trong cùng một codebase:

- **BTL 1**: ứng dụng đồ họa tương tác kiểu Unity mini, cho phép tạo scene, thêm object 2D/3D, mathematical surface, model `.obj/.ply`, đèn, camera, chỉnh transform/material/shader trực tiếp trên giao diện.
- **BTL 2**: pipeline sinh **synthetic dataset** cho bài toán computer vision từ scene đồ họa, bao gồm:
  - ảnh **RGB**
  - **depth map**
  - **segmentation mask**
  - **ground-truth bounding boxes**
  - nhãn **YOLO**
  - nhãn **COCO**
  - **metadata JSON** và **CSV custom**
  - panel **YOLO inference** ngay trong UI để test model `.pt`

Mục tiêu của project là biến một scene đồ họa thành một pipeline hoàn chỉnh để:

1. dựng scene trực quan trên UI,
2. xuất dataset nhiều định dạng,
3. train detector YOLO,
4. nạp lại weight vào UI để test nhận diện.

---

## 2. Dự án làm được gì?

### BTL 1
- Tạo và chỉnh sửa scene tương tác.
- Thêm object 2D, 3D, surface toán học, model từ file.
- Thêm light và camera trong scene.
- Chỉnh `Position / Rotation / Scale`.
- Chỉnh màu, shader, texture, flat shading, shininess.
- Có demo phần tối ưu SGD và lab quaternion SLERP.

### BTL 2
- Tạo road-scene procedural bằng asset thật trong `assets/models/`.
- Cho phép dùng **scene hiện tại của BTL 1** hoặc **procedural demo** làm nguồn sinh dữ liệu.
- Export dataset với các thư mục chuẩn:
  - `images/`
  - `depth/`
  - `masks/`
  - `labels_yolo/`
  - `annotations_coco/`
  - `annotations_custom/`
  - `metadata/`
  - `previews/`
- COCO hỗ trợ `segmentation` theo **polygon** hoặc **RLE**.
- Có panel preview trực tiếp trong UI cho:
  - `RGB`
  - `Depth`
  - `Mask`
  - `GT Boxes`
- Có panel YOLO inference cho phép:
  - nạp pretrained model
  - nạp fine-tuned model
  - chọn ảnh test
  - chỉnh `confidence`
  - chỉnh `image size`
  - chạy inference và xem box trực tiếp trong UI.

---

## 3. Các lớp object chính của dataset
Các class detection chính đang dùng trong bài:

- `person`
- `car`
- `bus`
- `truck`
- `motorbike`
- `traffic_light`
- `traffic_sign`

Ngoài ra project còn có object nền/phụ trợ như:

- `road`
- `city`

Lưu ý:
- `road` là object nền để render scene, không phải class detect chính trong file YOLO.
- `city` hiện dùng để dựng bối cảnh giao lộ / môi trường nền.

---

## 4. Cấu trúc thư mục

```text
BTL_DHMT/
├── README.md
├── main.py                         # Entry chính để chạy UI
├── controller.py                   # Điều phối action từ UI
├── model.py                        # State ứng dụng, scene, bridge BTL2
├── viewer.py                       # Render loop / viewport chính
├── requirements.txt
│
├── assets/
│   └── models/                     # Toàn bộ OBJ/MTL/texture thật
│
├── components/                     # UI panels
│   ├── main_menu.py
│   ├── hierarchy_panel.py
│   ├── inspector_panel.py
│   └── btl2_panel.py
│
├── btl1/                           # Wrapper / phần bài 1
├── btl2/                           # Package bài 2
│   ├── app.py                      # Pipeline sinh dataset
│   ├── main.py                     # CLI cho BTL2
│   ├── annotations/                # bbox / yolo / coco / metadata / validator
│   ├── renderer/                   # RGB / depth / segmentation passes
│   ├── scene/                      # procedural builder, camera, object loader
│   └── utils/
│
├── configs/
│   └── btl2/
│       ├── default.yaml
│       ├── demo_small.yaml
│       ├── showcase_full.yaml
│       └── showcase_finetune.yaml
│
├── scripts/
│   ├── generate_demo_dataset.py
│   ├── validate_dataset.py
│   ├── visualize_annotations.py
│   ├── build_showcase_finetune_dataset.py
│   ├── repair_btl2_scene_labels.py
│   └── train_yolo.py
│
├── outputs/
│   ├── btl2/                       # dataset export
│   └── training/yolo/              # pretrained + fine-tuned models
│
├── shaders/
│   ├── standard.vert/.frag
│   └── btl2/
│
├── core/
├── geometry/
├── libs/
└── docs/
```

---

## 5. Yêu cầu môi trường
Khuyến nghị:

- Python `3.10+`
- macOS hoặc môi trường có OpenGL + GLFW
- máy có GPU càng tốt nếu muốn train nhanh

Package trong [requirements.txt](requirements.txt):

- `PyOpenGL`
- `glfw`
- `numpy`
- `Pillow`
- `PyYAML`
- `ultralytics`

Cài đặt:

```bash
pip3 install -r requirements.txt
```

---

## 6. Cách chạy dự án

### 6.1. Chạy UI chính
```bash
python3 main.py
```

Hoặc:

```bash
python3 -m btl1.main
```

### 6.2. Chạy BTL2 bằng CLI
Sinh dataset từ config:

```bash
python3 -m btl2.main --config configs/btl2/default.yaml generate
```

Preview 1 frame:

```bash
python3 -m btl2.main --config configs/btl2/showcase_full.yaml preview
```

Override số frame / output / seed:

```bash
python3 -m btl2.main \
  --config configs/btl2/showcase_full.yaml \
  generate \
  --frames 20 \
  --output-dir outputs/btl2/my_dataset \
  --seed 42
```

---

## 7. Giải thích giao diện tổng quát
UI chính gồm 3 vùng lớn:

- **Hierarchy** bên trái: danh sách object trong scene.
- **Viewport** ở giữa: vùng hiển thị scene.
- **Inspector + BTL2 Panel** bên phải: chỉnh object và chạy pipeline dataset/inference.

---

## 8. Menu trên cùng

### 8.1. `File`
- `Import Model`: chọn file model `.obj/.ply`.
- `Exit`: thoát ứng dụng.

### 8.2. `BTL 1`
- `Normal Mode`: về chế độ dựng scene tự do.
- `Center Scene For Demo`: căn camera/scene để demo dễ nhìn hơn.
- `2D Shapes`: thêm shape 2D preview.
- `3D Shapes`: thêm shape 3D preview.
- `Mathematical Surface`: preview mặt toán học `z=f(x,y)`.
- `Model from file`: preview model từ file.
- `Optimization (SGD)`: vào phần demo tối ưu.

### 8.3. `BTL 2`
- `Generate Demo Dataset`: chạy nhanh luồng sinh dataset BTL2.

### 8.4. `Lab`
- `Sphere Z 180° SLERP Loop`: bật/tắt animation quaternion.
- `Rescan Sphere Targets`: quét lại các target sphere.

---

## 9. Hierarchy panel
Khi ở **Normal Mode**, bạn có thể **click chuột phải trong Hierarchy** để mở menu thêm object.

### 9.1. `Add 2D Object`
Thêm các object 2D như:
- Triangle
- Rectangle
- Pentagon
- Hexagon
- Circle
- Ellipse
- Trapezoid
- Star
- Arrow

### 9.2. `Add 3D Object`
Thêm object 3D primitive trong hệ BTL1.

### 9.3. `Add Mathematical Surface`
- `Z = f(x,y)`

### 9.4. `Add Model from file`
Thêm model `.obj/.ply` bất kỳ.

### 9.5. `Add Default Model`
Đây là menu quan trọng để demo nhanh các asset thật có sẵn trong `assets/models/` mà **không cần import tay**.

Hiện đang có các mục:
- `Road`
- `City / Intersection`
- `Person`
- `Car / Taxi`
- `Bus`
- `Truck`
- `Motorbike`
- `Traffic Light`
- `Traffic Sign`

### Lưu ý về `Road`
`Road` hiện tại **không còn bị scale kéo dài một khối duy nhất** nữa.
Khi thêm `Road`, hệ thống sẽ tự sinh ra **2 mảnh road ghép nối** từ model:

- `assets/models/road_props/old road.obj`

Cách này giúp texture không bị vỡ khi kéo dài quá mức.

### 9.6. `Add Light`
Thêm light object vào scene.

### 9.7. `Add Camera`
Thêm camera object vào scene.

### 9.8. Right click vào object trong tree
- `Delete`: xóa object.

---

## 10. Inspector panel
Inspector chỉ hoạt động đầy đủ khi ở **Normal Mode**.

### 10.1. `Transform`
Chỉnh:
- `Position`
- `Rotation`
- `Scale`

### 10.2. `Mesh Renderer`
Áp dụng cho object không phải camera/light.

Các thành phần chính:

- `Shader`: đổi shader/render mode.
- `Color`: đổi màu object.
- `Texture -> Browse`: chọn texture file.
- `Texture -> Clear`: xóa texture hiện tại.
- `Flat Shading`: bật/tắt flat shading.
- `Shininess`: độ bóng specular.

### 10.3. `Math Script`
Chỉ hiện với mathematical surface.

- nhập biểu thức `z = f(x,y)`
- bấm `Apply`

### 10.4. `Camera Settings`
Chỉ hiện với camera object.

- `FOV`
- `Near`
- `Far`

### 10.5. `Light Settings`
Chỉ hiện với light object.

- `Intensity`
- `Color`

---

## 11. BTL2 Dataset Builder panel
Đây là panel quan trọng nhất của BTL2. Nó được chia thành 6 phần.

### 11.1. `Pipeline status`
Hiển thị trạng thái tổng quát:
- `IDLE`
- `RUNNING`
- `DONE`
- `FAILED`

### 11.2. `1) Source`
Chọn nguồn scene để sinh dataset:

- `Use current BTL1 scene`
  - dùng scene bạn đang dựng trong BTL1
  - yêu cầu phải có ít nhất 1 camera object và 1 renderable object
- `Use procedural road demo`
  - dùng scene procedural của BTL2
  - render offscreen để xuất dataset

#### Nút trong phần Source
- `Refresh Scene Summary`
  - cập nhật số camera và số renderable trong scene hiện tại
- `Load Procedural Preview To BTL1 Scene`
  - đưa một frame procedural từ BTL2 sang scene BTL1 để xem trực tiếp trong viewport

### 11.3. `2) Config & Output`
Các trường chính:
- `Config file`
- `Output folder`
- `Frames`
- `Seed`

#### Nút
- `Sync From YAML`
  - đọc lại config từ file `.yaml`
- `Generate Dataset`
  - bắt đầu export dataset nếu validation hợp lệ

### 11.4. `3) Validation`
Kiểm tra trước khi generate.
Ví dụ các lỗi có thể xuất hiện:
- config path rỗng
- output folder rỗng
- frames <= 0
- current scene chưa có camera
- current scene chưa có renderable object

Nếu hợp lệ sẽ hiện:
- `Ready to generate.`

### 11.5. `4) Result`
Hiển thị kết quả sau khi generate:
- số frame đã sinh
- thư mục output
- sample RGB
- sample depth

### 11.6. `5) Preview`
Cho phép xem trực tiếp file dataset đã xuất.

Các tab preview:
- `RGB`
- `Depth`
- `Mask`
- `GT Boxes`

#### Ý nghĩa từng tab
- `RGB`: ảnh màu render thật
- `Depth`: depth map đã normalize để dễ nhìn
- `Mask`: segmentation mask preview có màu dễ đọc
- `GT Boxes`: bounding box ground-truth lấy từ metadata/mask, **không phụ thuộc YOLO inference**

#### Nút
- `Refresh Preview`
  - render lại preview đúng theo file dataset hiện tại

### 11.7. `6) YOLO Inference`
Cho phép nạp model `.pt` và test nhận diện ngay trên UI.

#### Trường
- `Weight file`: đường dẫn tới file `.pt`
- `Image file`: ảnh cần inference
- `Confidence`: ngưỡng confidence
- `Image size`: kích thước input cho model

#### Quick Select
Hiện tại có các preset nhanh:
- `YOLOv8s`
- `YOLOv8m`
- `YOLOv8x`
- `YOLO26s`
- `Fine-tuned`

#### Nút
- `Latest best.pt`
  - tự chọn `best.pt` fine-tuned mới nhất trong `outputs/training/yolo/`
- `Browse Weight`
  - chọn file `.pt` thủ công
- `Use Sample Image`
  - tự gợi ý một ảnh từ dataset hiện tại
- `Browse Image`
  - chọn ảnh test thủ công
- `Load Detector`
  - nạp model vào bộ nhớ
- `Run Inference`
  - chạy inference và hiển thị preview box

#### Dòng trạng thái
- `Loaded:`: model đang được nạp
- `Detections:`: tóm tắt các object vừa detect
- `Preview:`: đường dẫn ảnh preview sau inference

---

## 12. Các model YOLO đang có sẵn trong repo

### 12.1. Pretrained models
Trong thư mục `outputs/training/yolo/pretrained/` hiện có:

- `yolov8s.pt`
- `yolov8m.pt`
- `yolov8x.pt`
- `yolo26n.pt`
- `yolo26s.pt`

Các file này là **model gốc pretrained** để dùng nhanh hoặc làm baseline.

### 12.2. Fine-tuned model
Model fine-tuned chính đang ưu tiên mặc định:

- `outputs/training/yolo/unity_2400_yolov8s_640/weights/best.pt`

Khác biệt:
- `pretrained`: model chung có sẵn
- `fine-tuned`: model đã học trên dataset của project này

---

## 13. Dataset output gồm những gì?
Ví dụ một dataset trong `outputs/btl2/showcase_dataset/` sẽ có:

```text
outputs/btl2/showcase_dataset/
├── images/
│   ├── train/
│   └── val/
├── depth/
│   ├── train/
│   └── val/
├── masks/
│   ├── train/
│   └── val/
├── labels_yolo/
│   ├── train/
│   └── val/
├── annotations_coco/
│   ├── train.json
│   └── val.json
├── annotations_custom/
│   ├── train.csv
│   └── val.csv
├── metadata/
│   ├── train/
│   └── val/
├── previews/
└── dataset.yaml
```

### Ý nghĩa từng thư mục
- `images/`: ảnh RGB
- `depth/`: depth `.png` và có thể có thêm `.npy`
- `masks/`: segmentation mask
- `labels_yolo/`: label YOLO `.txt`
- `annotations_coco/`: file COCO `.json`
- `annotations_custom/`: CSV custom
- `metadata/`: JSON chi tiết theo frame
- `previews/`: ảnh preview để xem nhanh
- `dataset.yaml`: file cấu hình train YOLO

---

## 14. COCO export hiện hỗ trợ gì?
COCO hiện tại không còn để `segmentation: []` nữa.

Hệ thống hỗ trợ:
- `polygon`
- `rle`

Thiết lập trong YAML:

```yaml
annotations:
  coco_segmentation_mode: polygon
```

hoặc:

```yaml
annotations:
  coco_segmentation_mode: rle
```

Mặc định đang dùng:
- `polygon`

Segmentation COCO được suy ra trực tiếp từ ảnh mask đã render.

---

## 15. Các file config BTL2 quan trọng

### `configs/btl2/default.yaml`
Config mặc định cho sinh dataset road scene.

### `configs/btl2/demo_small.yaml`
Config demo nhỏ, chạy nhanh.

### `configs/btl2/showcase_full.yaml`
Config showcase để demo đủ class.

### `configs/btl2/showcase_finetune.yaml`
Config dùng cho bộ dữ liệu fine-tune / showcase nâng cao.

Các tham số thường chỉnh:
- `output_dir`
- `image_width`, `image_height`
- `num_frames`
- `train_split`
- `save_depth_npy`
- `camera.*`
- `scene.*`
- `lighting.*`
- `annotations.*`

---

## 16. Cách chạy các script tiện ích

### 16.1. Generate demo dataset
```bash
python3 scripts/generate_demo_dataset.py
```

### 16.2. Validate dataset
```bash
python3 scripts/validate_dataset.py outputs/btl2/showcase_dataset
```

### 16.3. Visualize annotations
```bash
python3 scripts/visualize_annotations.py outputs/btl2/showcase_dataset
```

### 16.4. Build showcase fine-tune dataset
```bash
python3 scripts/build_showcase_finetune_dataset.py \
  --source outputs/btl2/showcase_dataset \
  --output outputs/btl2/showcase_finetune_dataset
```

### 16.5. Train YOLO
Ví dụ train trên `unity_dataset`:

```bash
python3 scripts/train_yolo.py \
  --data outputs/btl2/unity_dataset \
  --model yolov8s.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 8 \
  --project outputs/training/yolo \
  --name unity_dataset_yolov8s_640
```

Ví dụ train trên nhiều dataset cùng lúc:

```bash
python3 scripts/train_yolo.py \
  --data outputs/btl2/unity_dataset outputs/btl2/showcase_dataset \
  --model yolov8s.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 8
```

---

## 17. Workflow khuyến nghị

### Workflow A - Dùng hoàn toàn trong UI
1. Chạy `python3 main.py`
2. Vào `BTL 1 -> Normal Mode`
3. Thêm object qua Hierarchy
4. Thêm ít nhất 1 camera
5. Vào panel `BTL2 Dataset Builder`
6. Chọn source:
   - `Use current BTL1 scene`
   - hoặc `Use procedural road demo`
7. Chọn config + output
8. `Generate Dataset`
9. Xem `RGB / Depth / Mask / GT Boxes`
10. Chọn model `.pt` trong `YOLO Inference`
11. `Load Detector`
12. `Run Inference`

### Workflow B - Sinh dataset bằng CLI rồi test trên UI
1. Generate dataset bằng `python3 -m btl2.main ... generate`
2. Chạy `python3 main.py`
3. Mở panel BTL2
4. `Use Sample Image`
5. Chọn `Fine-tuned` hoặc pretrained
6. Chạy inference

### Workflow C - Train detector rồi quay lại UI
1. Export dataset
2. Validate dataset
3. Train YOLO bằng `scripts/train_yolo.py`
4. Lấy `best.pt`
5. Trong UI chọn `Fine-tuned` hoặc browse tới weight đó
6. `Load Detector` -> `Run Inference`

---

## 18. Phân biệt GT Boxes và YOLO Boxes
Đây là điểm rất hay bị nhầm khi demo.

### GT Boxes
- được sinh từ pipeline dataset
- là **ground truth**
- dùng để xuất YOLO/COCO/metadata
- chính xác hơn cho mục đích dataset

### YOLO Inference Boxes
- là box model AI dự đoán
- có thể sai, thiếu, dư, lệch
- chỉ dùng để demo chất lượng detector

Kết luận:
- **dataset export** nên tin `GT Boxes`
- **panel inference** chỉ để test model

---

## 19. Ghi chú về asset road
Road hiện được xử lý theo hướng:
- dùng đúng model trong `assets/models/road_props/old road.obj`
- ghép **2 tile road** thay vì scale một mesh quá dài

Lý do:
- tránh texture stretching
- tránh hiện tượng road bị vỡ khi demo gần camera
- giúp preview scene và dataset đồng nhất hơn

---

## 20. Các file/đường dẫn quan trọng

### Config
- `configs/btl2/default.yaml`
- `configs/btl2/showcase_full.yaml`
- `configs/btl2/showcase_finetune.yaml`

### Model pretrained
- `outputs/training/yolo/pretrained/yolov8s.pt`
- `outputs/training/yolo/pretrained/yolov8m.pt`
- `outputs/training/yolo/pretrained/yolov8x.pt`
- `outputs/training/yolo/pretrained/yolo26s.pt`

### Model fine-tuned
- `outputs/training/yolo/unity_2400_yolov8s_640/weights/best.pt`

### Asset road
- `assets/models/road_props/old road.obj`
- `assets/models/road_props/old road.mtl`
- `assets/models/road_props/road.dds`

---

## 21. Lưu ý khi demo / bảo vệ
- Nếu cần box chuẩn để nói về dataset: dùng `GT Boxes`.
- Nếu cần nói về chất lượng AI: dùng `YOLO Inference`.
- Nếu model pretrained nhận sai class, đó là bình thường vì nó học trên domain chung, không phải asset riêng của project.
- Nếu muốn detection khớp nhất với scene của bạn, hãy dùng `Fine-tuned`.
- Nếu COCO bị hỏi về segmentation, hiện tại project đã hỗ trợ **polygon** và **RLE**.

---

## 22. Tóm tắt ngắn
Dự án này là một hệ thống hoàn chỉnh gồm:

- một editor scene đồ họa tương tác,
- một pipeline sinh synthetic dataset,
- nhiều định dạng annotation chuẩn,
- công cụ train YOLO,
- và UI inference để test model trực tiếp.

Nếu bạn cần bản ngắn gọn để nộp báo cáo, README này có thể rút thành:
- giới thiệu,
- cấu trúc thư mục,
- cách chạy,
- mô tả UI,
- mô tả output dataset,
- và workflow train/inference.
