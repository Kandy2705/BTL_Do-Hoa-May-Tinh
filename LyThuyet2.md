# LÝ THUYẾT ĐỒ HỌA MÁY TÍNH & TỐI ƯU HÓA AI
## Tài liệu hướng dẫn cho Đồ án BTL

---

## CHƯƠNG 1: HỆ THỐNG ĐỒ HỌA (GRAPHICS SYSTEM)

### 1.1. Đồ họa máy tính là gì?

**Computer Graphics (Đồ họa máy tính)** = Tất cả các khía cạnh tạo ra hình ảnh bằng máy tính, bao gồm:
- **Hardware (Phần cứng)**: Card đồ họa, màn hình
- **Software (Phần mềm)**: OpenGL, Direct3D
- **Ứng dụng**: Maya, Lightwave (cao cấp), OpenGL (cấp thấp)

### 1.2. Các ứng dụng của Đồ họa máy tính

| Ứng dụng | Ví dụ |
|-----------|--------|
| **Display of Information** | Hiển thị dữ liệu khoa học phức tạp, y tế |
| **Computer-Aided Design (CAD)** | Thiết kế kiến trúc, cơ khí, logic số |
| **Simulation and Animation** | Mô phỏng bay, huấn luyện phẫu thuật, VR, game |
| **User Interface** | Giao diện người dùng (icon, menu) |

### 1.3. Tổng quan hệ thống đồ họa

```
Input Devices ──► Processor ──► Frame Buffer ──► Output Devices
(Thiết bị vào)      (Bộ xử lý)     (Bộ đệm khung)    (Thiết bị ra)
```

### 1.4. Frame Buffer (Bộ đệm khung)

**Frame Buffer** = Vùng nhớ lưu các pixel của hình ảnh hiển thị.

| Khái niệm | Giải thích |
|-----------|------------|
| **Pixel** | Điểm ảnh nhỏ nhất, hình ảnh = mảng 2D các pixel |
| **Resolution** | Số pixel trong frame buffer (VD: 1920x1080) |
| **Depth/Precision** | Số bit cho mỗi pixel |

**Độ sâu màu**:
- 1 bit → 2 màu (đen/trắng)
- 8 bits → 256 màu
- 24 bits → True color (hơn 16 triệu màu)

### 1.5. Các loại Buffer trong OpenGL

| Buffer | Chức năng |
|--------|-----------|
| **Color Buffer** | Lưu màu của từng pixel |
| **Depth Buffer (Z-Buffer)** | Lưu khoảng cách từ camera đến vật |
| **Accumulation Buffer** | Buffer độ chính xác cao cho compositing |
| **Stencil Buffer** | Lưu mask cho các hiệu ứng đặc biệt |

---

## CHƯƠNG 2: ĐƯỜNG ỐNG DẪN ĐỒ HỌA (GRAPHICS PIPELINE)

### 2.1. Tổng quan Pipeline

**Graphics Pipeline (Đường ống dẫn đồ họa)** = Quy trình kết xuất đồ họa từ dữ liệu 3D ra màn hình 2D.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OPENGL GRAPHICS PIPELINE                                  │
│                                                                             │
│   Object Space    World Space    Camera Space    Clipping     NDC      Window│
│   (Không gian     (Không gian    (Không gian     Space        Space     Space│
│    đối tượng)     thế giới)      camera)        (Cắt xén)    (Tọa độ   (Màn hình)│
│                                                                       thiết bị)      │
│        │             │              │               │           │          ││        ▼             ▼              ▼               ▼           ▼          ▼│
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌──────────┐ ┌────────┐ ┌──────┐│
│   │  Model   │  │   View   │  │Project- │  │  Clip    │ │  Devide│ │View- ││
│   │  Matrix  │► │  Matrix  │► │  ion    │► │          │►│ by w   │►│ port ││
│   └─────────┘  └─────────┘  └─────────┘  └──────────┘ └────────┘ └──────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2. Chi tiết từng bước trong Pipeline

**Bước 1: Object Space (Không gian đối tượng)**
- Tọa độ cục bộ của vật thể
- Gốc tọa độ đặt tại tâm vật

**Bước 2: World Space (Không gian thế giới)**
- Áp dụng **Model Matrix**: xoay, tịnh tiến, co dãn vật
- Đặt vật vào vị trí thực trong thế giới 3D

**Bước 3: Camera Space (Không gian camera)**
- Áp dụng **View Matrix**: biến đổi để camera ở gốc tọa độ
- Camera mặc định nhìn theo hướng -Z

**Bước 4: Clipping Space (Không gian cắt xén)**
- Áp dụng **Projection Matrix**
- Xác định vật nào nằm trong khung nhìn

**Bước 5: NDC - Normalized Device Coordinates (Tọa độ thiết bị chuẩn)**
- Chia tọa độ cho w (perspective division)
- Tọa độ nằm trong khoảng [-1, 1]

**Bước 6: Window Space (Không gian cửa sổ)**
- Áp dụng **Viewport Transform**
- Chuyển sang tọa độ pixel trên màn hình

### 2.3. Ma trận trong Pipeline

| Ma trận | Chức năng | Công thức |
|---------|-----------|-----------|
| **Model** | Đặt vật vào thế giới | T(x,y,z) × R(θ) × S(sx,sy,sz) |
| **View** | Di chuyển camera | gluLookAt(eye, center, up) |
| **Projection** | Chiếu từ 3D sang 2D | glOrtho() hoặc glFrustum() |

### 2.4. GPU Pipeline (Vertex → Fragment)

```
Vertices ──► Vertex Shader ──► Assembly ──► Rasterization ──► Fragment Shader ──► Output
  (đỉnh)       (xử lý đỉnh)      (ghép hình)   (chia pixels)      (tô màu pixel)
```

**Vertex Shader**: Xử lý từng đỉnh (biến đổi, ánh sáng)
**Fragment Shader**: Tô màu từng pixel (texture, lighting)


---

## CHƯƠNG 3: CÁC PHÉP BIẾN ĐỔI HÌNH HỌC (TRANSFORMATIONS)

### 3.1. Tọa độ Đồng nhất (Homogeneous Coordinates)

**Homogeneous Coordinates** = Biểu diễn điểm 3D bằng vector 4 chiều:

```
Điểm 3D: P(x, y, z)
Điểm 4D: P = [x, y, z, w]

Điểm thực (w ≠ 0):
x' = x/w, y' = y/w, z' = z/w

Vector (w = 0):
v = [vx, vy, vz, 0]
```

**Tại sao cần Homogeneous Coordinates?**
- Cho phép biểu diễn cả điểm lẫn vector
- Tất cả phép biến đổi (dịch, xoay, co dãn) đều có thể thực hiện bằng nhân ma trận 4x4

### 3.2. Các phép biến đổi cơ bản

#### a) Translation (Tịnh tiến)

```
Ma trận T(dx, dy, dz):
┌               ┐
│ 1  0  0  dx  │
│ 0  1  0  dy  │
│ 0  0  1  dz  │
│ 0  0  0  1   │
└───────────────┘

P' = T × P
```

#### b) Scaling (Co dãn)

```
Ma trận S(sx, sy, sz):
┌               ┐
│ sx 0  0  0   │
│ 0  sy 0  0   │
│ 0  0  sz 0   │
│ 0  0  0  1   │
└───────────────┘
```

#### c) Rotation (Xoay) - Quanh trục Z:

```
┌                    ┐
│ cosθ  -sinθ  0  0 │
│ sinθ   cosθ  0  0 │
│  0      0    1  0 │
│  0      0    0  1 │
└────────────────────┘
```

### 3.3. Concatenation (Ghép phép biến đổi)

Nhiều ma trận có thể nhân lại thành 1 ma trận duy nhất:

```
M = T × R × S
P' = M × P
```

**Lưu ý**: Ma trận bên PHẢI được áp dụng TRƯỚC.

---

## CHƯƠNG 4: VIEWING (QUAN SÁT)

### 4.1. Camera trong OpenGL

**Quy ước OpenGL**:
- Camera mặc định đặt tại gốc tọa độ (0, 0, 0)
- Hướng nhìn theo hướng **-Z**
- View volume mặc định: hình lập phương cạnh 2

### 4.2. gluLookAt()

```c
gluLookAt(eye_x, eye_y, eye_z,    // Vị trí camera
           center_x, center_y, center_z,  // Điểm nhìn vào
           up_x, up_y, up_z);      // Vector hướng lên
```

### 4.3. Các loại Projection (Phép chiếu)

| Loại | Mô tả | Ứng dụng |
|------|--------|-----------|
| **Orthographic** | Chiếu song song, không có viễn cận | Bản vẽ kỹ thuật |
| **Perspective** | Chiếu hội tụ, có viễn cận | Chụp ảnh, game |

### 4.4. glOrtho() và glFrustum()

```c
glOrtho(left, right, bottom, top, near, far);     // Chiếu trực giao
glFrustum(left, right, bottom, top, near, far);   // Chiếu phối cảnh
gluPerspective(fovy, aspect, near, far);          // Chiếu phối cảnh (đơn giản)
```

### 4.5. Viewport

```c
glViewport(x, y, width, height);
// Chuyển từ NDC sang tọa độ pixel
```


---

## CHƯƠNG 5: LIGHTING VÀ SHADING (ÁNH SÁNG VÀ TÔ BÓNG)

### 5.1. Tại sao cần Shading?

Nếu chỉ dùng màu đặc (flat color), hình ảnh sẽ:
- Thiếu chiều sâu
- Không thực tế

→ Cần tính toán ánh sáng dựa trên:
- Nguồn sáng
- Vị trí camera
- Hướng bề mặt (normal)
- Thuộc tính vật liệu

### 5.2. Các thành phần ánh sáng

#### a) Ambient Light (Ánh sáng môi trường)

- Ánh sáng tổng quát, không đến từ đâu cụ thể
- Giúp vật không bị tối đen hoàn toàn
- Mô phỏng ánh sáng phản xạ từ các bề mặt khác

```
I_ambient = ka × Ia
```

#### b) Diffuse Light (Ánh sáng khuếch tán)

- Tùy góc nghiêng mà vật sáng/tối khác nhau
- Mặt đón nắng trực tiếp → SÁNG
- Mặt nghiêng away → TỐI

```
I_diffuse = kd × (L · N) × Id
```

**L** = vector từ điểm đến nguồn sáng
**N** = pháp tuyến của bề mặt

#### c) Specular Light (Ánh sáng phản chiếu)

- Tạo "điểm sáng" trên bề mặt bóng (kim loại, nhựa)
- Phụ thuộc vào **Shininess** (độ bóng)

```
I_specular = ks × (R · V)^α × Is
```

### 5.3. Mô hình Phong (Phong Model)

```
I = ka × Ia + kd × (L · N) × Id + ks × (R · V)^α × Is
```

### 5.4. Mô hình Blinn-Phong (Modified Phong)

Sử dụng **Half Vector** H = (L + V) / |L + V| thay vì R:

```
I = ka × Ia + kd × (L · N) × Id + ks × (N · H)^α × Is
```

**Ưu điểm**: Tính toán nhanh hơn (chỉ cần normalize thay vì tính reflection)

### 5.5. Nguồn sáng trong OpenGL

```c
// Định nghĩa nguồn sáng
GLfloat light_pos[] = {1.0, 2.0, 3.0, 1.0};  // w=1: điểm, w=0: song song
glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
glEnable(GL_LIGHTING);
glEnable(GL_LIGHT0);
```

### 5.6. Thuộc tính vật liệu

```c
GLfloat mat_ambient[] = {0.2, 0.2, 0.2, 1.0};
GLfloat mat_diffuse[] = {1.0, 0.8, 0.0, 1.0};
GLfloat mat_specular[] = {1.0, 1.0, 1.0, 1.0};
GLfloat shininess = 100.0;

glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);
glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
glMaterialf(GL_FRONT, GL_SHININESS, shininess);
```

---

## CHƯƠNG 6: RASTERIZATION VÀ HIDDEN-SURFACE REMOVAL

### 6.1. Rasterization (Quét đổi)

**Rasterization** = Quá trình chuyển đỉnh/tam giác thành các pixel trên màn hình.

```
Primitives (đỉnh) → Rasterization → Fragments (điểm ảnh)
```

### 6.2. Clipping (Cắt xén)

**Clipping** = Loại bỏ các phần nằm ngoài view frustum.

**Thuật toán Cohen-Sutherland**:
- Chia không gian thành 9 vùng bằng outcode
- Kiểm tra nhanh để loại bỏ hoặc tính giao điểm

### 6.3. Z-Buffer Algorithm (Thuật toán đệm độ sâu)

**Vấn đề**: Khi 2 vật chồng lên nhau, pixel nào hiển thị?

**Giải pháp**: Lưu khoảng cách z của pixel gần nhất.

```c
// Bật Z-Buffer
glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH);
glEnable(GL_DEPTH_TEST);

// Xóa cả color và depth buffer
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
```

**Thuật toán**:
```
1. Khởi tạo Z-Buffer = INFINITY
2. Với mỗi polygon:
   Với mỗi pixel trong polygon:
     Tính z_pixel
     Nếu z_pixel < Z-Buffer[x,y]:
       Z-Buffer[x,y] = z_pixel
       Color Buffer[x,y] = màu của polygon
```


---

## CHƯƠNG 7: TEXTURE MAPPING (ÁNH XẠ KẾT CẤU)

### 7.1. Texture Mapping là gì?

**Texture Mapping** = "Dán" ảnh 2D lên bề mặt 3D.

```
Ảnh 2D (Texture) ──► Bề mặt 3D (Geometry)
```

**Ứng dụng**:
- Tăng chi tiết bề mặt mà không cần thêm polygon
- Mô phỏng vật liệu (gỗ, đá, vải...)

### 7.2. UV Coordinates (Tọa độ UV)

**UV** = Tọa độ trên texture (U: ngang, V: dọc)
- U, V ∈ [0, 1]

```
Texture (ảnh 2D)
┌──────────────────┐
│ (0,1)      (1,1)│
│    ┌────────┐   │
│    │  Mesh  │   │  UV(0.3, 0.7)
│    └────────┘   │     = điểm nằm 30% ngang, 70% dọc
│ (0,0)      (1,0)│
└──────────────────┘
```

### 7.3. Texture Coordinates trong OpenGL

```c
glBegin(GL_POLYGON);
  glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, 0.0);
  glTexCoord2f(1.0, 0.0); glVertex3f( 1.0, -1.0, 0.0);
  glTexCoord2f(1.0, 1.0); glVertex3f( 1.0,  1.0, 0.0);
  glTexCoord2f(0.0, 1.0); glVertex3f(-1.0,  1.0, 0.0);
glEnd();
```

### 7.4. Thiết lập Texture trong OpenGL

```c
// 1. Tạo texture
glGenTextures(1, &textureID);
glBindTexture(GL_TEXTURE_2D, textureID);

// 2. Thiết lập tham số
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

// 3. Load ảnh
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, 
             GL_UNSIGNED_BYTE, imageData);

// 4. Bật texture
glEnable(GL_TEXTURE_2D);
```

### 7.5. Wrapping và Filtering

**Wrapping** (Khi UV nằm ngoài [0,1]):
- `GL_REPEAT`: Lặp lại
- `GL_CLAMP`: Kẹt ở biên

**Filtering** (Khi texture nhỏ/lớn hơn pixel):
- `GL_NEAREST`: Lấy texel gần nhất (nhanh, pixelated)
- `GL_LINEAR`: Nội suy tuyến tính (mượt)

---

## CHƯƠNG 8: SHADER TRONG OPENGL

### 8.1. Shader là gì?

**Shader** = Chương trình chạy trên GPU, xử lý dữ liệu đỉnh/pixel.

| Loại | Xử lý | Chạy trên |
|------|-------|-----------|
| **Vertex Shader** | Từng đỉnh | GPU |
| **Fragment Shader** | Từng pixel | GPU |

### 8.2. GLSL - OpenGL Shading Language

```glsl
// Vertex Shader
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 MVP;

void main() {
    gl_Position = MVP * vec4(aPos, 1.0);
}

// Fragment Shader
#version 330 core
out vec4 FragColor;

void main() {
    FragColor = vec4(1.0, 0.5, 0.2, 1.0);
}
```

### 8.3. Attribute và Uniform

```glsl
// Attribute: Dữ liệu riêng cho mỗi đỉnh
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;

// Uniform: Dữ liệu chung cho tất cả đỉnh/pixel
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 lightPos;
```


---

## CHƯƠNG 9: TỐI ƯU HÓA AI - STOCHASTIC GRADIENT DESCENT

### 9.1. Bài toán tối ưu hóa

**Mục tiêu**: Tìm tham số θ để **Loss** (hàm mất mát) đạt **giá trị nhỏ nhất**.

```
Loss = f(θ)
Tìm θ* sao cho: θ* = argmin_θ Loss(θ)
```

### 9.2. Gradient Descent (GD)

```python
θ = θ - α × ∇Loss(θ)
# α: learning rate (tốc độ học)
```

**Ưu điểm**: Chính xác
**Nhược điểm**: Chậm với dữ liệu lớn (cần tính gradient trên TOÀN BỘ dataset)

### 9.3. Stochastic Gradient Descent (SGD)

```python
θ = θ - α × ∇Loss(θ; x_i)  # Chỉ dùng 1 mẫu
```

**Ưu điểm**: Nhanh, có thể thoát local minima
**Nhược điểm**: Dao động mạnh

### 9.4. Mini-batch SGD

```python
θ = θ - α × (1/batch_size) × Σ∇Loss(θ; x_i)  # Dùng batch mẫu
```

**Ưu điểm**: Cân bằng giữa tốc độ và ổn định
**Nhược điểm**: Cần chọn batch size phù hợp

### 9.5. Momentum

```python
v = β × v + (1 - β) × ∇Loss(θ)
θ = θ - α × v
```

**Ưu điểm**: Giảm dao động, nhanh hơn, vượt ổ gà nhỏ
**Nhược điểm**: Có thể "quá tốc"

### 9.6. Adam (Adaptive Moment Estimation)

```python
m = β₁ × m + (1 - β₁) × g      # Gradient trung bình
v = β₂ × v + (1 - β₂) × g²    # Bình phương gradient
m̂ = m / (1 - β₁^t)
v̂ = v / (1 - β₂^t)
θ = θ - α × m̂ / (√v̂ + ε)
```

**Ưu điểm**: Tự điều chỉnh learning rate, hoạt động tốt hầu hết bài toán
**Nhược điểm**: Tốn bộ nhớ hơn

### 9.7. So sánh các thuật toán

| Thuật toán | Tốc độ | Ổn định | Bộ nhớ | Khi nào dùng |
|------------|--------|---------|--------|--------------|
| GD | Chậm | Cao | Ít | Dataset nhỏ |
| SGD | Nhanh | Thấp | Ít | Online learning |
| Mini-batch | Nhanh | Trung bình | Trung bình | **Tiêu chuẩn** |
| Momentum | Nhanh | Khá | Ít | Dataset có local minima |
| Adam | Nhanh | Cao | Nhiều | **Mặc định khi không biết dùng gì** |

---

## TỔNG KẾT

### Kiến thức cốt lõi cần nắm:

1. **Pipeline**: Object Space → World Space → Camera Space → Clipping Space → NDC → Window Space
2. **Transformations**: Model Matrix → View Matrix → Projection Matrix
3. **Lighting**: Ambient + Diffuse + Specular (Phong/Blinn-Phong)
4. **Hidden-Surface**: Z-Buffer Algorithm
5. **Texture**: UV Mapping, Wrapping, Filtering
6. **SGD**: GD → SGD → Mini-batch → Momentum → Adam

---

**Tài liệu được biên soạn theo nội dung bài giảng HCMUT - Computer Graphics**
