# LÝ THUYẾT ĐỒ HỌA MÁY TÍNH & TỐI ƯU HÓA AI
## Tài liệu hướng dẫn từ con số 0 cho Đồ án BTL

---

> **Thân gửi các bạn sinh viên thân mến!** 
> 
> Nếu bạn đang đọc tài liệu này với tâm trạng "ngợp" và "chẳng hiểu gì cả" — xin hãy thở một hơi thật sâu. Bạn không hề cô đơn đâu. Đồ họa máy tính và AI nghe có vẻ đáng sợ, nhưng thực ra chúng chỉ là những ý tưởng được đóng gói phức tạp mà thôi. Trong tài liệu này, thầy/cô sẽ cùng bạn "gỡ từng nút thắt", biến những khái niệm trừu tượng thành hình ảnh quen thuộc. Hãy đọc từ từ, đọc đi đọc lại, và nhớ rằng: **ngay cả những chuyên gia cũng từng ở vị trí của bạn hôm nay.**

---

# CHƯƠNG 1: Nền tảng Đồ họa 3D (Xóa mù chữ OpenGL)

## 1.1. OpenGL Pipeline — "Dây chuyền sản xuất" kết xuất đồ họa

### 🎬 Ngữ cảnh: Từ clay (đất sét) đến bức tranh hoàn chỉnh

Hãy tưởng tượng bạn muốn vẽ một bức tranh phong cảnh. Dưới đây là "dây chuyền sản xuất" mà máy tính sử dụng để biến dữ liệu thô (tọa độ các điểm) thành hình ảnh trên màn hình:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        QUY TRÌNH OPENGL PIPELINE                           │
│                                                                             │
│  [Dữ liệu thô]                                                             │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────┐    Đây là nơi bạn (lập trình viên) can thiệp!            │
│  │  Vertex     │◄─── Bạn viết code ở đây (Vertex Shader)                   │
│  │  Shader     │    "Tôi muốn xử lý từng điểm đỉnh trước"                  │
│  └──────┬──────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────┐                                                            │
│  │ Shape       │  Máy tính nối các điểm thành hình                        │
│  │ Assembly    │  "Lấy 3 điểm nối thành tam giác"                          │
│  └──────┬──────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────┐    Đây cũng là nơi bạn can thiệp!                         │
│  │ Rasterization│◄── Fragment Shader (thường gọi là Pixel Shader)          │
│  └──────┬──────┘    "Tôi muốn tô màu từng pixel bên trong tam giác"       │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────┐                                                            │
│  │ Per-Fragment│  Kiểm tra độ sâu, blend màu, xóa pixel bị che khuất      │
│  │ Operations │  "Pixel này có nhìn thấy được không?"                      │
│  └──────┬──────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  [Màn hình hiển thị] 🎉                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 🔍 Vertex Shader (Shader đỉnh) — "Người nắn đất sét"

**Vertex (đỉnh)** = Một điểm trong không gian 3D, có tọa độ (x, y, z). Nó giống như một "điểm đinh" trên bản đồ.

**Shader (kịch bản/chương trình)** = Một chương trình nhỏ chạy trên GPU (card đồ họa), viết bằng ngôn ngữ GLSL. GPU có hàng nghìn lõi nhỏ, nên nó xử lý song song cực nhanh.

**Vertex Shader** là người thợ nắn đất sét đầu tiên. Nhiệm vụ của nó:
- Nhận tọa độ đỉnh (Vertex) từ CPU gửi sang
- **Biến đổi (Transform)**: Xoay, tịnh tiến, phóng to/thu nhỏ đối tượng
- **Chiếu (Project)**: Chuyển từ không gian 3D sang không gian 2D (vì màn hình chỉ có 2D!)
- Gửi kết quả cho bước tiếp theo

**Ví dụ dân dã**: Bạn có một con robot bằng đất sét. Vertex Shader giống như bạn xoay con robot, di chuyển nó, và chuẩn bị đặt nó vào vị trí chụp ảnh.

### 🔍 Fragment Shader (Shader phân mảnh/điểm ảnh) — "Người tô màu"

**Fragment (phân mảnh)** = Tương đương với **Pixel (điểm ảnh)**. Nếu tam giác của bạn có diện tích 100 pixel, sẽ có 100 Fragment được tạo ra.

**Fragment Shader** là người thợ sơn màu. Nhiệm vụ:
- Nhận thông tin về mỗi pixel (vị trí, texture, độ sáng...)
- Tính toán màu cuối cùng cho pixel đó
- "Tô màu" pixel dựa trên ánh sáng, vật liệu, texture...

**Ví dụ dân dã**: Tiếp tục với robot bằng đất sét. Sau khi Vertex Shader đặt robot vào vị trí, Fragment Shader sẽ quyết định: "Điểm này màu xanh, điểm kia màu đỏ, chỗ kia có bóng đổ".

### 💡 Tóm tắt bằng hình ảnh

```
Tam giác 3 đỉnh         Rasterization                Fragment Shader
(3 Vertices)    ───────────────────►   Tam giác được "lấp đầy"   ──────►  Màu sắc cuối cùng
                      ( chia nhỏ            bằng các Pixel           cho từng Pixel
                        thành pixels)          (Fragments)
```

---

## 1.2. Ma trận MVP (Model - View - Projection) — "Bộ ba huyền thoại"

### 🧩 Ma trận (Matrix) là gì?

**Matrix (ma trận)** = Một bảng số được sắp xếp theo hàng và cột. Trong đồ họa 3D, ma trận 4x4 được dùng để biểu diễn các phép biến đổi (xoay, tịnh tiến, co dãn...).

**Tại sao phải dùng ma trận?** Vì bạn có thể "nhân chồng" nhiều phép biến đổi lại với nhau thành một phép tính duy nhất. Thay vì xoay rồi tịnh tiến rồi co dãn (3 phép tính), bạn nhân 3 ma trận lại = 1 phép tính duy nhất. Máy tính thích điều này vì nó **nhanh hơn**.

### 🎯 Ba loại ma trận bạn CẦN BIẾT

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          BA LỚP BIẾN ĐỔI                                    │
│                                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────────┐                              │
│  │ WORLD   │    │ CAMERA  │    │ PROJECTION  │                              │
│  │ SPACE   │───►│ VIEW    │───►│             │                              │
│  │         │    │         │    │             │                              │
│  │ "Vị trí │    │ "Mắt    │    │ "Lens của   │                              │
│  │  thực   │    │  nhìn   │    │  máy ảnh    │                              │
│  │  của    │    │  của    │    │  (wide/     │                              │
│  │  vật    │    │  bạn    │    │  tele)"      │                              │
│  │  trong  │    │  đặt ở  │    │             │                              │
│  │  thế    │    │  đâu?"  │    │             │                              │
│  │  giới   │    │         │    │             │                              │
│  └─────────┘    └─────────┘    └─────────────┘                              │
│       │              │                │                                      │
│       ▼              ▼                ▼                                      │
│  ┌─────────┐    ┌─────────┐    ┌─────────────┐                              │
│  │  MODEL  │    │  VIEW   │    │ PROJECTION  │                              │
│  │ MATRIX  │    │ MATRIX  │    │   MATRIX   │                              │
│  └─────────┘    └─────────┘    └─────────────┘                              │
│                                                                             │
│              MVP_MATRIX = Projection × View × Model                         │
│              (nhân từ PHẢI qua TRÁI)                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1️⃣ Model Matrix (Ma trận mô hình) — "Vị trí & tư thế của vật thể"

**Chức năng**: Biến đổi tọa độ từ **Local Space (không gian địa phương)** sang **World Space (không gian thế giới)**.

**Local Space** = Tọa độ của vật khi bạn "ngồi trên vật đó" (gốc tọa độ đặt ở tâm vật).

**World Space** = Tọa độ thực trong thế giới 3D (giống tọa độ trên bản đồ GPS).

**Ví dụ dân dã**: Bạn có một con thỏ bông. Khi bạn cầm nó lên và đặt xuống bàn (ở vị trí x=2, y=0, z=3), Model Matrix giống như ghi chú: "Con thỏ bông này đang ở vị trí này trên bàn, và nó đang quay mặt về hướng nam".

Model Matrix bao gồm:
- **Translation (Tịnh tiến)**: Di chuyển vật từ gốc tọa độ đến vị trí thực
- **Rotation (Xoay)**: Xoay vật theo các trục X, Y, Z
- **Scale (Co dãn)**: Phóng to/thu nhỏ vật

### 2️⃣ View Matrix (Ma trận view/camera) — "Vị trí và hướng nhìn của camera"

**Chức năng**: Biến đổi tọa độ từ **World Space** sang **View Space (không gian camera)**.

**View Space** = Không gian tính từ vị trí camera. Camera đặt ở gốc tọa độ (0,0,0) và nhìn về phía -Z.

**Ví dụ dân dã**: Giống như khi bạn chụp ảnh. Dù thế giới có vạn vật, camera chỉ "thấy" những gì nằm trong khung hình và từ góc nhìn của nó. View Matrix giống như việc bạn:
1. Đứng vào vị trí chụp (đặt camera ở đâu?)
2. Hướng mặt về phía nào (camera nhìn đâu?)

### 3️⃣ Projection Matrix (Ma trận chiếu) — "Lens của camera"

**Chức năng**: Biến đổi từ **View Space** sang **Clip Space**, thực hiện phép chiếu từ 3D sang 2D.

Có 2 loại projection phổ biến:

| Loại | Tên gọi | Ví dụ thực tế |
|------|---------|---------------|
| **Orthographic** (Phép chiếu trực giao) | Nhìn song song, không có viễn cận | Bản vẽ kỹ thuật, CAD |
| **Perspective** (Phép chiếu phối cảnh) | Có viễn cận, đường ray hội tụ | Chụp ảnh thực tế, mắt người |

**Projection Matrix** giống như **lens** của máy ảnh:
- Lens góc rộng (wide-angle): Nhìn nhiều thứ, nhưng vật ở xa trông nhỏ hơn nhiều
- Lens tele: Nhìn ít thứ, nhưng vật ở xa vẫn to

### 🔗 Tại sao phải NHÂN tất cả lại với nhau?

```
position_clipspace = Projection × View × Model × position_local
```

**Lý do thực tế**:

1. **Hiệu năng**: Thay vì CPU phải nhân 3 lần với 3 ma trận riêng lẻ cho hàng triệu đỉnh, ta nhân 3 ma trận với nhau TRƯỚC thành 1 ma trận MVP, rồi CPU chỉ nhân **1 lần** với mỗi đỉnh.

2. **Thứ tự QUAN TRỌNG**: 
   - Đầu tiên: Model (đặt vật vào thế giới)
   - Tiếp theo: View (đặt camera)
   - Cuối cùng: Projection (chiếu lên ảnh 2D)
   
   **THỨ Tự SAI = KẾT QUẢ SAI!**

**Ví dụ dân dã**: Hãy tưởng tượng bạn muốn chụp ảnh một người đang nhảy:
1. Đầu tiên, người đó phải ở đúng vị trí trong phòng (Model)
2. Sau đó bạn đứng vào vị trí chụp và hướng máy ảnh về phía người đó (View)
3. Cuối cùng bạn zoom lens để vừa khung hình (Projection)

Nếu bạn làm ngược (zoom trước khi biết người đứng ở đâu), kết quả sẽ thảm họa!

---

# CHƯƠNG 2: Bóc tách Phần 1 (Thực hành 3D Engine)

## 2.1. File 3D (.obj, .ply) — "Bản đồ xây dựng" cho máy tính

### 📦 File .obj và .ply chứa gì?

Khi bạn tải một mô hình 3D từ internet (ví dụ: con mèo.stl), bên trong file đó chỉ là **TEXT thuần túy** chứa các con số. Không có gì "ma thuật" cả!

### 🧱 Các thành phần cơ bản

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CẤU TRÚC FILE .OBJ                                │
│                                                                             │
│  # Comment (ghi chú, máy tính bỏ qua)                                      │
│  mtllib materials.mtl    ◄─── Link đến file màu (material)                 │
│                                                                             │
│  o Cube                  ◄─── Tên đối tượng (object)                        │
│                                                                             │
│  # Vertices (đỉnh) - Tọa độ các điểm trong không gian 3D                   │
│  v 1.0 2.0 3.0           ◄─── Đỉnh 1: x=1, y=2, z=3                        │
│  v 4.0 5.0 6.0           ◄─── Đỉnh 2                                       │
│  v 7.0 8.0 9.0           ◄─── Đỉnh 3                                       │
│  ...                                                                     │
│                                                                             │
│  # Normals (pháp tuyến) - Hướng "mặt" của bề mặt                          │
│  vn 0.0 1.0 0.0          ◄─── Hướng lên trên (+Y)                          │
│  vn 0.0 0.0 1.0          ◄─── Hướng ra ngoài (+Z)                          │
│                                                                             │
│  # Texture Coordinates (UV) - Tọa độ texture 2D                            │
│  vt 0.5 0.5              ◄─── Điểm giữa của ảnh texture                    │
│  vt 1.0 0.0              ◄─── Góc phải dưới                               │
│                                                                             │
│  # Faces (mặt) - Cách nối các đỉnh thành tam giác                          │
│  f 1 2 3                 ◄─── Mặt 1: nối đỉnh 1-2-3 thành tam giác         │
│  f 1 3 4                 ◄─── Mặt 2: nối đỉnh 1-3-4                       │
│  f 1//1 2//1 3//1       ◄─── Face có chỉ định normal                      │
│  f 1/1/1 2/2/1 3/3/1     ◄─── Face có đủ: vertex/uv/normal                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 🔤 Từ vựng quan trọng:

| Thuật ngữ | Ý nghĩa | Ví dụ dân dã |
|-----------|---------|--------------|
| **Vertex (đỉnh)** | Một điểm trong không gian 3D | Điểm góc của một khối Rubik |
| **Face (mặt)** | Một tam giác tạo từ 3 đỉnh | Mỗi mặt của khối Rubik được chia thành tam giác |
| **Normal (pháp tuyến)** | Vector vuông góc với mặt, chỉ hướng "mặt" đang nhìn đâu | Đầu kim xuyên qua mặt bàn, chỉ thẳng lên trần nhà |
| **UV Coordinates (tọa độ UV)** | Vị trí trên texture 2D (U=ngang, V=dọc) | Tọa độ trên tấm ảnh (U=trái-phải, V=trên-dưới) |

### 🧩 Máy tính đọc và nối như thế nào?

```
Bước 1: Đọc danh sách VERTICES (các điểm A, B, C, D, E...)
        A(0,0,0) ─── B(1,0,0) ─── C(1,1,0)
        │                      │
        │                      │
        D(0,1,0) ─── E(1,1,0) ─── F(2,1,0)

Bước 2: Đọc danh sách FACES (cách nối)
        Face 1: 1 2 3  (nối đỉnh A-B-C)
        Face 2: 1 4 5  (nối đỉnh A-D-E)
        ...

Bước 3: Vẽ từng tam giác lên màn hình
        ┌─────┐
        │  /\│   ← Mỗi mặt là một tam giác
        │ /  │     (tam giác không có "mặt trước/sau", 
        │/   │      bạn nhìn ở đâu là mặt đó)
        └─────┘
```

**Điều quan trọng**: Máy tính **KHÔNG** hiểu khái niệm "hình vuông", "hình hộp". Nó chỉ hiểu **tam giác** (Triangle). Muốn vẽ hình vuông? Hãy chia thành 2 tam giác!

---

## 2.2. Texture (Kết cấu/Ảnh dán) — "Giấy gói quà cho khối 3D"

### 🎁 Texture là gì?

**Texture (kết cấu)** = Một bức ảnh 2D (PNG, JPG...) được "dán" lên bề mặt 3D. Nó giống như:

> Bạn có một khối Rubik trắng xóa. Dán giấy màu lên từng mặt → Khối Rubik hoàn chỉnh!
> 
> Bạn có một chiếc hộp giấy. Bọc giấy gói có họa tiết bên ngoài → Chiếc hộp quà đẹp!

### 🗺️ UV Mapping — "Cách gấp giấy"

**UV Mapping (ánh xạ UV)** = Quá trình gán mỗi điểm trên bề mặt 3D đến một điểm trên ảnh 2D.

```
Không gian 3D (Model)                    Không gian 2D (Texture)
                                           ┌──────────────────┐
    ┌───────────────┐                      │  U →              │
    │               │                      │  ┌────────────┐   │
    │    A(0,0,0)    │◄──────┐              │  │            │   │
    │   /│\         │       │  UV:(0.2,0.8) │  │    🐱      │   │
    │  / │ \        │       └──────────────┼─►│            │   │
    │ /  │  \       │                      │  │            │   │
    │/   │   \      │                       │  └────────────┘   │
    └────┴────┘     │                       │        ▲           │
                    │                       │        │            │
                    │                       │        V            │
                    │                       │   V ↓               │
                    │                       └─────────────────────┘
```

**UV không phải XYZ**! UV dùng để đo vị trí trên ẢNH TEXTURE:
- **U**: Tọa độ ngang (0 = trái, 1 = phải)
- **V**: Tọa độ dọc (0 = dưới, 1 = trên)

**Ví dụ thực tế**:
- UV (0, 0) = Điểm dưới-trái của ảnh texture
- UV (1, 1) = Điểm trên-phải của ảnh texture  
- UV (0.5, 0.5) = Chính giữa ảnh

### 🎨 Texture Wrapping — "Lặp lại hay kẹt biên?"

Khi texture được ánh xạ ra ngoài phạm vi 0-1:

| Chế độ | Ý nghĩa | Ví dụ |
|--------|---------|-------|
| **REPEAT** | Lặp lại tile | Gạch hoa, vải nỉ |
| **CLAMP_TO_EDGE** | Kẹt ở biên | Khung tranh |
| **MIRRORED_REPEAT** | Lặp có đảo ngược | Gạch hoa đối xứng |

### 📊 UV và Texture Coordinates trong code

```python
# Đỉnh với UV coordinate
vertices = [
    # position      # UV
    0.5,  0.5, 0.0,  1.0, 1.0,  # đỉnh trên-phải → UV(1,1) = trên-phải texture
    0.5, -0.5, 0.0,  1.0, 0.0,  # đỉnh dưới-phải → UV(1,0) = dưới-phải texture
   -0.5, -0.5, 0.0,  0.0, 0.0,  # đỉnh dưới-trái → UV(0,0) = dưới-trái texture
   -0.5,  0.5, 0.0,  0.0, 1.0,  # đỉnh trên-trái → UV(0,1) = trên-trái texture
]
```

---

## 2.3. Ánh sáng Blinn-Phong — "Làm cho thế giới 3D có chiều sâu"

### 💡 Tại sao cần mô hình chiếu sáng?

Trong thực tế, bạn NHÌN THẤY vật thể vì:
1. Vật thể PHẢN XẠ ánh sáng từ đèn/mặt trời vào MẮT bạn
2. Nếu không có ánh sáng → Bạn nhìn thấy gì? **MÀU ĐEN** (trừ khi vật tự phát sáng)

Trong đồ họa 3D, chúng ta CẦN MÔ PHỎNG quá trình này bằng toán học.

### 🌟 Ba thành phần của Blinn-Phong

Blinn-Phong = **Phong shading** cải tiến với tính toán Specular nhanh hơn.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  BA THÀNH PHẦN ÁNH SÁNG BLINN-PHONG                         │
│                                                                             │
│    ┌─────────────────────────────────────────────────────────────────┐     │
│    │  1. AMBIENT (Ánh sáng môi trường)                              │     │
│    │                                                                 │     │
│    │  🌙 "Ánh trăng rằm"                                            │     │
│    │  • Ánh sáng tổng quát, không đến từ đâu cụ thể                 │     │
│    │  • Giúp vật không bị "tối đen" hoàn toàn ở chỗ khuất ánh sáng │     │
│    │  • Cường độ yếu, màu nhạt (thường là xám nhạt hoặc trắng)     │     │
│    └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│    ┌─────────────────────────────────────────────────────────────────┐     │
│    │  2. DIFFUSE (Ánh sáng khuếch tán)                              │     │
│    │                                                                 │     │
│    │  ☀️ "Nắng chiều mai"                                           │     │
│    │  • Ánh sáng TỚI từ nguồn sáng (đèn, mặt trời)                 │     │
│    │  • Tùy góc nghiêng mà vật sáng TỐI khác nhau                  │     │
│    │  • Mặt đón nắng trực tiếp → SÁNG                              │     │
│    │  • Mặt nghiêng away → TỐI hơn                                 │     │
│    └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│    ┌─────────────────────────────────────────────────────────────────┐     │
│    │  3. SPECULAR (Ánh sáng phản chiếu/gương)                      │     │
│    │                                                                 │     │
│    │  ✨ "Tia sáng lấp lánh trên mặt nước"                         │     │
│    │  • Vật liệu BÓNG (nhựa, kim loại) mới có specular mạnh        │     │
│    │  • Vật liệu NHAM/MỜ (gỗ, giấy) → specular yếu                │     │
│    │  • Tạo "điểm nổi" sáng trên bề mặt                            │     │
│    │  • Phụ thuộc SHININESS (độ bóng):                             │     │
│    │    - Shininess cao → Vùng sáng nhỏ, CHÓI LỌI                  │     │
│    │    - Shininess thấp → Vùng sáng lớn, MỜ                       │     │
│    └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 🔬 Công thức (nhưng dễ hiểu!)

```
Màu cuối = Ambient + Diffuse + Specular
```

**Chi tiết từng phần**:

| Thành phần | Công thức đơn giản | Ý nghĩa |
|------------|-------------------|---------|
| **Ambient** | `k_a × I_a` | `k_a` = hệ số ambient (0-1), `I_a` = cường độ ánh sáng môi trường |
| **Diffuse** | `k_d × max(N·L, 0) × I_d` | `N` = pháp tuyến, `L` = hướng tới nguồn sáng, `I_d` = cường độ diffuse |
| **Specular** | `k_s × max(H·N, 0)^shininess × I_s` | `H` = vector nửa góc, `shininess` = độ bóng, `I_s` = cường độ specular |

### 🎯 Giải thích bằng hình ảnh

```
                    LIGHT (Nguồn sáng ☀️)
                          │
                          │ L (hướng tới ánh sáng)
                          ▼
                          │
                    ┌──────┴──────┐
                    │             │
                    │   OBJECT    │
                    │   (Vật thể) │
                    │             │
                    └──────┬──────┘
                           │
              N (Pháp tuyến - hướng mặt) ──► (vuông góc với bề mặt)
                           │
                           │ H (Half-vector - nửa góc giữa L và V)
                           ▼
                    ┌──────────────┐
                    │    CAMERA    │
                    │    (Mắt) V   │
                    └──────────────┘
```

### 🎭 Diffuse: N vs L quyết định độ sáng

```
Trường hợp 1: N thẳng hàng với L (mặt đón nắng trực tiếp)
N = (0, 1, 0)    L = (0, 1, 0)
N·L = 1.0 → MÀU SÁNG TỐI ĐA

Trường hợp 2: N vuông góc với L (mặt nghiêng 90°)
N = (0, 1, 0)    L = (1, 0, 0)  
N·L = 0.0 → MÀU TỐI (không có ánh sáng diffuse)

Trường hợp 3: N ngược hướng L (mặt quay lưng với nguồn sáng)
N = (0, 1, 0)    L = (0, -1, 0)
N·L = -1.0 → clamp(0) = MÀU TỐI (vì bị che khuất hoàn toàn)
```

### 🌟 Blinn vs Phong: Khác nhau ở đâu?

| Tiêu chí | Phong (gốc) | Blinn (cải tiến) |
|----------|-------------|------------------|
| **Specular tính bằng** | Reflection vector R | Half-vector H |
| **Tốc độ** | Chậm hơn (cần tính arccos) | Nhanh hơn (chỉ cần normalize) |
| **Chất lượng** | Đẹp với góc rộng | Đẹp với góc hẹp, không bị "đứt gãy" |
| **Được dùng trong project** | ✅ | ✅ |

**Half-vector H** = Trung bình cộng của L (hướng tới ánh sáng) và V (hướng tới camera), rồi normalize:
```
H = normalize(L + V)
```

### 🌟 Gouraud Shading vs Phong Shading

**Gouraud Shading** = Shading nội suy màu tại các đỉnh, rồi nội suy sang pixel.

**Phong Shading** = Nội suy pháp tuyến (Normal) tại các đỉnh, rồi tính màu tại pixel.

| Tiêu chí | **Gouraud Shading** | **Phong Shading** |
|-----------|---------------------|-------------------|
| **Cách hoạt động** | Tính màu tại **đỉnh**, nội suy ra pixel | Nội suy **pháp tuyến** ra pixel, rồi tính màu |
| **Chất lượng** | Có thể thấy "đường" giữa các đỉnh | Mượt hơn, đẹp hơn |
| **Tốc độ** | **Nhanh** (tính lighting 1 lần/đỉnh) | **Chậm** (tính lighting mỗi pixel) |
| **Ứng dụng trong project** | ✅ Vertex Shader | ✅ Fragment Shader |

---

### 🌟 Hướng dẫn chi tiết: Gouraud Shading (Tính tại Đỉnh)

**Bước 1: Tính pháp tuyến tại mỗi đỉnh**
```
Normal tại đỉnh = trung bình cộng các Normal của các mặt kề
```

**Bước 2: Tính Phong lighting tại mỗi đỉnh**
```
Màu_tại_đỉnh = Ambient + Diffuse + Specular
```

**Bước 3: Nội suy màu ra pixel bên trong tam giác**
```
Pixel = interpolate(Màu_đỉnh_1, Màu_đỉnh_2, Màu_đỉnh_3)
```

**Minh họa:**

```
GOURAUD SHADING:

Bước 1: Tính Normal tại đỉnh
    Normal_1 = (N_mặt_1 + N_mặt_2 + N_mặt_3) / 3
    
Bước 2: Tính màu tại đỉnh (Phong lighting)
    ●─────────●
    │         │
    │  ▲     │     Màu_1 = Ambient + Diffuse(N1) + Specular(N1)
    │ ╱│╲    │     Màu_2 = Ambient + Diffuse(N2) + Specular(N2)
    │╱ │ ╲   │     Màu_3 = Ambient + Diffuse(N3) + Specular(N3)
    ●─────────●
    
Bước 3: Nội suy màu (Gouraud)
    Màu_pixel = Màu_1 * w1 + Màu_2 * w2 + Màu_3 * w3
    (w1, w2, w3 = trọng số barycentric)
```

---

### 🌟 Hướng dẫn chi tiết: Phong Shading (Tính tại Pixel)

**Bước 1: Tính pháp tuyến tại mỗi đỉnh**

**Bước 2: Nội suy pháp tuyến ra pixel**
```
N_pixel = normalize(N_đỉnh_1 * w1 + N_đỉnh_2 * w2 + N_đỉnh_3 * w3)
```

**Bước 3: Tính Phong lighting tại pixel**
```
Màu_pixel = Ambient + Diffuse(N_pixel) + Specular(N_pixel)
```

**Minh họa:**

```
PHONG SHADING:

Bước 1: Tính Normal tại đỉnh
    ●─────────●
    │         │
    │  ▲     │     N_1, N_2, N_3 (pháp tuyến tại mỗi đỉnh)
    │ ╱│╲    │
    │╱ │ ╲   │
    ●─────────●
    
Bước 2: Nội suy Normal (tại mỗi pixel)
    N_pixel = w1*N_1 + w2*N_2 + w3*N_3
    Sau đó: N_pixel = normalize(N_pixel)
    
Bước 3: Tính màu tại pixel (Phong lighting)
    Màu_pixel = Ambient + Diffuse(N_pixel) + Specular(N_pixel)
```

---

### 🌟 Kết hợp: Màu + Texture + Shading

**Câu hỏi**: Làm sao để vừa dùng **màu chọn**, vừa dùng **texture**, vừa có **shading**?

**Trả lời**: Kết hợp theo công thức:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CÔNG THỨC KẾT HỢP: MÀU + TEXTURE + SHADING          │
│                                                                             │
│  Màu cuối = (Màu_nền × Texture) × Shading                                │
│                                                                             │
│  Chi tiết:                                                                 │
│  ─────────────────────────────────────────────────────────────────────────│
│  1. Màu nền (Base Color): màu do người dùng chọn hoặc màu mặc định      │
│  2. Texture: ảnh được "dán" lên vật (UV mapping)                        │
│  3. Shading: ánh sáng (Ambient + Diffuse + Specular)                      │
│                                                                             │
│  Trong code Fragment Shader:                                               │
│  ─────────────────────────────────────────────────────────────────────────│
│                                                                             │
│  vec3 baseColor = u_color;           // Màu chọn                          │
│  vec3 texColor = texture(u_texture, texcoord).rgb;  // Màu từ texture   │
│                                                                             │
│  // Kết hợp màu nền và texture                                            │
│  vec3 materialColor = mix(baseColor, texColor, u_use_texture);           │
│                                                                             │
│  // Áp dụng shading lên màu đã kết hợp                                    │
│  vec3 finalColor = materialColor * lightingResult;                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Luồng xử lý trong Fragment Shader:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LUỒNG XỬ LÝ TRONG FRAGMENT SHADER                       │
│                                                                             │
│  Input:                                                                    │
│  ├── vertex_color (màu từ đỉnh)                                         │
│  ├── texcoord (tọa độ UV)                                                │
│  ├── normal (pháp tuyến đã nội suy)                                       │
│  └── lights (thông tin ánh sáng)                                          │
│                                                                             │
│  Bước 1: Lấy màu từ Texture (nếu có)                                      │
│  ├── texColor = texture(u_texture, texcoord)                              │
│  └── Nếu không có texture: texColor = (1, 1, 1)                          │
│                                                                             │
│  Bước 2: Kết hợp màu vertex và texture                                     │
│  ├── color = vertex_color × texColor                                       │
│  └── Hoặc: color = mix(vertex_color, texColor, use_texture)                │
│                                                                             │
│  Bước 3: Tính Shading (Phong lighting)                                   │
│  ├── Ambient = ka × Ia                                                    │
│  ├── Diffuse = kd × max(N·L, 0) × Id                                     │
│  └── Specular = ks × (N·H)^shininess × Is                               │
│                                                                             │
│  Bước 4: Áp dụng Shading lên màu                                         │
│  └── finalColor = color × (Ambient + Diffuse + Specular)                  │
│                                                                             │
│  Output: finalColor (màu cuối của pixel)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Trong project của bạn:**
- **Mode A, B, C, D** = Phong Shading (Fragment Shader tính lighting)
- **Mode E, F** = Gouraud Shading (Vertex Shader tính lighting, Fragment nội suy màu)

---

## 2.4. Depth Map (Bản đồ độ sâu) — "Ký ức của camera về khoảng cách"

### 🎯 Z-Buffer: Ai ở TRƯỚC ai?

**Depth Buffer (bộ đệm độ sâu)** = Một ma trận cùng kích thước màn hình, lưu khoảng cách từ camera đến pixel gần nhất.

**Vấn đề**: Khi 2 vật chồng lên nhau trên màn hình, máy tính cần quyết định: **Pixel nào HIỂN THỊ?**

```
    Camera đang nhìn thẳng về phía trước...
    
         Camera 👁️
            │
            │    khoảng cách = 5
            ▼
       ┌─────────┐
       │  Object │    ← A (gần hơn, khoảng cách 5)
       │    A    │    
       └─────────┘
            │
            │    khoảng cách = 10
            ▼
       ┌─────────┐
       │  Object │    ← B (xa hơn, khoảng cách 10)
       │    B    │    → Bị CHE KHUẤT bởi A!
       └─────────┘
```

### 📊 Thuật toán Z-Buffer

```python
# Mỗi pixel trên màn hình có một "ký ức" về khoảng cách
for each pixel (x, y):
    depth_buffer[x, y] = INFINITY  # Ban đầu: vô cùng xa
    color_buffer[x, y] = BLACK

for each triangle:
    for each pixel inside triangle:
        z = calculate_depth(pixel)  # Tính khoảng cách z của pixel này
        
        if z < depth_buffer[x, y]:  # Nếu pixel này GẦN hơn
            depth_buffer[x, y] = z  # Cập nhật ký ức
            color_buffer[x, y] = triangle_color  # Vẽ pixel này
        else:
            # Bỏ qua! Pixel này bị che khuất bởi thứ gì đó gần hơn
            pass
```

### 🎨 Từ Depth Buffer đến Grayscale (độ sâu → màu xám)

**Depth Map** = Hiển thị depth buffer dưới dạng ảnh grayscale.

```
Khoảng cách thực tế (Z)    │    Giá trị màu xám
────────────────────────────┼─────────────────────
0 (rất gần camera)         │    255 (TRẮNG)
1                         │    200
2                         │    150
3                         │    100
4                         │    50
5 (rất xa camera)         │    0 (ĐEN)
```

**Công thức chuyển đổi**:
```
gray = 255 × (1 - normalized_z)
```
hoặc
```
gray = 255 × (far - z) / (far - near)
```

### 📷 Depth Map trong Shadow Mapping

**Shadow Mapping** = Kỹ thuật đổ bóng bằng cách so sánh độ sâu:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     QUY TRÌNH SHADOW MAPPING                                │
│                                                                             │
│  BƯỚC 1: RENDER TỪ GÓC NHÌN CỦA ÁNH SÁNG (Shadow Camera)                   │
│  ──────────────────────────────────────────────────────────────────────     │
│                    ☀️ Light                                                 │
│                     │                                                       │
│                     │ Chiếu ra màn hình                                    │
│                     ▼                                                       │
│              ┌─────────────┐                                               │
│              │ SHADOW MAP  │ ← Lưu khoảng cáách từ ánh sáng                │
│              │  (Depth)    │   đến vật gần nhất                            │
│              └─────────────┘                                               │
│                                                                             │
│  BƯỚC 2: RENDER TỪ GÓC NHÌN CỦA CAMERA CHÍNH                              │
│  ──────────────────────────────────────────────────────────────────────     │
│                    👁️ Camera                                                │
│                     │                                                       │
│                     ▼                                                       │
│         ┌───────────────────────┐                                           │
│         │  Vật thể + Bóng đổ   │                                           │
│         │                       │                                           │
│         │   ? → Có bị che khuất │                                           │
│         │     bởi vật khác      │                                           │
│         │     không?            │                                           │
│         └───────────────────────┘                                           │
│                                                                             │
│  BƯỚC 3: SO SÁNH                                                           │
│  ──────────────────────────────────────────────────────────────────────     │
│  • Tính vị trí pixel trong không gian ánh sáng                             │
│  • Đọc giá trị z từ Shadow Map                                             │
│  • Nếu z_from_camera > z_from_shadow_map → Pixel trong BÓNG!               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# CHƯƠNG 3: Bóc tách Phần 2 (Tối ưu hóa AI - SGD)

## 3.1. Đặt vấn đề — "Viên bi lăn trong sương mù"

### 🎬 Ngữ cảnh: Hành trình tìm đáy thung lũng

Hãy tưởng tượng:

> 🌫️ Bạn đang trong một **thung lũng núi** bao phủ bởi **sương mù dày đặc**. 
> 
> 👀 Bạn **không nhìn thấy gì** ngoài một vùng nhỏ xung quanh mình.
> 
> 🎯 Mục tiêu: Tìm được **đáy thung lũng** (nơi thấp nhất).
> 
> ❓ Câu hỏi: Bạn sẽ đi theo hướng nào?

Câu trả lời: **Bạn sẽ nhìn xuống chân mình** — hướng dốc ngược lại! Và cứ thế bước đi, mỗi bước đi xuống chỗ thấp hơn, cho đến khi... bạn không thể đi xuống nữa. Đó là đáy!

### 🏔️ Bề mặt hàm mất mát (Loss Surface) — "Bản đồ địa hình"

**Loss Surface (bề mặt hàm mất mát)** = Một bề mặt 3D trong không gian, nơi:
- **Trục X, Y**: Tham số của mô hình AI (ví dụ: weights - trọng số)
- **Trục Z (Height/Độ cao)**: Giá trị Loss (hàm mất mát) - **CÀNG THẤP CÀNG TỐT**

```
                Loss (Z) - Thấp nhất ở đáy
                     ▲
                    /│\         /\          (Các điểm cực tiểu - local minima)
                   / │ \       /  \         
                  /  │  \     /    \        
    -1.0         /   │   \   /      \       1.0
  ───────►      /    │    \ /        \      ◄───── Parameters (X, Y)
  Parameter1   /     │     X          \     
              ▼      │    /│\         \    
                      ▼   / │ \         ▼   
                     Đáy  /  │  \           
                    (Global   │   \         
                     Min)     │    \         
                             /\    \        
                            /  \    \       
                           /    \    \      
                          ▼     ▼    ▼      
                      Local   Local  Local   
                      Min     Min    Min    
```

### 🎯 Loss Function (Hàm mất mát) — "Thước đo sai lầm"

**Loss Function (hàm mất mát)** = Một hàm toán học đo lường **"Độ sai lệch"** giữa:
- Dự đoán của AI
- Kết quả thực sự

**Ví dụ dân dã**:
- Bạn đoán điểm thi của bạn là **8.0**
- Điểm thực tế là **7.0**
- Sai lệch = |8.0 - 7.0| = **1.0 điểm**
- → Loss = 1.0 (Càng lớn = Càng sai!)

**Trong AI**: Mục tiêu huấn luyện = **Tìm tham số** sao cho Loss → **NHỎ NHẤT CÓ THỂ**

---

## 3.2. Gradient (Đạo hàm) — "La bàn chỉ đường"

### 🧭 Gradient là gì?

**Gradient (đạo hàm/vec-tơ gradient)** = Vector chỉ hướng **tăng dốc NHANH NHẤT** tại một điểm.

- Gradient **chỉ lên** = Hướng **đi lên** nhanh nhất
- **-Gradient** = Hướng **đi xuống** nhanh nhất (ta cần hướng này!)

### 📐 Giải thích bằng hình ảnh 1D (1 tham số)

```
    Loss
      ▲
      │         slope = dy/dx = 2 (dương → đang đi lên)
      │        /
      │       /  ← Điểm hiện tại
      │      /
      │     / slope = -1 (âm → đang đi xuống)
      │    /
      │   /
      │  / slope = 0 (bằng phẳng → đã đến đáy!)
      │ /
      └────────────────────────────────► Parameter
```

### 📐 Trong không gian 2D (2 tham số)

```
                    Gradient
                       ▲
                       │∇L = (∂L/∂x, ∂L/∂y)
                       │
                       │
                       │
    -1.0 ◄─────────────┼─────────────► 1.0
        Parameter X    │
                       │
                       │
                       │  ∂L/∂x = 2 (tăng theo X)
                       │  ∂L/∂y = 1 (tăng theo Y)
                       │
                       ▼
                       │
                   Parameter Y
```

### 💡 Gradient trong SGD — "La bàn cho viên bi"

```
Gradient Descent = "Đi ngược gradient" = "Đi xuống dốc"

new_position = current_position - learning_rate × gradient

Giải thích:
- Bạn đang ở vị trí A
- La bàn (gradient) chỉ hướng LÊN dốc
- Bạn đi NGƯỢC lại → Đi XUỐNG dốc
- "Learning Rate" = Độ dài mỗi bước đi
```

---

## 3.3. Các thuật toán tối ưu hóa — "So sánh phong cách đi"

### 📊 Tổng quan so sánh

| Thuật toán | Tốc độ học | Bộ nhớ | Ổn định | Ưu điểm nổi bật |
|------------|------------|--------|---------|------------------|
| **GD** | Chậm | Ít | Cao | Đơn giản nhất |
| **SGD** | Trung bình | Ít | Thấp | Nhanh cho dữ liệu lớn |
| **Mini-batch SGD** | Nhanh | Trung bình | Trung bình | Cân bằng tốt nhất |
| **Momentum** | Nhanh | Ít | Khá | Vượt "ổ gà" |
| **Nesterov** | Nhanh | Ít | Tốt | Momentum thông minh hơn |
| **Adam** | Nhanh | Nhiều | Rất cao | "Vạn năng", hoạt động tốt hầu hết mọi lúc |

---

## 3.4. Chi tiết từng thuật toán

### 📗 1. Gradient Descent (GD) — "Người đi bộ cẩn thận"

**Định nghĩa**: Cập nhật tham số sử dụng gradient tính trên **TOÀN BỘ dataset** (tất cả dữ liệu huấn luyện).

**Công thức**:
```
θ = θ - α × ∇L(θ)

Trong đó:
- θ = tham số cần tối ưu
- α (alpha) = learning rate (tốc độ học)
- ∇L(θ) = gradient của hàm mất mát
```

**Nguyên lý hoạt động**:
```
┌─────────────────────────────────────────────────────────────────┐
│  GRADIENT DESCENT - Mỗi bước cần duyệt TOÀN BỘ dữ liệu        │
│                                                                 │
│  Dataset: 1,000,000 ảnh mèo 🐱🐱🐱🐱🐱...🐱                     │
│                                                                 │
│  Step 1: ████████████████ (duyệt hết 1 triệu ảnh)              │
│          ↓                                                       │
│  Step 2: ████████████████ (duyệt hết 1 triệu ảnh)              │
│          ↓                                                       │
│  Step 3: ████████████████ (duyệt hết 1 triệu ảnh)              │
│          ↓                                                       │
│          ...                                                     │
│                                                                 │
│  ⚠️ MỖI BƯỚC = 1 TRIỆU PHÉP TÍNH!                              │
│  ⏰ Rất CHẬM nhưng CHÍNH XÁC                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Ví dụ dân dã**: 
> Bạn muốn tìm điểm thấp nhất trên một ngọn núi. GD giống như bạn đi bộ từng bước NHỎ, mỗi bước đều nhìn toàn cảnh núi trước khi quyết định đi đâu. Cẩn thận, chính xác, nhưng CỰC KỲ CHẬM.

**Ưu điểm**:
- ✅ Đảm bảo hội tụ đến điểm tối ưu toàn cục (với hàm lồi)
- ✅ Đường đi mượt, ít dao động

**Nhược điểm**:
- ❌ **Rất chậm** với dataset lớn
- ❌ Cần load TOÀN BỘ dữ liệu vào RAM (không khả thi với big data)

**Khi nào dùng**:
- Dataset nhỏ (< 10,000 mẫu)
- Khi cần độ chính xác cao nhất

---

### 📙 2. Stochastic Gradient Descent (SGD) — "Người nhảy bật"

**Định nghĩa**: Cập nhật tham số sử dụng gradient tính trên **MỘT mẫu** ngẫu nhiên tại mỗi bước.

**"Stochastic" nghĩa là gì?**
> **Stochastic (ngẫu nhiên)** = Không có quy luật, random. Đối lập với "Deterministic" (có quy luật cố định).

**Công thức**: Giống GD, nhưng gradient tính từ 1 mẫu:
```
θ = θ - α × ∇L(θ; x_i; y_i)

Trong đó:
- (x_i, y_i) = một cặp input-output ngẫu nhiên
```

**Nguyên lý hoạt động**:
```
┌─────────────────────────────────────────────────────────────────┐
│  STOCHASTIC GD - Mỗi bước chỉ dùng 1 MẪU                       │
│                                                                 │
│  Dataset: 1,000,000 ảnh 🐱🐱🐱🐱🐱...🐱                         │
│                                                                 │
│  Step 1: Lấy ngẫu nhiên 1 ảnh 🐱                               │
│          █                                                       │
│          ↓ Tính gradient → Cập nhật θ                         │
│                                                                 │
│  Step 2: Lấy ngẫu nhiên 1 ảnh khác 🐱                          │
│            █                                                     │
│            ↓ Tính gradient → Cập nhật θ                        │
│                                                                 │
│  Step 3: Lấy ngẫu nhiên 1 ảnh nữa 🐱                           │
│              █                                                   │
│              ↓ Tính gradient → Cập nhật θ                      │
│                                                                 │
│  ⚡ MỖI BƯỚC = 1 PHÉP TÍNH!                                     │
│  🚀 Rất NHANH nhưng "RUNG RINH"                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Ví dụ dân dã**:
> SGD giống như bạn **bịt mắt** và lùa một đàn bò qua đồi. Mỗi lần bò đi sai hướng, bạn đẩy nhẹ theo hướng ngược lại. Bạn **không cần nhìn toàn cảnh**, chỉ cần phản ứng với từng con bò. Nhanh, nhưng đường đi có thể **RUNG RINH** không theo quỹ đạo mượt mà.

**Ưu điểm**:
- ✅ **Cực kỳ nhanh** - mỗi bước chỉ cần 1 mẫu
- ✅ Có thể thoát khỏi local minima (do nhiễu ngẫu nhiên)
- ✅ Không cần load toàn bộ dữ liệu

**Nhược điểm**:
- ❌ **Dao động mạnh** - đường đi "rung rinh", không mượt
- ❌ Khó hội tụ chính xác - có thể "nhảy qua nhảy lại" quanh điểm tối ưu
- ❌ Cần schedule giảm learning rate

**Khi nào dùng**:
- Dataset lớn
- Online learning (dữ liệu đến liên tục)
- Khi cần thoát local minima

---

### 📒 3. Mini-batch SGD — "Sự thỏa hiệp hoàn hảo"

**Định nghĩa**: Cập nhật tham số sử dụng gradient tính trên **MỘT NHÓM NHỎ** (batch) các mẫu.

**Thường dùng batch_size = 32, 64, 128**

**Công thức**:
```
θ = θ - α × ∇L(θ; x_{i:i+batch}; y_{i:i+batch})
```

**Nguyên lý hoạt động**:
```
┌─────────────────────────────────────────────────────────────────┐
│  MINI-BATCH SGD - Mỗi bước dùng BATCH mẫu (32, 64, 128...)     │
│                                                                 │
│  Dataset: 1,000,000 ảnh 🐱🐱🐱🐱🐱...🐱                         │
│                                                                 │
│  Step 1: Lấy ngẫu nhiên 32 ảnh 🐱🐱🐱🐱🐱🐱🐱🐱🐱🐱🐱...        │
│          ████████████████                                       │
│          ↓ Tính gradient trung bình → Cập nhật θ               │
│                                                                 │
│  Step 2: Lấy 32 ảnh tiếp theo (hoặc shuffle)                   │
│            ████████████████                                     │
│            ↓ Tính gradient trung bình → Cập nhật θ             │
│                                                                 │
│  ⚖️ CÂN BẰNG: Đủ nhanh + Đủ ổn định                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Ví dụ dân dã**:
> Mini-batch SGD giống như bạn đi bộ qua thành phố:
> - Không đi từng bước như SGD (sẽ mất phương hướng)
> - Không cần nhìn toàn cảnh như GD (quá chậm)
> - Thay vào đó, cứ **10 bước** lại nhìn bản đồ một lần để xác nhận hướng đi
> 
> Đây là **tiêu chuẩn công nghiệp** trong Deep Learning!

**Ưu điểm**:
- ✅ **Tốc độ nhanh** - vector hóa được (tính song song trên GPU)
- ✅ **Ổn định hơn SGD** - gradient trung bình từ nhiều mẫu
- ✅ Bộ nhớ vừa phải
- ✅ Tốt cho GPU (vì GPU thích xử lý nhiều dữ liệu cùng lúc)

**Nhược điểm**:
- ❌ Vẫn có thể dao động
- ❌ Cần chọn batch size phù hợp (quá nhỏ = dao động, quá lớn = chậm)

**Khi nào dùng**:
- ✅ **ĐÂY LÀ LỰA CHỌN MẶC ĐỊNH** cho hầu hết các bài toán Deep Learning
- Dataset vừa và lớn
- Huấn luyện trên GPU

---

### 📕 4. Momentum — "Viên bi lấy đà"

**Định nghĩa**: SGD + **"Động lượng"** - tích lũy vận tốc từ các bước trước để đi MƯỢT HƠN.

**"Momentum" nghĩa là gì?**
> **Momentum (động lượng)** = "Quán tính" - vật đang chuyển động sẽ TIẾP TỤC chuyển động theo hướng đó.

**Công thức**:
```
v_t = β × v_{t-1} + (1 - β) × ∇L(θ)    # Cập nhật "vận tốc"
θ = θ - α × v_t                          # Dùng vận tốc để cập nhật

Thường β = 0.9 (90% vận tốc cũ được giữ lại)
```

**Nguyên lý hoạt động**:
```
┌─────────────────────────────────────────────────────────────────┐
│  MOMENTUM - Tích lũy vận tốc qua các bước                      │
│                                                                 │
│  Bước 1: gradient ↓→ v = 0.1↓                                   │
│              ↓                                                  │
│  Bước 2: gradient ↓→ v = 0.9×0.1 + 0.1↓ = 0.19↓                │
│              ↓                                                  │
│  Bước 3: gradient ↓→ v = 0.9×0.19 + 0.1↓ = 0.27↓              │
│              ↓                                                  │
│  Bước 4: gradient ↑→ v = 0.9×0.27 + (-0.1)↑ = 0.14↓           │
│              ↓                                                  │
│          Vẫn đi xuống vì ĐÀ từ các bước trước!                │
│                                                                 │
│  💡 Đỉnh núi nhỏ (local minimum)? ĐỦ ĐÀ → VƯỢT QUA!           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**So sánh SGD vs Momentum**:
```
SGD:        ↓  ↑  ↓  ↑  ↓  ↑  ↓  ↑  ↓    (dao động mạnh)
            
Momentum:   ↓  ↓  ↓  ↓  ↓  ↑  ↓  ↓  ↓    (đi mượt hơn)
            └──────────────────────────►
                   Vẫn giữ đà đi xuống dù gradient đổi hướng
```

**Ví dụ dân dã**:
> Momentum giống như một **viên bi lăn xuống dốc**. Khi gặp một "ổ gà" nhỏ:
> - SGD: Dừng lại, lắc lư ở đáy ổ gà, có thể không thoát ra
> - Momentum: Viên bi đang có VẬN TỐC lớn, nó LƯỚT QUA ổ gà và tiếp tục!
> 
> "Động lượng" giúp vượt qua các cực tiểu địa phương (local minima) nhỏ!

**Ưu điểm**:
- ✅ **Giảm dao động** - đường đi mượt hơn
- ✅ **Nhanh hơn** - tích lũy vận tốc = bước đi hiệu quả hơn
- ✅ **Thoát local minima nhỏ** - có "đà" vượt ổ gà
- ✅ Hội tụ ổn định hơn

**Nhược điểm**:
- ❌ Có thể "quá tốc" - vượt quá điểm tối ưu rồi phải quay lại
- ❌ Thêm 1 hyperparameter (β) cần điều chỉnh

**Khi nào dùng**:
- Dataset có nhiều local minima
- Khi SGD dao động quá nhiều

---

### 📘 5. Nesterov Accelerated Gradient (NAG) — "Momentum thông minh hơn"

**Định nghĩa**: Momentum NHÌN TRƯỚC - tính gradient tại vị trí **DỰ ĐOÁN** thay vì vị trí hiện tại.

**Công thức**:
```
# Momentum thường:
v_t = β × v_{t-1} + α × ∇L(θ)

# Nesterov - Nhìn trước:
v_t = β × v_{t-1} + α × ∇L(θ + β × v_{t-1})  ← Xem trước!
θ = θ - v_t
```

**Nguyên lý hoạt động**:
```
┌─────────────────────────────────────────────────────────────────┐
│  NESTEROV - Nhìn trước rồi mới quyết định                       │
│                                                                 │
│  Vị trí hiện tại: ●                                            │
│  Vận tốc hiện tại: →→→                                         │
│                                                                 │
│  Momentum thường: Tính gradient tại ●                          │
│  Nesterov: Tính gradient tại ●→→→ (vị trí dự đoán)            │
│                                                                 │
│  "Nếu tôi tiếp tục đi theo hướng này, tôi sẽ đến đâu?"         │
│  "À, hóa ra phía trước là DỐC LÊN, tôi nên GIẢM TỐC!"          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**So sánh Momentum vs Nesterov**:
```
Momentum:     ●────→  🌄 Dốc lên! Nhưng đà đã có, vẫn phải leo...
              
Nesterov:     ●────→  🌄 Nhìn thấy dốc trước, GIẢM tốc sớm!
```

**Ví dụ dân dã**:
> Nesterov giống như bạn **lái xe có GPS nhìn trước**. Thay vì:
> - Momentum: Đang lao 100km/h → Gặp dốc → Phanh gấp → Vẫn trượt lên một đoạn
> 
> Nesterov: Đang lao 100km/h → GPS thấy dốc phía trước → **GIẢM TỐC TRƯỚC** → Qua dốc mượt hơn!

**Ưu điểm**:
- ✅ **Thông minh hơn** - phản ứng với gradient thay đổi sớm hơn
- ✅ Hội tụ **nhanh hơn** momentum thường trong nhiều trường hợp
- ✅ Ít "vọt" hơn

**Nhược điểm**:
- ❌ Code phức tạp hơn một chút
- ❌ Ít được dùng phổ biến như Adam

---

### 📓 6. Adam (Adaptive Moment Estimation) — "Thuật toán vạn năng"

**Định nghĩa**: Kết hợp Momentum + **Adaptive Learning Rate** - tự điều chỉnh tốc độ học cho từng tham số.

**"Adaptive" nghĩa là gì?**
> **Adaptive (thích nghi)** = Tự thay đổi theo tình huống. Như đôi giày thông minh tự co giãn theo chân!

**Adam = Momentum + RMSprop**

| Thành phần | Công thức | Ý nghĩa |
|------------|-----------|---------|
| **Moment 1 (m)** | `m_t = β₁ × m_{t-1} + (1-β₁) × g_t` | Tích lũy gradient (như Momentum) |
| **Moment 2 (v)** | `v_t = β₂ × v_{t-1} + (1-β₂) × g_t²` | Bình phương gradient (đo "độ biến động") |
| **Bias correction** | `m̂ = m_t / (1-β₁^t)` | Sửa bias ở các bước đầu |
| **Update** | `θ = θ - α × m̂ / (√v̂ + ε)` | Cập nhật với learning rate thích nghi |

Thường: `β₁ = 0.9`, `β₂ = 0.999`, `ε = 10⁻⁸`

**Nguyên lý hoạt động**:
```
┌─────────────────────────────────────────────────────────────────┐
│  ADAM - Tự điều chỉnh learning rate cho từng tham số         │
│                                                                 │
│  Tham số A: Gradient ổn định → Learning rate CAO               │
│  Tham số B: Gradient biến động mạnh → Learning rate THẤP       │
│                                                                 │
│  m̂ (gradient trung bình - hướng đi)                            │
│  ───────────────────────────────────────                        │
│  √v̂ (biến động - mức độ thay đổi)                             │
│                                                                 │
│  = Đi MẠNH ở tham số ổn định, đi YẾU ở tham số biến động       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Tại sao Adam "vạn năng"?**
```
┌─────────────────────────────────────────────────────────────────┐
│                    TẠI SAO ADAM PHỔ BIẾN NHẤT?                  │
│                                                                 │
│  1️⃣ Tự điều chỉnh learning rate:                               │
│     • Tham số cần thay đổi nhanh → lr cao                      │
│     • Tham số dao động nhiều → lr thấp                          │
│                                                                 │
│  2️⃣ Kết hợp ưu điểm của cả Momentum và RMSprop:               │
│     • Momentum: Vượt dốc, giảm dao động                        │
│     • RMSprop: Adaptive learning rate                          │
│                                                                 │
│  3️⃣ Hoạt động tốt với hầu hết mọi bài toán:                   │
│     • Computer Vision (CNN)                                    │
│     • Natural Language Processing (Transformer)                │
│     • Generative AI                                            │
│     • Reinforcement Learning                                  │
│                                                                 │
│  📊 Adam được dùng trong ~70% các dự án Deep Learning!         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Ví dụ dân dã**:
> Adam giống như một **người hướng dẫn leo núi thông minh**:
> - Như Momentum: Họ nhớ đường đã đi, có "đà" vượt ổ gà
> - Nhưng THÔNG MINH HƠN: Nếu đoạn đường gồ ghề, họ **đI CHẬM LẠI**; đoạn đường bằng phẳng, họ **ĐI NHANH HƠN**
> - Và họ làm điều này TỰ ĐỘNG cho TỪNG BƯỚC CHÂN!

**Ưu điểm**:
- ✅ **Adaptive learning rate** - tự điều chỉnh cho từng tham số
- ✅ **Ít hyperparameter** - chỉ cần điều chỉnh α (thường dùng mặc định 0.001)
- ✅ **Hội tụ nhanh** trong hầu hết các bài toán
- ✅ Hoạt động tốt "out-of-the-box"
- ✅ Kháng nhiễu tốt

**Nhược điểm**:
- ❌ Có thể không hội tụ lý tưởng với một số hàm phức tạp
- ❌ Nghiên cứu gần đây cho thấy có thể hội tụ về local minima không tốt trong một số trường hợp
- ❌ Tốn thêm bộ nhớ cho 2 moment (m, v)

**Khi nào dùng**:
- ✅ **LỰA CHỌN MẶC ĐỊNH** khi không biết dùng gì
- Hầu hết mọi bài toán Deep Learning
- Khi cần kết quả nhanh mà không cần tuning nhiều

---

## 3.5. Các hàm mất mát trong project (Loss Functions)

### 📊 Hàm Himmelblau

**Công thức**: `f(x, y) = (x² + y - 11)² + (x + y² - 7)²`

**Đặc điểm**:
- Có **4 điểm cực tiểu** (local minima)
- Điểm tối ưu thấp nhất (global minimum) = **0** tại 4 vị trí

**Ý nghĩa**: Giống như một **bồn nước có 4 đáy** - thuật toán có thể rơi vào bất kỳ đáy nào tùy điểm khởi tạo.

### 📊 Hàm Rosenbrock (Banana Function)

**Công thức**: `f(x, y) = (a - x)² + b(y - x²)²` (thường a=1, b=100)

**Đặc điểm**:
- Có dạng **"thung lũng cong"** giống chữ U nghiêng
- Điểm tối ưu: `(x, y) = (1, 1)` với `f = 0`
- Đường đi dài, hẹp, cong → Khó hội tụ

**Ý nghĩa**: Thử thách "dốc hẹp" - thuật toán cần đi dọc theo thung lũng rất dài. Momentum/Adam rất hữu ích ở đây!

### 📊 Hàm Booth

**Công thức**: `f(x, y) = (x + 2y - 7)² + (2x + y - 5)²`

**Đặc điểm**:
- Có **1 điểm tối ưu** tại `(1, 3)` với `f = 0`
- Bề mặt tương đối "mềm", dễ hội tụ

**Ý nghĩa**: Hàm đơn giản để test thuật toán ban đầu.

### 📊 Hàm Quadratic

**Công thức**: `f(x, y) = x² + y²`

**Đặc điểm**:
- Hình **paraboloid** đối xứng
- Điểm tối ưu tại `(0, 0)` với `f = 0`
- Đường đồng mức là **hình tròn**

**Ý nghĩa**: Hàm lồi hoàn hảo - luôn hội tụ về global minimum.

---

## 3.6. Các khái niệm quan trọng khác

### 📚 Epoch, Batch, Iteration

| Thuật ngữ | Định nghĩa | Ví dụ |
|-----------|------------|-------|
| **Epoch** | 1 lần duyệt qua **TOÀN BỘ** dataset | 1 epoch = xem hết 1 triệu ảnh |
| **Batch** | Nhóm mẫu dùng trong 1 lần cập nhật | 32 ảnh |
| **Iteration** | 1 lần cập nhật tham số | 1 triệu ảnh / 32 = 31,250 iterations/epoch |

**Ví dụ**: Dataset 1 triệu ảnh, batch size 32:
- 1 Epoch = 1 triệu ảnh = 31,250 iterations
- 10 Epochs = 10 triệu ảnh = 312,500 iterations

### 📚 Learning Rate Schedule

**Learning Rate Schedule** = Lịch trình thay đổi learning rate theo thời gian.

```
LR cao ─────────────────────────────────────────
│        ╲                                      Đầu: học nhanh
│         ╲                                     (bước lớn)
│          ╲____________                       
│                       ╲____                  
│                            ╲____            
│                                 ╲______      
│                                      ╲____ 
│                                           ╲___
└───────────────────────────────────────────────► Epoch
              Learning rate giảm dần
              → Học tinh tế hơn ở cuối
```

**Tại sao cần giảm LR?**
- Đầu: Cần bước LỚN để tiến nhanh
- Cuối: Cần bước NHỎ để không "nhảy qua" điểm tối ưu

---

# KẾT LUẬN

## 🎓 Tổng kết những điều bạn đã học

### Chương 1 - Nền tảng Đồ họa 3D:
- ✅ Hiểu **OpenGL Pipeline**: Vertex Shader → Rasterization → Fragment Shader
- ✅ Hiểu **MVP Matrix**: Model → View → Projection
- ✅ Biết cách GPU xử lý song song hàng triệu điểm

### Chương 2 - 3D Engine:
- ✅ Hiểu cấu trúc file **.obj/.ply**: Vertices, Normals, Faces, UV
- ✅ Hiểu **Texture Mapping**: UV coordinates như "tọa độ trên ảnh"
- ✅ Nắm vững **Blinn-Phong**: Ambient + Diffuse + Specular
- ✅ Hiểu **Depth Map/Z-Buffer**: Camera nhớ khoảng cách để quyết định pixel nào hiển thị

### Chương 3 - SGD Optimization:
- ✅ Hiểu **Loss Surface**: Bề mặt 3D cần tìm điểm thấp nhất
- ✅ Hiểu **Gradient**: La bàn chỉ hướng đi xuống
- ✅ So sánh được 6 thuật toán: GD, SGD, Mini-batch, Momentum, Nesterov, Adam
- ✅ Biết khi nào dùng thuật toán nào

## 💪 Lời nhắn cuối cùng

> Các bạn thân mến,
> 
> Đồ họa máy tính và AI nghe có vẻ phức tạp, nhưng bạn đã làm được ĐẦU TIÊN - đó là **HIỂU** những khái niệm cốt lõi. Điều này không hề dễ dàng!
> 
> Bây giờ, khi bạn đọc code trong project, hãy nhớ:
> - **Vertex Shader** = "Tôi xử lý từng điểm đỉnh"
> - **Fragment Shader** = "Tôi tô màu từng pixel"
> - **MVP Matrix** = "Tôi đặt vật vào thế giới, rồi chiếu lên màn hình"
> - **Gradient** = "Tôi chỉ hướng đi xuống"
> - **Adam** = "Tôi thông minh, tự điều chỉnh"
> 
> Bạn đã có kiến thức nền tảng vững chắc. Giờ là lúc **ÁP DỤNG**!
> 
> Chúc các bạn hoàn thành tốt đồ án 💪🚀

---

**Tài liệu được viết với ❤️ bởi Giáo sư AI (giả lập)**

*"Học một lần, hiểu mãi mãi"* — Nhưng đừng quên thực hành!
