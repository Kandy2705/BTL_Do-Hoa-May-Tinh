# BTL Đồ Họa Máy Tính

## Mô tả dự án
Dự án đồ họa máy tính sử dụng OpenGL và Dear ImGui để hiển thị các đối tượng 3D với các shader khác nhau. Để xem chi tiết: [Xem chi tiết dự án tại đây](./ChiTiet.md)
## Yêu cầu hệ thống
- Python 3.7+
- OpenGL
- GLFW
- Dear ImGui (PyImGui)
- NumPy
- PyOpenGL

## Cài đặt
Sử dụng pip để cài đặt các thư viện cần thiết:
```bash
pip install glfw imgui[glfw] PyOpenGL PyOpenGL_accelerate numpy
```

## Cách chạy dự án

1. Mở terminal hoặc command prompt
2. Di chuyển đến thư mục dự án:
```bash
cd "Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT"
```

3. Chạy file chính:
```bash
python main.py
```
hoặc
```bash
python3 main.py
```

## Điều khiển
- **Chuột trái + Kéo**: Xoay camera
- **Cuộn chuột**: Zoom in/out
- **Phím W**: Chuyển đổi chế độ hiển thị (wireframe/solid)
- **Phím S**: Chuyển đổi shader
- **Phím Q**: Thoát ứng dụng
- **Giao diện ImGui**: Chọn hình dạng từ menu dropdown

## Cấu trúc thư mục
- `main.py`: File chính để khởi chạy ứng dụng
- `controller.py`: Điều khiển logic và xử lý input
- `model.py`: Quản lý dữ liệu và các đối tượng 3D
- `viewer.py`: Hiển thị OpenGL và rendering
- `geometry/`: Chứa các file định nghĩa hình học
- `shaders/`: Chứa các file shader (vertex, fragment)
- `libs/`: Các thư viện hỗ trợ
- `assets/`: Tài nguyên hình ảnh và models
