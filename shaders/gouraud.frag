#version 330 core

precision mediump float;  // Độ chính xác trung bình cho floating point

// Input từ vertex shader (đã được nội suy giữa các vertices)
in vec3 colorInterp;  // Màu đã tính lighting ở vertex shader

// Output màu cuối cùng cho screen
out vec4 out_color;

void main() {
  // Gouraud shading: lighting đã được tính ở vertex shader
  // Fragment shader chỉ cần output màu đã nội suy
  // → Mượt hơn flat shading nhưng ít chính xác hơn Phong shading
  out_color = vec4(colorInterp, 1.0);  // Thêm alpha = 1.0
}
