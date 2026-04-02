#version 330 core

// Thuộc tính đầu vào từ CPU cho mỗi vertex
layout(location = 0) in vec3 position;  // Tọa độ vertex trong không gian object
layout(location = 1) in vec3 color;     // Màu gốc của vertex

// Biến toàn cục từ CPU
uniform mat4 projection, modelview;     // Ma trận chiếu phối và nhìn

// Output cho fragment shader
out vec3 normal_interp;  // Vector pháp tuyến đã biến đổi
out vec3 vertPos;        // Vị trí vertex trong không gian thế giới
out vec3 colorInterp;    // Màu gốc để truyền đi

void main(){
  // Truyền màu gốc cho fragment shader
  colorInterp = color;
  
  // Biến đổi tọa độ vertex sang không gian thế giới
  vec4 vertPos4 = modelview * vec4(position, 1.0);
  vertPos = vec3(vertPos4) / vertPos4.w;  // Chia w để hiệu ứng phối cảnh

  // Biến đổi vector pháp tuyến
  // Với cube, pháp tuyến bằng vị trí đã chuẩn hóa
  mat4 normal_matrix = transpose(inverse(modelview));
  normal_interp = vec3(normal_matrix * vec4(normalize(position), 0.0));

  // Biến đổi tọa độ sang không gian màn hình
  gl_Position = projection * vertPos4;
}
