#version 330 core

// Thuộc tính đầu vào từ CPU cho mỗi vertex
layout(location = 0) in vec3 position;    // Tọa độ vertex trong không gian object
layout(location = 1) in vec3 color;       // Màu gốc của vertex

// Các biến toàn cục (uniforms) từ CPU
uniform mat4 projection, modelview;          // Ma trận chiếu phối và nhìn
uniform mat3 K_materials;                   // Thuộc tính vật liệu [môi trường, khuếch tán, phản chiếu]
uniform mat3 I_light;                      // Cường độ ánh sáng [môi trường, khuếch tán, phản chiếu]
uniform float shininess;                     // Độ bóng của bề mặt vật liệu
uniform vec3 light_pos;                     // Vị trí nguồn sáng

out vec3 colorInterp;  // Màu đầu ra cho fragment shader (sau khi nội suy)

void main(){
  // BƯỚC 1: Biến đổi tọa độ vertex sang không gian thế giới
  vec4 vertPos4 = modelview * vec4(position, 1.0);
  vec3 vertPos = vec3(vertPos4) / vertPos4.w;  // vật xa thì nhìn nhỏ hơn
  // vật gần thì nhìn to hơn
  
  // BƯỚC 2: Biến đổi vector pháp tuyến
  // Với cube, pháp tuyến bằng vị trí đã chuẩn hóa vì cube nằm tại gốc tọa độ
  mat4 normal_matrix = transpose(inverse(modelview));
  vec3 N = normalize(vec3(normal_matrix * vec4(normalize(position), 0.0)));
  
  // BƯỚC 3: Tính các vector chiếu sáng
  vec3 L = normalize(light_pos - vertPos);     // Vector từ vertex đến nguồn sáng
  vec3 R = reflect(-L, N);                  // Vector phản chiếu của ánh sáng reflect(I,N)=I−2(I⋅N)N
  vec3 V = normalize(-vertPos);               // Vector từ vertex đến máy ảnh
  
  // BƯỚC 4: Tính các thành phần chiếu sáng theo mô hình Phong
  float NdotL = max(dot(N, L), 0.0);       // Cường độ khuếch tán
  float specAngle = max(dot(R, V), 0.0);      // Góc phản chiếu
  float specular = pow(specAngle, shininess);    // Cường độ phản chiếu
  
  // BƯỚC 5: Kết hợp các thành phần chiếu sáng
  vec3 g = vec3(NdotL, specular, 1.0);      // [khuếch tán, phản chiếu, môi trường]
  vec3 lighting = matrixCompMult(K_materials, I_light) * g;
  
  // BƯỚC 6: Kết hợp chiếu sáng với màu gốc của vertex
  colorInterp = 0.5 * lighting + 0.5 * color;
  
  // BƯỚC 7: Biến đổi tọa độ sang hệ tọa độ màn hình
  gl_Position = projection * vertPos4;
}
