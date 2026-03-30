# pylint: disable=invalid-name, bad-whitespace, too-many-arguments
"""
Công cụ hình học cơ bản để bổ trợ cho numpy
Quaternion, ma trận 4x4 đồ họa, và các tiện ích vector.
@author: franco
"""
# Python built-in modules
import math                 # chủ yếu cho các hàm lượng giác
from numbers import Number  # hữu ích để kiểm tra kiểu của tham số: vô hướng hay vector?

# external module
import numpy as np          # ma trận, vector & quaternion là mảng numpy


# Một số hàm hữu ích trên vectors -------------------------------------------
def vec(*iterable):
    """ Tạo numpy vector từ bất kỳ iterable(tuple...) hoặc vector nào """
    # Nếu có nhiều hơn 1 phần tử, trả về toàn bộ iterable
    # Nếu chỉ có 1 phần tử, trả về phần tử đó (tránh mảng lồng nhau)
    return np.asarray(iterable if len(iterable) > 1 else iterable[0], 'f')


def normalized(vector):
    """ Chuẩn hóa vector, có kiểm tra chia không """
    # Tính độ dài (norm) của vector: sqrt(x² + y² + z²)
    norm = math.sqrt(sum(vector*vector))
    # Nếu norm > 0, trả về vector chuẩn hóa, ngược lại trả về vector gốc
    return vector / norm if norm > 0. else vector


def lerp(point_a, point_b, fraction):
    """ Nội suy tuyến tính giữa hai điểm có các toán tử tuyến tính """
    # Công thức: A + t*(B - A), với t là tỷ lệ (0 <= t <= 1)
    # t=0 -> point_a, t=1 -> point_b, t=0.5 -> điểm giữa
    return point_a + fraction * (point_b - point_a)


# Các tiện ích ma trận 4x4 điển hình cho OpenGL ------------------------------------
def identity():
    """ Ma trận đơn vị 4x4 """
    # Ma trận đơn vị có 1 trên đường chéo chính và 0 ở các vị trí còn lại
    return np.identity(4, 'f')


def ortho(left, right, bot, top, near, far):
    """ Ma trận chiếu trực giao cho OpenGL """
    # Chiếu trực giao: các đường song song vẫn song song sau khi chiếu
    # Dùng cho 2D hoặc các ứng dụng kỹ thuật
    dx, dy, dz = right - left, top - bot, far - near
    rx, ry, rz = -(right+left) / dx, -(top+bot) / dy, -(far+near) / dz
    return np.array([[2/dx, 0,    0,     rx],
                     [0,    2/dy, 0,     ry],
                     [0,    0,    -2/dz, rz],
                     [0,    0,    0,     1]], 'f')


def perspective(fovy, aspect, near, far):
    """ Ma trận chiếu phối cảnh, từ trường nhìn và tỷ lệ khung hình """
    # Chiếu phối cảnh: tạo hiệu ứng 3D với đối tượng xa nhỏ hơn đối tượng gần
    # fovy: trường nhìn theo chiều dọc (độ)
    # aspect: tỷ lệ khung hình (rộng/cao)
    _scale = 1.0/math.tan(math.radians(fovy)/2.0)
    sx, sy = _scale / aspect, _scale
    zz = (far + near) / (near - far)
    zw = 2 * far * near/(near - far)
    return np.array([[sx, 0,  0,  0],
                     [0,  sy, 0,  0],
                     [0,  0, zz, zw],
                     [0,  0, -1,  0]], 'f')


def frustum(xmin, xmax, ymin, ymax, zmin, zmax):
    """ Ma trận chiếu frustum cho OpenGL, từ tọa độ min và max"""
    # Frustum: hình chóp cụt - một dạng chiếu phối cảnh tổng quát hơn
    # Xác định khối nhìn bằng 6 mặt phẳng: trái, phải, dưới, trên, gần, xa
    a = (xmax+xmin) / (xmax-xmin)
    b = (ymax+ymin) / (ymax-ymin)
    c = -(zmax+zmin) / (zmax-zmin)
    d = -2*zmax*zmin / (zmax-zmin)
    sx = 2*zmin / (xmax-xmin)
    sy = 2*zmin / (ymax-ymin)
    return np.array([[sx, 0,  a, 0],
                     [0, sy,  b, 0],
                     [0,  0,  c, d],
                     [0,  0, -1, 0]], 'f')


def translate(x=0.0, y=0.0, z=0.0):
    """ Ma trận dịch chuyển từ tọa độ (x,y,z) hoặc một vector x """
    matrix = np.identity(4, 'f')
    # Gán tọa độ dịch chuyển vào cột cuối cùng của ma trận
    matrix[:3, 3] = vec(x, y, z) if isinstance(x, Number) else vec(x)
    return matrix


def scale(x, y=None, z=None):
    """ Ma trận tỷ lệ, với tỷ lệ đồng nhất (chỉ x) hoặc theo từng chiều (x,y,z) """
    x, y, z = (x, y, z) if isinstance(x, Number) else (x[0], x[1], x[2])
    # Nếu chỉ có x, áp dụng tỷ lệ đồng nhất cho cả 3 chiều
    y, z = (x, x) if y is None or z is None else (y, z)
    # Ma trận đường chéo với các giá trị tỷ lệ
    return np.diag((x, y, z, 1))


def sincos(degrees=0.0, radians=None):
    """ Tiện ích quay nhanh để tính sin và cos của một góc """
    radians = radians if radians else math.radians(degrees)
    return math.sin(radians), math.cos(radians)


def rotate(axis=(1., 0., 0.), angle=0.0, radians=None):
    """ Ma trận quay 4x4 quanh 'axis' với 'angle' độ hoặc 'radians' """
    # Sử dụng công thức Rodrigues' rotation formula
    x, y, z = normalized(vec(axis))  # Chuẩn hóa trục quay
    s, c = sincos(angle, radians)     # sin và cos của góc quay
    nc = 1 - c
    return np.array([[x*x*nc + c,   x*y*nc - z*s, x*z*nc + y*s, 0],
                     [y*x*nc + z*s, y*y*nc + c,   y*z*nc - x*s, 0],
                     [x*z*nc - y*s, y*z*nc + x*s, z*z*nc + c,   0],
                     [0,            0,            0,            1]], 'f')


def lookat(eye, target, up):
    """ Tính ma trận xem 4x4 từ điểm 'eye' đến 'target',
        vector 'up' 3d cố định hướng """
    # Tạo hệ tọa độ camera:
    # view: hướng từ mắt đến mục tiêu (trục Z của camera)
    # right: vector phải (trục X của camera) 
    # up: vector trên (trục Y của camera)
    view = normalized(vec(target)[:3] - vec(eye)[:3])
    up = normalized(vec(up)[:3])
    right = np.cross(view, up)
    up = np.cross(right, view)
    rotation = np.identity(4)
    rotation[:3, :3] = np.vstack([right, up, -view])
    # Kết hợp rotation và translation (di chuyển camera ngược lại)
    return rotation @ translate(-eye)


# các hàm quaternion -------------------------------------------------------
def quaternion(x=vec(0., 0., 0.), y=0.0, z=0.0, w=1.0):
    """ Khởi tạo quaternion, w=phần thực và, x,y,z hoặc vector x phần ảo """
    # Quaternion: w + xi + yj + zk, với i,j,k là đơn vị ảo
    x, y, z = (x, y, z) if isinstance(x, Number) else (x[0], x[1], x[2])
    return np.array((w, x, y, z), 'f')


def quaternion_from_axis_angle(axis, degrees=0.0, radians=None):
    """ Tính quaternion từ một trục vector và góc quay quanh trục này """
    # Chuyển đổi biểu diễn axis-angle sang quaternion
    sin, cos = sincos(radians=radians*0.5) if radians else sincos(degrees*0.5)
    return quaternion(normalized(vec(axis))*sin, w=cos)


def quaternion_from_euler(yaw=0.0, pitch=0.0, roll=0.0, radians=None):
    """ Tính quaternion từ ba góc Euler theo độ hoặc radian """
    # Chuyển đổi góc Euler (yaw, pitch, roll) sang quaternion
    # yaw: quay quanh trục Y, pitch: quay quanh trục X, roll: quay quanh trục Z
    siy, coy = sincos(yaw * 0.5, radians[0] * 0.5 if radians else None)
    sir, cor = sincos(roll * 0.5, radians[1] * 0.5 if radians else None)
    sip, cop = sincos(pitch * 0.5, radians[2] * 0.5 if radians else None)
    return quaternion(x=coy*sir*cop - siy*cor*sip, y=coy*cor*sip + siy*sir*cop,
                      z=siy*cor*cop - coy*sir*sip, w=coy*cor*cop + siy*sir*sip)


def quaternion_mul(q1, q2):
    """ Tính quaternion kết hợp các phép quay của hai quaternion """
    # Nhân quaternion: tích của hai quaternion tương ứng với kết hợp các phép quay
    return np.dot(np.array([[q1[0], -q1[1], -q1[2], -q1[3]],
                            [q1[1],  q1[0], -q1[3],  q1[2]],
                            [q1[2],  q1[3],  q1[0], -q1[1]],
                            [q1[3], -q1[2],  q1[1],  q1[0]]]), q2)


def quaternion_matrix(q):
    """ Tạo ma trận quay 4x4 từ quaternion q """
    # Chuyển đổi quaternion sang ma trận quay 4x4
    q = normalized(q)  # chỉ các quaternion đơn vị mới là các phép quay hợp lệ.
    nxx, nyy, nzz = -q[1]*q[1], -q[2]*q[2], -q[3]*q[3]
    qwx, qwy, qwz = q[0]*q[1], q[0]*q[2], q[0]*q[3]
    qxy, qxz, qyz = q[1]*q[2], q[1]*q[3], q[2]*q[3]
    return np.array([[2*(nyy + nzz)+1, 2*(qxy - qwz),   2*(qxz + qwy),   0],
                     [2 * (qxy + qwz), 2 * (nxx + nzz) + 1, 2 * (qyz - qwx), 0],
                     [2 * (qxz - qwy), 2 * (qyz + qwx), 2 * (nxx + nyy) + 1, 0],
                     [0, 0, 0, 1]], 'f')


def quaternion_slerp(q0, q1, fraction):
    """ Nội suy cầu phương của hai quaternion theo 'fraction' """
    # SLERP: Spherical Linear Interpolation - nội suy trơn trên đường tròn đơn vị
    # chỉ các quaternion đơn vị mới là các phép quay hợp lệ.
    q0, q1 = normalized(q0), normalized(q1)
    dot = np.dot(q0, q1)

    # nếu tích vô hướng âm, các quaternion có tính thuận ngược nhau
    # và slerp sẽ không lấy đường ngắn nhất. Sửa bằng cách đảo ngược một quaternion.
    q1, dot = (q1, dot) if dot > 0 else (-q1, -dot)

    theta_0 = math.acos(np.clip(dot, -1, 1))  # góc giữa các vector đầu vào
    theta = theta_0 * fraction                # góc giữa q0 và kết quả
    q2 = normalized(q1 - q0*dot)              # {q0, q2} now orthonormal basis

    return q0*math.cos(theta) + q2*math.sin(theta)


# một class trackball dựa trên các hàm quaternion được cung cấp -------------------
class Trackball:
    """Trackball ảo để xem cảnh 3D. Độc lập với hệ thống window."""

    def __init__(self, yaw=0., roll=0., pitch=0., distance=3., radians=None):
        """ Xây dựng trackball mới với chế độ xem được chỉ định, góc theo độ """
        # Khởi tạo quaternion từ góc Euler
        self.rotation = quaternion_from_euler(yaw, roll, pitch, radians)
        self.distance = max(distance, 0.001)  # khoảng cách tối thiểu để tránh lỗi
        self.pos2d = vec(0.0, 0.0)  # vị trí 2D cho pan

    def drag(self, old, new, winsize):
        """ Di chuyển trackball từ vị trí cũ sang mới 2d đã chuẩn hóa windows """
        # Chuẩn hóa tọa độ chuột về [-1, 1]
        old, new = ((2*vec(pos) - winsize) / winsize for pos in (old, new))
        # Cập nhật quaternion quay
        self.rotation = quaternion_mul(self._rotate(old, new), self.rotation)

    def zoom(self, delta, size):
        """ Thu phóng trackball theo hệ số delta được chuẩn hóa bởi kích thước windows """
        # delta: sự thay đổi chuột, size: kích thước window
        self.distance = max(0.001, self.distance * (1 - 50*delta/size))

    def pan(self, dx, dy):
        """ Dịch chuyển trong tham chiếu camera theo dx, dy """
        # Dịch chuyển 2D tỷ lệ với khoảng cách camera
        self.pos2d += vec(dx, dy) * 0.001 * self.distance

    def view_matrix(self):
        """ Ma trận xem, bao gồm quay, dịch chuyển và thu phóng """
        # Tạo ma trận quay từ quaternion
        rotation = quaternion_matrix(self.rotation)
        # Tạo ma trận dịch chuyển (lùi camera ra xa)
        translation = translate(0, 0, -self.distance)
        # Tạo ma trận pan (dịch chuyển ngang)
        pan = translate(self.pos2d[0], self.pos2d[1], 0)
        # Kết hợp: pan * translation * rotation
        return pan @ translation @ rotation

    def projection_matrix(self, winsize):
        """ Dùng FOV, Near, Far có thể tùy chỉnh thay vì Fix cứng """
        # Lấy giá trị nếu có, nếu không thì dùng mặc định của Unity (60, 0.1, 1000)
        fov = getattr(self, 'fov', 60.0)
        near = getattr(self, 'near', 0.1)
        far = getattr(self, 'far', 1000.0)
        
        return perspective(fov, winsize[0] / winsize[1], near, far)

    def _rotate(self, p1, p2):
        """ Tính quaternion quay từ hai điểm trên mặt cầu đơn vị """
        # Tính vector giữa hai điểm
        p1, p2 = normalized(vec(p1)), normalized(vec(p2))
        # Vector vuông góc với cả hai
        axis = np.cross(p1, p2)
        # Góc quay giữa hai vector
        angle = math.acos(np.clip(np.dot(p1, p2), -1, 1))
        # Trả về quaternion quay
        return quaternion_from_axis_angle(axis, radians=angle)

    def matrix(self):
        """ Rotational component of trackball position """
        return quaternion_matrix(self.rotation)

    def _project3d(self, position2d, radius=0.8):
        """ Project x,y on sphere OR hyperbolic sheet if away from center """
        p2, r2 = sum(position2d*position2d), radius*radius
        zcoord = math.sqrt(r2 - p2) if 2*p2 < r2 else r2 / (2*math.sqrt(p2))
        return vec(*position2d, zcoord)

    def _rotate(self, old, new):
        """ Rotation of axis orthogonal to old & new's 3D ball projections """
        old, new = (normalized(self._project3d(pos)) for pos in (old, new))
        phi = 2 * math.acos(np.clip(np.dot(old, new), -1, 1))
        return quaternion_from_axis_angle(np.cross(old, new), radians=phi)
