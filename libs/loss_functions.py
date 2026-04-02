import numpy as np

class OptimizationFunction:
    def __init__(self, name, equation_str, domain_range):
        self.name = name
        self.equation_str = equation_str # Dùng để truyền cho bộ vẽ3D (math_surface3d)
        self.domain_range = domain_range # Giới hạn không gian x, y hợp lý để vẽ cho đẹp

    def compute(self, x, y):
        """Tính giá trị f(x, y) - Độ cao Z"""
        pass

    def gradient(self, x, y):
        """Tính Đạo hàm [df/dx, df/dy] - Hướng dốc"""
        pass

# ==========================================
# 1. HÀM HIMMELBLAU
# ==========================================
# Đây là một hàm benchmark kinh điển trong tối ưu không lồi.
# Điểm hay của nó là có nhiều cực tiểu cục bộ, nên rất hợp để minh họa
# hành vi khác nhau của các thuật toán tối ưu.
class Himmelblau(OptimizationFunction):
    def __init__(self):
        super().__init__("Himmelblau", "(x**2 + y - 11)**2 + (x + y**2 - 7)**2", (-5.0, 5.0))

    def compute(self, x, y):
        return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

    def gradient(self, x, y):
        # Gradient ở đây đến từ đạo hàm riêng của hàm:
        # f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
        # Ta lấy:
        # df/dx theo quy tắc dây chuyền
        # df/dy theo quy tắc dây chuyền
        # rồi ghép lại thành vector [df/dx, df/dy].
        dx = 4 * x * (x**2 + y - 11) + 2 * (x + y**2 - 7)
        dy = 2 * (x**2 + y - 11) + 4 * y * (x + y**2 - 7)
        return np.array([dx, dy])

# ==========================================
# 2. HÀM ROSENBROCK (Banana Function) với a=1, b=100
# ==========================================
# Rosenbrock còn gọi là "banana function" vì thung lũng tối ưu của nó cong.
# Hàm này rất hay để test optimizer vì minimum thì đơn giản,
# nhưng đường đi đến minimum lại khó.
class Rosenbrock(OptimizationFunction):
    def __init__(self):
        super().__init__("Rosenbrock", "(1 - x)**2 + 100 * (y - x**2)**2", (-2.0, 2.0))

    def compute(self, x, y):
        return (1.0 - x)**2 + 100.0 * (y - x**2)**2

    def gradient(self, x, y):
        # Gradient được đạo hàm từ:
        # f(x, y) = (1 - x)^2 + 100 (y - x^2)^2
        # Điểm cần nhớ để giải thích:
        # - vế đầu kéo x về 1
        # - vế sau ép y tiến gần x^2
        dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x**2)
        dy = 200.0 * (y - x**2)
        return np.array([dx, dy])

# ==========================================
# 3. HÀM BOOTH
# ==========================================
# Booth là một hàm lồi 2 biến khá "sạch", thường dùng để minh họa
# bài toán tối ưu 2D vì chỉ có một cực tiểu toàn cục.
class Booth(OptimizationFunction):
    def __init__(self):
        super().__init__("Booth", "(x + 2*y - 7)**2 + (2*x + y - 5)**2", (-10.0, 10.0))

    def compute(self, x, y):
        return (x + 2*y - 7)**2 + (2*x + y - 5)**2

    def gradient(self, x, y):
        # Gradient của Booth cũng là đạo hàm riêng trực tiếp từ hai bình phương:
        # f(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2
        dx = 2 * (x + 2*y - 7) + 4 * (2*x + y - 5)
        dy = 4 * (x + 2*y - 7) + 2 * (2*x + y - 5)
        return np.array([dx, dy])

# ==========================================
# 4. HÀM QUADRATIC (Hình cái Bát đơn giản)
# ==========================================
# Đây là hàm đơn giản nhất để minh họa gradient descent:
# f(x, y) = x^2 + y^2
# Bề mặt là một cái bát parabol, minimum nằm ở gốc tọa độ.
class Quadratic(OptimizationFunction):
    def __init__(self):
        super().__init__("Quadratic 2D", "x**2 + y**2", (-5.0, 5.0))

    def compute(self, x, y):
        return x**2 + y**2

    def gradient(self, x, y):
        # Vì f = x^2 + y^2 nên gradient là [2x, 2y].
        return np.array([2.0*x, 2.0*y])

# Từ điển để gọi nhanh
LOSS_FUNCTIONS = {
    "Himmelblau": Himmelblau(),
    "Rosenbrock": Rosenbrock(),
    "Booth": Booth(),
    "Quadratic 2D": Quadratic()
}
