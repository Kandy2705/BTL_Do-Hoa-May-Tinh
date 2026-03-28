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
class Himmelblau(OptimizationFunction):
    def __init__(self):
        super().__init__("Himmelblau", "(x**2 + y - 11)**2 + (x + y**2 - 7)**2", (-5.0, 5.0))

    def compute(self, x, y):
        return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

    def gradient(self, x, y):
        dx = 4 * x * (x**2 + y - 11) + 2 * (x + y**2 - 7)
        dy = 2 * (x**2 + y - 11) + 4 * y * (x + y**2 - 7)
        return np.array([dx, dy])

# ==========================================
# 2. HÀM ROSENBROCK (Banana Function) với a=1, b=100
# ==========================================
class Rosenbrock(OptimizationFunction):
    def __init__(self):
        super().__init__("Rosenbrock", "(1 - x)**2 + 100 * (y - x**2)**2", (-2.0, 2.0))

    def compute(self, x, y):
        return (1.0 - x)**2 + 100.0 * (y - x**2)**2

    def gradient(self, x, y):
        dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x**2)
        dy = 200.0 * (y - x**2)
        return np.array([dx, dy])

# ==========================================
# 3. HÀM BOOTH
# ==========================================
class Booth(OptimizationFunction):
    def __init__(self):
        super().__init__("Booth", "(x + 2*y - 7)**2 + (2*x + y - 5)**2", (-10.0, 10.0))

    def compute(self, x, y):
        return (x + 2*y - 7)**2 + (2*x + y - 5)**2

    def gradient(self, x, y):
        dx = 2 * (x + 2*y - 7) + 4 * (2*x + y - 5)
        dy = 4 * (x + 2*y - 7) + 2 * (2*x + y - 5)
        return np.array([dx, dy])

# ==========================================
# 4. HÀM QUADRATIC (Hình cái Bát đơn giản)
# ==========================================
class Quadratic(OptimizationFunction):
    def __init__(self):
        super().__init__("Quadratic 2D", "x**2 + y**2", (-5.0, 5.0))

    def compute(self, x, y):
        return x**2 + y**2

    def gradient(self, x, y):
        return np.array([2.0*x, 2.0*y])

# Từ điển để gọi nhanh
LOSS_FUNCTIONS = {
    "Himmelblau": Himmelblau(),
    "Rosenbrock": Rosenbrock(),
    "Booth": Booth(),
    "Quadratic 2D": Quadratic()
}
