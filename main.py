import sys
import os
import traceback

# Đưa thư mục gốc của project vào sys.path để các file nội bộ import được lẫn nhau
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controller import AppController

def main():
    try:
        # Tạo bộ điều phối chính của chương trình.
        # Bên trong controller sẽ tự dựng model, viewer và đăng ký callback input.
        controller = AppController()

        # Đi vào vòng lặp chính: nhận input, cập nhật state và render từng frame.
        controller.run()
    except KeyboardInterrupt:
        print("\nỨng dụng bị ngắt bởi người dùng")
    except Exception as e:
        # Nếu lỗi thật sự xảy ra thì in cả message lẫn stack trace để debug nhanh.
        print(f"Lỗi: {e}")
        traceback.print_exc()
    finally:
        print("Ứng dụng đã đóng")

if __name__ == "__main__":
    main()
