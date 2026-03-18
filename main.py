import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controller import AppController

def main():
    try:
        controller = AppController()
        controller.run()
    except KeyboardInterrupt:
        print("\nỨng dụng bị ngắt bởi người dùng")
    except Exception as e:
        print(f"Lỗi: {e}")
    finally:
        print("Ứng dụng đã đóng")

if __name__ == "__main__":
    main()
