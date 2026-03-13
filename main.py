from viewer import Viewer
from geometry.cube3d import Cube

def main():
    app = Viewer(width=1280, height=720)
    # Khởi tạo và thêm một hình 3D vào scene
    cube = Cube("./shaders/color_interp.vert", "./shaders/color_interp.frag").setup()
    app.add(cube)
    app.run()

if __name__ == "__main__":
    main()