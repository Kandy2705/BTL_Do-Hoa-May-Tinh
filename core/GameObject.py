import uuid
from enum import Enum

class GameObject:
    def __init__(self, name: str):
        self.id = uuid.uuid4()
        self.name = name
        self.type = Type.GAMEOBJECTDEF

        self.shader = 0
        self.color = [1.0, 1.0, 1.0, 0.0]
        self.texture_filename = ""
        self.rendering_mode = "flat_color"
        self.visible = True

        self.position = [0.0, 0.0, 0.0]
        self.rotation = [0.0, 0.0, 0.0]
        self.scale = [1.0, 1.0, 1.0]

class GameObjectMath(GameObject):
    def __init__(self, name: str):
        super().__init__(name)
        self.type = Type.GAMEOBJECTMATH
        self.math_script = "(x**2 + y - 11)**2 + (x + y**2 - 7)**2"

class GameObjectOBJ(GameObject):
    def __init__(self, name: str):
        super().__init__(name)
        self.type = Type.GAMEOBJECTOBJ
        self.obj_ply_file = ""

class GameObjectLight(GameObject):
    def __init__(self, name: str):
        super().__init__(name)
        self.type = Type.LIGHT
        self.light_type = None
        self.light_color = [1.0, 1.0, 1.0]
        self.light_intensity = 1.0

class GameObjectCamera(GameObject):
    def __init__(self, name: str):
        super().__init__(name)
        self.type = Type.CAMERA
        self.camera_fov = 60.0
        self.camera_near = 0.1
        self.camera_far = 100.0


class Type(Enum):
    GAMEOBJECTDEF = 0
    GAMEOBJECTMATH = 1
    GAMEOBJECTOBJ = 2
    LIGHT = 3
    CAMERA = 4
