# Chu Thich Code Bao Cao

File nay la ban "doc code va giai thich" de ban mo code tren man hinh va noi bang loi cua minh.
Muc tieu cua file nay khong phai viet lai toan bo source, ma la chi ro:

- file nay dong vai tro gi
- moi dong/nhom dong quan trong dang lam gi
- khi thay hoi thi nen mo dong nao va noi cau gi

---

## 1. [main.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/main.py#L1)

### Vai tro tong quat

`main.py` la diem vao cua chuong trinh. File nay rat ngan, nhung no quyet dinh ung dung bat dau tu dau, tao controller nao, va khi loi xay ra thi in ra nhu the nao.

### Giai thich tung dong

- Dong 1: `import sys`
  Dung de thao tac voi moi truong Python, o day chu yeu de chinh `sys.path`.

- Dong 2: `import os`
  Dung de lay duong dan thu muc hien tai cua project.

- Dong 3: `import traceback`
  Dung de in chi tiet stack trace neu chuong trinh loi, de luc debug de thay duoc no vo o dau.

- Dong 5: `sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))`
  Dong nay dua thu muc goc cua project vao dau danh sach tim module.
  Noi ngan gon voi thay: "Em chen root folder cua project vao `sys.path` de Python import duoc cac file noi bo nhu `controller.py`, `model.py`, `viewer.py`."

- Dong 7: `from controller import AppController`
  Nhap lop dieu phoi chinh cua chuong trinh.
  Day la lop giu vai tro "bo nao" cua app.

- Dong 9: `def main():`
  Tao ham `main` de gom toan bo logic khoi dong vao mot cho ro rang.

- Dong 10: `try:`
  Dung khoi `try` de neu co loi thi chuong trinh van thong bao duoc ro rang.

- Dong 11: `controller = AppController()`
  Tao mot doi tuong controller.
  Luc nay ben trong controller se tao `Viewer`, `Model`, coordinate system va dang ky callback.

- Dong 12: `controller.run()`
  Day moi la lenh chay that su.
  Khi ham nay bat dau, app se vao vong lap chinh va render lien tuc tung frame.

- Dong 13: `except KeyboardInterrupt:`
  Neu nguoi dung dung app bang `Ctrl + C` thi xu ly rieng.

- Dong 14: `print("\\nỨng dụng bị ngắt bởi người dùng")`
  In ra thong bao de biet app bi dung chu dong, khong phai crash.

- Dong 15: `except Exception as e:`
  Bat moi loi con lai.

- Dong 16: `print(f"Lỗi: {e}")`
  In ra thong diep loi ngan gon.

- Dong 17: `traceback.print_exc()`
  In stack trace day du.
  Cau noi de bao cao: "Neu co bug, em khong chi in ten loi ma in ca vet goi ham de debug nhanh hon."

- Dong 18: `finally:`
  Du loi hay khong thi van chay doan nay.

- Dong 19: `print("Ứng dụng đã đóng")`
  Bao cho terminal biet ung dung da ket thuc.

- Dong 21: `if __name__ == "__main__":`
  Kiem tra file nay co dang duoc chay truc tiep hay chi dang duoc import.

- Dong 22: `main()`
  Neu dang chay truc tiep thi goi ham `main`.

### Cach noi nhanh khi show file nay

"File `main.py` chi lam 3 viec: chuan bi duong dan import, tao `AppController`, va goi `run()` de vao main loop. Em boc no trong `try/except` de neu loi thi terminal van in ro stack trace."

---

## 2. [controller.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/controller.py#L12)

### Vai tro tong quat

`controller.py` la trung tam dieu phoi theo huong MVC.

- `Model` giu trang thai
- `Viewer` ve giao dien va OpenGL
- `Controller` nghe input, xu ly su kien, roi cap nhat nguoc vao model va view

Neu thay hoi "chuong trinh cua em bat dau chay tu dau va ai dieu khien ai", thi file nay la cau tra loi dep nhat.

### 2.1. Ham khoi tao `__init__`

Tham chieu: [controller.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/controller.py#L13)

- Dong 13: khai bao constructor cua `AppController`.

- Dong 14: `self.view = view or Viewer()`
  Neu ben ngoai khong dua `view` vao thi tu tao mot `Viewer` moi.

- Dong 15: `self.model = model or AppModel()`
  Tuong tu, neu khong co `model` thi tu tao `AppModel`.

- Dong 18: `self.view.set_model_reference(self.model)`
  Viewer can biet model hien tai de:
  - lay object dang duoc chon
  - lay scene objects
  - lay camera dang active
  - tuong tac gizmo

- Dong 20: tao `CoordinateSystem`
  Day la he truc toa do/grid de giup nguoi dung dinh huong trong scene.

- Dong 21: `self._setup_coordinate_system()`
  Khoi tao shader, VAO, UManager cho he truc.

- Dong 23 den 26:
  Dang ky callback nguoc tu `Viewer` ve `Controller`.
  Tuc la:
  - scroll thi goi `on_scroll`
  - di chuot thi goi `on_mouse_move`
  - bam chuot thi goi `on_mouse_button`
  - bam phim thi goi `on_key`

Noi de bao cao:
"Controller chinh la noi noi `input` cua nguoi dung voi `state` cua he thong."

### 2.2. `on_scroll`

Tham chieu: [controller.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/controller.py#L28)

- Dong 29: lay kich thuoc cua cua so.
- Dong 30: dua gia tri scroll vao `trackball.zoom(...)`.

Y nghia:
Scroll chuot khong tu sua camera truc tiep, ma di qua `trackball`, nghia la camera duoc dieu khien theo mot abstraction thong nhat.

### 2.3. `on_mouse_move`

Tham chieu: [controller.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/controller.py#L32)

Ham nay xu ly keo chuot trai tuy theo tool dang duoc chon.

- Dong 33: import `imgui` ngay trong ham.
  Lam vay de biet khi nao UI dang "giu chuot".

- Dong 34 den 36:
  Neu `imgui` dang bat su kien chuot thi controller khong duoc tranh quyen.
  Neu khong co doan nay thi keo slider UI cung se lam camera xoay theo.

- Dong 38:
  Chi xu ly logic duoi day khi chuot trai dang duoc giu.

- Dong 39: lay tool hien tai tu model.

#### Nhanh `hand`

- Dong 42 den 45: tinh do lech chuot `dx`, `dy`, roi lay `trackball`.
- Dong 48 den 49:
  Neu `trackball.pan` la mot ham thi goi truc tiep de pan camera.
- Dong 50 den 52:
  Neu version trackball nao do luu `target` thay vi ham `pan`, thi sua truc tiep vao tam nhin.

Noi ngan gon:
"Hand tool la pan camera theo mat phang man hinh."

#### Nhanh `rotate` hoac `select`

- Dong 55 den 56:
  Goi `trackball.drag(...)` de xoay goc nhin camera.

Noi ngan gon:
"Khi dang o select hoac rotate, keo chuot se quay camera quanh tam scene."

#### Nhanh `move`

- Dong 59 den 62:
  Van tinh `dx`, `dy` va lay `trackball`.

- Dong 65 den 70:
  Dich chuyen theo hai truc ngang/doc.

- Dong 73 den 74:
  Sua `distance` de tao cam giac di vao/di ra theo truc sau.

Noi ngan gon:
"Move tool trong camera mode dang la mot dang pan + doi khoang cach camera."

- Dong 77 den 78:
  `scale` chua xu ly o day vi phan nay de cho gizmo.

- Dong 80:
  Luon cap nhat `last_mouse_pos` de frame sau con biet chuot da di bao xa.

### 2.4. `on_key`

Tham chieu: [controller.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/controller.py#L87)

Ham nay gom tat ca phim tat lon cua chuong trinh.

- Dong 89:
  Chi xu ly khi phim duoc nhan hoac giu.

- Dong 90 den 94: phim `W`
  - Neu dang o che do SGD thi doi render mode cua mat loss
  - Neu khong thi doi polygon mode cua OpenGL: fill, line, point

- Dong 95 den 96: phim `Q`
  Tat cua so.

- Dong 97 den 98: phim `S`
  Chuyen qua 4 shader mode.

- Dong 99 den 100: phim `G`
  Bat/tat he truc/grid.

- Dong 102 den 113: phim mui ten
  Dung de pan camera bang ban phim.

- Dong 116 den 130: phim `1`, `2`, `3`
  Bat/tat tung den trong scene.
  Luc bao cao, co the noi:
  "Em cho phep bat tat nhieu nguon sang bang phim tat, moi nguon sang la mot object trong scene."

- Dong 133 den 145: phim `C`
  Duyet qua cac camera co trong scene va luan chuyen goc nhin.

- Dong 148 den 162: phim tat cho SGD
  - `Space`: chay / tam dung
  - `R`: reset
  - `T`: bat/tat trajectory

### 2.5. `_setup_coordinate_system`

Tham chieu: [controller.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/controller.py#L164)

- Dong 166 den 167:
  Chon shader noi suy mau don gian cho truc toa do.

- Dong 169 den 171:
  Tao `VAO`, `Shader`, `UManager`.

- Dong 173:
  Truyen cac tai nguyen vua tao vao `coord_system.setup(...)`.

Noi ngan gon:
"Coordinate system la mot object render rieng, duoc khoi tao mot lan de ve truc va luoi tham chieu."

### 2.6. `_process_ui_actions`

Tham chieu: [controller.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/controller.py#L175)

Day la ham rat quan trong.
UI panel khong sua model truc tiep, ma tra ve mot dictionary `actions`.
Controller doc dictionary nay va quyet dinh sua gi.

Co the nho theo cong thuc:

`UI -> actions -> controller -> model/view`

Nhung nhanh ban nen hoc thuoc:

- Dong 177 den 188:
  Khi category doi, model doi category, grid doi mode, neu la SGD thi khoi tao visualizer.

- Dong 190 den 193:
  Khi doi shape thi nap lai drawable dang preview.

- Dong 195 den 196:
  Khi doi shader thi model doi shader.

- Dong 201 den 204:
  Khi doi cong thuc toan hoc, neu dang preview `MathematicalSurface` thi reload lai ngay.

- Dong 206 den 209:
  Khi doi file model, neu dang preview custom model thi reload lai ngay.

- Dong 214 den 216:
  Doi mau object dang preview.

- Dong 218 den 239:
  Xu ly texture tong quat va texture rieng cho tung object trong scene.

- Dong 241 den 254:
  Tao them `Light` va `Camera` moi trong hierarchy.

Noi voi thay:
"Em dung mot action dispatcher nho trong controller de giao dien va logic tach nhau ra, de panel chi can tra ve y dinh cua nguoi dung, con controller moi la noi quyet dinh xu ly."

---

## 3. [model.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/model.py#L20)

### Vai tro tong quat

`model.py` la noi giu trang thai toan cuc cua ung dung.

Neu `controller` la nguoi dieu phoi, thi `model` la noi luu:

- dang chon category nao
- dang chon shape nao
- dang chon shader nao
- duong dan model / texture la gi
- scene co nhung object nao
- SGD dang dung loss nao, learning rate nao, optimizer nao

### 3.1. Dau file

- Dong 1:
  `from __future__ import annotations`
  Giup annotation cua type linh hoat hon.

- Dong 3 den 5:
  Import cac thu vien can cho viec dynamic import, tinh toan, va khai bao type.

- Dong 8:
  Import cac class `GameObject`.

- Dong 11:
  Dat alias `ShaderPaths` = cap `(vertex_shader, fragment_shader)`.

- Dong 14 den 15:
  `_default_shader_paths()` tra ve bo shader `standard.vert` va `standard.frag`.
  Day la bo shader tong hop ho tro mau, texture, phong, gouraud, rainbow interpolation...

### 3.2. `__init__`

Tham chieu: [model.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/model.py#L22)

Day la noi khoi tao toan bo state.

- Dong 24: `selected_idx = -1`
  Chua chon hinh nao trong danh sach preview.

- Dong 25: `selected_category = 5`
  Gia tri category mac dinh.

- Dong 26: `selected_shader = 0`
  Shader mode dang chon.

- Dong 28 den 29:
  `active_drawable` la object dang preview.
  `drawables` la danh sach drawable dang duoc render.

- Dong 31:
  Cong thuc toan hoc mac dinh cho `MathematicalSurface`.

- Dong 32:
  Duong dan file model cho preview `.obj/.ply`.

- Dong 33:
  Duong dan texture dang duoc chon.

- Dong 34 va 40:
  Mau object.
  O file nay ban co the noi them voi thay la co mot lan gan mau cam roi sau do doi lai ve trang.

- Dong 37:
  `object_type` dung de biet Inspector dang lam viec voi mesh, den hay camera.

- Dong 38:
  Vi tri object dang duoc chon trong hierarchy.

- Dong 42:
  Tool dang active: `select`, `hand`, `move`, `rotate`, `scale`.

- Dong 45 den 72:
  Toan bo state cho phan SGD.
  Day la mot khoi state rieng ben trong model gom:
  - visualizer
  - loss function
  - learning rate
  - momentum
  - batch size
  - max iterations
  - speed
  - trajectory on/off
  - render mode cua surface SGD
  - optimizer nao dang bat
  - vi tri khoi tao cua tung optimizer
  - trang thai run/pause
  - so buoc hien tai

- Dong 75:
  `display_mode = 0` la xem RGB.
  `display_mode = 1` la xem depth map.

- Dong 78:
  Tao `Scene`.

- Dong 81:
  `hierarchy_objects` la ban metadata phuc vu panel hierarchy.

- Dong 84 den 87:
  `mesh_components` la mot bo state mac dinh cho object mesh.

### 3.3. `menu_options`

Tham chieu: [model.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/model.py#L89)

Property nay tra ve danh sach menu phu thuoc vao category.

- Neu dang category 2D thi tra ve 9 hinh 2D.
- Neu dang category 3D thi tra ve cac khoi 3D.
- Neu dang category math thi tra ve `Mathematical Surface`.
- Neu dang category file thi tra ve `Model from .obj/.ply file`.
- Con lai thi la `SGD Visualization`.

Noi ngan gon:
"UI khong hard-code menu o panel, ma hoi model xem category hien tai co nhung option nao."

### 3.4. `shader_names`

Tham chieu: [model.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/model.py#L127)

Property nay tra ten de dua len combo box trong UI:

- `Solid Color`
- `Gouraud`
- `Phong`
- `Rainbow Interpolation`

### 3.5. `_shape_factories`

Tham chieu: [model.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/model.py#L131)

Ham nay la bang anh xa:

- category nao
- se nap class nao
- class do nam trong module nao

Vi du:

- `Cube` -> `geometry.3d.cube3d.Cube`
- `MathematicalSurface` -> `geometry.math_surface3d.MathematicalSurface`
- `ModelLoader` -> `geometry.model_loader3d.ModelLoader`

Noi ngan gon:
"Em dung factory mapping de tu ten UI suy ra lop hinh hoc can tao."

### 3.6. `_shader_paths`

Tham chieu: [model.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/model.py#L168)

Ham nay doi tu `selected_shader` sang duong dan shader:

- `0` -> `color_interp`
- `1` -> `gouraud`
- `2` -> `phong`
- con lai -> `standard`

Y tuong:
preview mode dung nhieu bo shader khac nhau de minh hoa nhanh cac kieu to mau.

### 3.7. `load_active_drawable`

Tham chieu: [model.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/model.py#L177)

Day la mot trong nhung ham quan trong nhat cua model.

No lam 6 buoc:

1. Xoa drawable cu.
2. Kiem tra xem co shape nao dang duoc chon khong.
3. Lay factory phu hop.
4. `importlib.import_module(...)` de nap module dong.
5. Tao object drawable.
6. Goi `setup()` de day du lieu len GPU.

Giai thich theo dong:

- Dong 178 den 179:
  Reset preview hien tai.

- Dong 181 den 185:
  Neu index khong hop le thi dung som.

- Dong 187 den 189:
  Lay `module_name`, `class_name` tu bang factory.

- Dong 191 den 196:
  Nap dong module va lay ra class can dung.

- Dong 198:
  Lay bo shader tu `_shader_paths()`.

- Dong 200 den 217:
  Neu dang la `MathematicalSurface` thi parse chuoi cong thuc toan hoc thanh mot ham `f(x, y)`.
  O day:
  - `safe_dict` gioi han cac ham cho phep
  - `exec(...)` bien chuoi thanh ham Python
  - neu loi thi quay ve ham mac dinh

- Dong 218 den 223:
  Neu la `ModelLoader` thi dua them `filename`.

- Dong 224 den 225:
  Cac shape con lai thi tao thang bang constructor.

- Dong 227:
  `drawable.setup()` la buoc quan trong.
  O buoc nay shape moi tao VAO/VBO/EBO/shader va nap du lieu len GPU.

- Dong 228 den 229:
  Neu dang o shader mode 3 thi set `render_mode = 3`.

- Dong 230 den 231:
  Dua drawable vao danh sach render va dat no lam drawable dang active.

### 3.8. `set_selected`, `set_category`, `set_shader`

Tham chieu:

- [model.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/model.py#L233)
- [model.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/model.py#L239)
- [model.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/model.py#L246)

Ba ham nay rat giong nhau:

- doi state
- roi goi `load_active_drawable()`

Noi de bao cao:
"Moi khi nguoi dung doi hinh, doi category hoac doi shader, em khong sua truc tiep tren drawable cu ma tao lai drawable moi cho sach state."

### 3.9. `_sync_scene_object_visuals`

Tham chieu: [model.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/model.py#L273)

Ham nay dong bo state cua `scene object` xuong `drawable`.

No lam cac viec:

- Dong 274 den 276:
  Neu object khong co drawable thi thoat.

- Dong 278 den 279:
  Dua `shader` tu object xuong `render_mode` cua drawable.

- Dong 281 den 285:
  Neu object co mau thi cap nhat mau cho drawable.
  Rieng mode 3 cua `MathematicalSurface` co the khoi phuc lai mau auto.

- Dong 287 den 288:
  Neu object co `texture_filename` thi load texture do.

- Dong 290 den 293:
  Neu global flat shading dang bat thi ep drawable cung bat theo.

### 3.10. `add_hierarchy_object`

Tham chieu: [model.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/model.py#L295)

Ham nay tao object moi trong scene khi nguoi dung bam Add.

Tu duy cua ham:

1. Xac dinh object loai gi
2. Tao `GameObject` dung class
3. Neu can thi tao `drawable`
4. Dua vao `Scene`
5. Dua vao `hierarchy_objects` de UI ve duoc panel

Ban nen mo ham nay neu thay hoi:

- "Em them object vao scene bang cach nao?"
- "Camera va den co phai cung la object khong?"

Tra loi dep:
"Da, em thong nhat mo hinh scene theo object. Mesh, light, camera deu la object; khac nhau o component va drawable di kem."

---

## 4. [viewer.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/viewer.py#L16)

### Vai tro tong quat

`viewer.py` la noi:

- tao cua so GLFW
- khoi tao OpenGL context
- khoi tao ImGui
- lay camera dang active
- render object, UI, gizmo

Neu `model` tra loi cau hoi "du lieu gi dang co",
thi `viewer` tra loi cau hoi "ve no len man hinh bang cach nao".

### 4.1. `__init__`

Tham chieu: [viewer.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/viewer.py#L17)

- Dong 18:
  `glfw.init()` khoi tao GLFW.

- Dong 19 den 21:
  Cau hinh OpenGL version 3.3 core profile.

- Dong 22:
  Tao cua so.

- Dong 23:
  Gan OpenGL context vao thread hien tai.

- Dong 25:
  Bat depth test de object gan che object xa.

- Dong 26:
  Dat mau nen.

- Dong 28 den 29:
  Tao ImGui context va renderer cho GLFW.

- Dong 31 den 32:
  Tao `default_trackball` va camera index.
  Scene camera tu do se dung trackball nay.

- Dong 35:
  Luu reference den model, ban dau chua co.

- Dong 38:
  Dat style giao dien theo huong "Unity dark".

- Dong 40 den 43:
  Cac callback ben ngoai se duoc controller gan vao day.

- Dong 45:
  `fill_modes` la mot cycle giup doi qua lai fill, line, point.

- Dong 47:
  Tao `TransformGizmo`.

- Dong 49 den 53:
  Nap icon texture cho toolbar.

- Dong 55 den 58:
  Dang ky callback cua GLFW.

### 4.2. `trackball`

Tham chieu: [viewer.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/viewer.py#L64)

Property nay rat hay.
No quyet dinh hien tai app dang nhin qua:

- Scene camera tu do
- hay mot camera object trong scene

Giai thich:

- Dong 67 den 68:
  Neu chua co model thi tra ve camera du phong.

- Dong 70:
  Lay tat ca object nao co `camera_fov`.

- Dong 73 den 74:
  Neu index = 0 thi dung `default_trackball`.

- Dong 77 den 79:
  Neu dang o game camera nao do thi tra ve `trackball` cua camera do.

### 4.3. `on_mouse_move` va `on_mouse_button`

Tham chieu:

- [viewer.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/viewer.py#L109)
- [viewer.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/viewer.py#L146)

Hai ham nay xu ly chuot o phia viewer:

- chuot phai -> xoay camera
- chuot trai -> pan camera neu la hand tool
- chuot trai -> bat dau / keo gizmo neu dang chon move/rotate/scale

Noi ngan gon:
"Viewer lo phan tuong tac truc tiep voi cua so va gizmo, con controller lo phan mapping su kien sang logic tong the."

### 4.4. `load_texture`

Tham chieu: [viewer.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/viewer.py#L172)

Ham nay nap mot file anh thanh texture OpenGL.

- Dong 173:
  Mo file anh va doi sang RGBA.

- Dong 175:
  Lat anh cho dung he toa do texture cua OpenGL.

- Dong 179:
  Xin GPU cap phat `texture_id`.

- Dong 180 den 184:
  Bind texture, upload du lieu pixel, dat filter.

### 4.5. `draw_drawables`

Tham chieu: [viewer.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/viewer.py#L204)

Day la ham render chinh.

No lam cac viec:

1. Lay `view matrix` va `projection matrix`
2. Gom cac light trong scene
3. Lay `display_mode` va `cam_far`
4. Neu dang o SGD thi ve SGD visualizer
5. Neu khong thi ve tung drawable trong scene
6. Neu dang chon 1 object thi ve gizmo

Noi de bao cao:
"Viewer khong giu hinh hoc. Viewer chi lay state tu model/scene roi truyen projection, view, light, texture, shader uniform xuong GPU."

---

## 5. Cach show code voi thay

Neu thay bat mo code, ban nen mo theo thu tu nay:

1. [main.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/main.py#L9)
   Noi ve diem vao va `controller.run()`.

2. [controller.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/controller.py#L13)
   Noi ve MVC, callback, input.

3. [controller.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/controller.py#L87)
   Noi ve phim tat, den, camera, SGD.

4. [model.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/model.py#L22)
   Noi ve state.

5. [model.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/model.py#L177)
   Noi ve dynamic loading shape.

6. [viewer.py](/Users/kandy2705/Documents/Học tập/Đồ hoạ máy tính/BTL_DHMT/viewer.py#L17)
   Noi ve OpenGL window, ImGui, render.

---

## 6. Cach noi ngan gon de de nho

- `main.py`: diem vao
- `controller.py`: nghe input va dieu phoi
- `model.py`: giu state va tao object
- `viewer.py`: mo cua so va ve len man hinh

Cong thuc nho nhanh:

`main -> controller -> model/view -> shader -> GPU`

