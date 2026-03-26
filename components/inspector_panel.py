import imgui
import glfw
from core.GameObject import Type

class InspectorPanel:
    @staticmethod
    def draw(model, cube_texture_id):
        actions = {}
        win_w, win_h = glfw.get_window_size(glfw.get_current_context())
        
        imgui.set_next_window_position(win_w - 320, 20)
        imgui.set_next_window_size(320, win_h - 20)
        imgui.begin("Inspector", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE)
        
        selected_objects = model.scene.selected_objects
        
        if len(selected_objects) == 0:
            imgui.text_disabled("No Object Selected")
        elif len(selected_objects) > 1:
            imgui.text_disabled(f"{len(selected_objects)} Objects Selected")
        else:
            target = selected_objects[0]
            
            # --- HEADER ---
            current_visible = getattr(target, 'visible', True)
            changed_visible, new_visible = imgui.checkbox("##active", current_visible)
            if changed_visible: target.visible = new_visible
                
            imgui.same_line()
            imgui.image(cube_texture_id, 16, 16)
            imgui.same_line()
            
            changed_name, new_name = imgui.input_text("##name", target.name, 256)
            if changed_name: target.name = new_name
            
            # --- COMPONENT: TRANSFORM (Luôn có) ---
            if imgui.collapsing_header("Transform", imgui.TREE_NODE_DEFAULT_OPEN):
                imgui.columns(2, "trans_layout", False)
                imgui.set_column_width(0, 80)
                
                imgui.text("Position"); imgui.next_column()
                imgui.push_item_width(-1)
                changed_pos, new_pos = imgui.drag_float3(f"##pos_{target.id}", *target.position, 0.1)
                if changed_pos: actions['update_attr'] = {"obj": target, "attr": "position", "val": list(new_pos)}
                imgui.pop_item_width(); imgui.next_column()
                
                imgui.text("Rotation"); imgui.next_column()
                imgui.push_item_width(-1)
                changed_rot, new_rot = imgui.drag_float3(f"##rot_{target.id}", *target.rotation, 1.0)
                if changed_rot: actions['update_attr'] = {"obj": target, "attr": "rotation", "val": list(new_rot)}
                imgui.pop_item_width(); imgui.next_column()
                
                imgui.text("Scale"); imgui.next_column()
                imgui.push_item_width(-1)
                changed_sca, new_sca = imgui.drag_float3(f"##sca_{target.id}", *target.scale, 0.1)
                if changed_sca: actions['update_attr'] = {"obj": target, "attr": "scale", "val": list(new_sca)}
                imgui.pop_item_width(); imgui.next_column()
                
                imgui.columns(1)

            # =========================================================
            # DYNAMIC COMPONENTS (Phân biệt bằng Enum)
            # =========================================================
            
            # --- MESH RENDERER (Cho OBJ, MATH, DEFAULT) ---
            if target.type not in [Type.CAMERA, Type.LIGHT]:
                if imgui.collapsing_header("Mesh Renderer", imgui.TREE_NODE_DEFAULT_OPEN):
                    imgui.columns(2, "mesh_cols", False)
                    imgui.set_column_width(0, 80)
                    
                    imgui.text("Shader"); imgui.next_column()
                    imgui.push_item_width(-1)
                    changed_shader, new_shader = imgui.combo(f"##shader_{target.id}", target.shader, model.shader_names)
                    if changed_shader: actions['update_attr'] = {"obj": target, "attr": "shader", "val": new_shader}
                    imgui.pop_item_width(); imgui.next_column()
                    
                    imgui.text("Color"); imgui.next_column()
                    imgui.push_item_width(-1)
                    changed_color, new_color = imgui.color_edit3(f"##color_{target.id}", target.color[0], target.color[1], target.color[2])
                    if changed_color: 
                        updated_color = list(new_color) + [target.color[3]] # Giữ nguyên Alpha
                        actions['update_attr'] = {"obj": target, "attr": "color", "val": updated_color}
                    imgui.pop_item_width(); imgui.next_column()
                    
                    imgui.text("Texture"); imgui.next_column()
                    # Truyền nguyên object vào action thay vì chỉ truyền ID như cũ
                    if imgui.button(f"Browse##texture_browse_{target.id}"):
                        actions['browse_texture_for_object'] = {"obj": target}
                    imgui.same_line()
                    
                    if target.texture_filename:
                        if imgui.button(f"Clear##texture_clear_{target.id}"):
                            actions['update_attr'] = {"obj": target, "attr": "texture_filename", "val": ""}
                        imgui.same_line()
                        imgui.text(target.texture_filename)
                    imgui.next_column()
                    
                    imgui.columns(1)
                    
            # --- MATH SCRIPT (Chỉ hiển thị thêm cho MATH) ---
            if target.type == Type.GAMEOBJECTMATH:
                if imgui.collapsing_header("Math Script", imgui.TREE_NODE_DEFAULT_OPEN):
                    imgui.text("z = f(x, y):")
                    imgui.push_item_width(-1)
                    
                    # Dùng biến tạm để người dùng gõ thoải mái
                    changed_func, new_func = imgui.input_text(f"##fxy_{target.id}", target.math_script, 256)
                    if changed_func: 
                        target.math_script = new_func # Lưu tạm vào object
                        
                    imgui.pop_item_width()
                    
                    imgui.spacing()
                    if imgui.button(f"Apply##apply_math_{target.id}"):
                        # GỬI MỘT ACTION RIÊNG BIỆT CHO CONTROLLER
                        actions['apply_math'] = {"obj": target}

            # --- CAMERA SETTINGS ---
            elif target.type == Type.CAMERA:
                if imgui.collapsing_header("Camera Settings", imgui.TREE_NODE_DEFAULT_OPEN):
                    imgui.columns(2, "cam_cols", False)
                    imgui.set_column_width(0, 80)
                    
                    imgui.text("FOV"); imgui.next_column()
                    imgui.push_item_width(-1)
                    changed_fov, new_fov = imgui.drag_float(f"##fov_{target.id}", target.camera_fov, 1.0, 1.0, 179.0)
                    if changed_fov: actions['update_attr'] = {"obj": target, "attr": "camera_fov", "val": new_fov}
                    imgui.pop_item_width(); imgui.next_column()
                    
                    imgui.text("Near"); imgui.next_column()
                    imgui.push_item_width(-1)
                    changed_near, new_near = imgui.drag_float(f"##near_{target.id}", target.camera_near, 0.1, 0.01, 10.0)
                    if changed_near: actions['update_attr'] = {"obj": target, "attr": "camera_near", "val": new_near}
                    imgui.pop_item_width(); imgui.next_column()
                    
                    imgui.text("Far"); imgui.next_column()
                    imgui.push_item_width(-1)
                    changed_far, new_far = imgui.drag_float(f"##far_{target.id}", target.camera_far, 1.0, 1.0, 1000.0)
                    if changed_far: actions['update_attr'] = {"obj": target, "attr": "camera_far", "val": new_far}
                    imgui.pop_item_width(); imgui.next_column()
                    
                    imgui.columns(1)

            # --- LIGHT SETTINGS ---
            elif target.type == Type.LIGHT:
                if imgui.collapsing_header("Light Settings", imgui.TREE_NODE_DEFAULT_OPEN):
                    imgui.columns(2, "light_cols", False)
                    imgui.set_column_width(0, 80)
                    
                    imgui.text("Intensity"); imgui.next_column()
                    imgui.push_item_width(-1)
                    changed_intensity, new_intensity = imgui.drag_float(f"##intensity_{target.id}", target.light_intensity, 0.1, 0.0, 10.0)
                    if changed_intensity: actions['update_attr'] = {"obj": target, "attr": "light_intensity", "val": new_intensity}
                    imgui.pop_item_width(); imgui.next_column()
                    
                    imgui.text("Color"); imgui.next_column()
                    imgui.push_item_width(-1)
                    changed_light_color, new_light_color = imgui.color_edit3(f"##light_color_{target.id}", *target.light_color)
                    if changed_light_color: actions['update_attr'] = {"obj": target, "attr": "light_color", "val": list(new_light_color)}
                    imgui.pop_item_width(); imgui.next_column()
                    
                    imgui.columns(1)

        imgui.end()
        return actions