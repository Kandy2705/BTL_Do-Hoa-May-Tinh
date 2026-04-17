import imgui
import glfw
from core.GameObject import Type

class InspectorPanel:
    @staticmethod
    def draw(model, cube_texture_id):
        # Inspector không sửa scene trực tiếp.
        # Nó chỉ dựng UI và trả về một dictionary actions để controller xử lý.
        actions = {}
        win_w, win_h = glfw.get_window_size(glfw.get_current_context())
        
        # Ensure minimum window size
        win_w = max(win_w, 800)
        win_h = max(win_h, 600)
        
        inspector_h = max(win_h - 20, 100)
        if bool(getattr(model, "chemistry_panel_visible", False)) and model.selected_category == 5:
            inspector_h = max(300, min(int(win_h * 0.52), win_h - 285))

        imgui.set_next_window_position(win_w - 320, 20)
        imgui.set_next_window_size(320, inspector_h)
        imgui.begin("Inspector", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE)
        
        # Check if in Normal mode (category 5)
        is_normal_mode = (model.selected_category == 5)
        
        selected_objects = model.scene.selected_objects
        
        if not is_normal_mode:
            # Ở các mode preview (2D, 3D, Math, File, SGD), inspector chỉ hiển thị thông tin cơ bản.
            # Non-Normal mode - show simplified inspector
            imgui.text_colored("View Only Mode", 0.7, 0.7, 0.7)
            imgui.separator()
            
            mode_names = {
                0: "2D Shapes",
                1: "3D Shapes", 
                2: "Mathematical Surface",
                3: "Model from file",
                4: "SGD Visualization",
                6: "BTL 2 - Synthetic Road Scene"
            }
            mode_name = mode_names.get(model.selected_category, "Unknown")
            imgui.text(f"Current Mode: {mode_name}")
            imgui.separator()
            
            # Show shape info if applicable
            if hasattr(model, 'menu_options') and model.selected_category < 5:
                options = model.menu_options()
                if model.selected_idx >= 0 and model.selected_idx < len(options):
                    imgui.text(f"Shape: {options[model.selected_idx]}")
            elif model.selected_category == 6:
                imgui.text_wrapped("Mode này nối BTL 1 và BTL 2: BTL 1 cung cấp nền tảng đồ họa tương tác, còn BTL 2 dùng pipeline render đa pass để xuất dữ liệu huấn luyện.")
            
            imgui.end()
            return actions
        
        # Normal mode - full inspector
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
            
            # Transform là component cơ bản nhất: object nào cũng có vị trí, xoay, scale.
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
            
            # Mesh Renderer chỉ áp dụng cho object có thể vẽ được:
            # mesh thường, mathematical surface, custom model...
            if target.type not in [Type.CAMERA, Type.LIGHT]:
                if imgui.collapsing_header("Mesh Renderer", imgui.TREE_NODE_DEFAULT_OPEN):
                    imgui.columns(2, "mesh_cols", False)
                    imgui.set_column_width(0, 80)
                    
                    imgui.text("Shader"); imgui.next_column()
                    imgui.push_item_width(-1)
                    changed_shader, new_shader = imgui.combo(f"##shader_{target.id}", target.shader, model.shader_names())
                    if changed_shader: 
                        actions['update_attr'] = {"obj": target, "attr": "shader", "val": new_shader}
                        
                        # --- THÊM 2 DÒNG NÀY ĐỂ BÁO CHO SHADER ĐỔI CHẾ ĐỘ ---
                        if hasattr(target, 'drawable') and target.drawable:
                            target.drawable.render_mode = new_shader
                            
                    imgui.pop_item_width(); imgui.next_column()
                    
                    # 1. CHỌN MÀU CƠ BẢN (Yêu cầu B)
                    imgui.text("Color"); imgui.next_column()
                    imgui.push_item_width(-1)
                    changed_color, new_color = imgui.color_edit3(f"##color_{target.id}", target.color[0], target.color[1], target.color[2])
                    if changed_color: 
                        # Cập nhật mảng an toàn không bị lỗi Index
                        updated_color = list(new_color)
                        if len(target.color) > 3: updated_color.append(target.color[3]) 
                        
                        actions['update_attr'] = {"obj": target, "attr": "color", "val": updated_color}
                        # Update luôn xuống drawable nếu có
                        if hasattr(target, 'drawable') and target.drawable:
                            target.drawable.set_color(new_color) 
                    imgui.pop_item_width(); imgui.next_column()
                    
                    imgui.text("Texture"); imgui.next_column()
                    # Truyền nguyên object vào action thay vì chỉ truyền ID như cũ
                    if imgui.button(f"Browse##texture_browse_{target.id}"):
                        actions['browse_texture_for_object'] = {"obj": target}
                    imgui.same_line()
                    
                    if target.texture_filename:
                        if imgui.button(f"Clear##texture_clear_{target.id}"):
                            actions['update_attr'] = {"obj": target, "attr": "texture_filename", "val": ""}
                            if hasattr(target, 'drawable') and target.drawable:
                                target.drawable.set_texture("")
                        imgui.same_line()
                        imgui.text(target.texture_filename)
                    imgui.next_column()

                    imgui.columns(1) 
                    
                    imgui.spacing()
                    imgui.spacing()
                    
                    imgui.columns(2, "lighting_cols", False)  # Start new columns
                    imgui.set_column_width(0, 80)

                    if hasattr(target, 'drawable') and target.drawable:
                        
                        # # 3. CÔNG TẮC ÁNH SÁNG (Yêu cầu C)
                        # imgui.text("Lighting"); imgui.next_column()
                        # current_light = getattr(target.drawable, 'lighting_enabled', True)
                        # changed_light, new_light = imgui.checkbox(f"Enable Phong Shading##light_{target.id}", current_light)
                        # if changed_light:
                        #     actions['update_attr'] = {"obj": target.drawable, "attr": "lighting_enabled", "val": new_light}
                        # imgui.next_column()
                        
                        # Padding giữa các sections
                        imgui.spacing()

                        # 4. CÔNG TẮC FLAT SHADING (Yêu cầu A)
                        imgui.text("Flat Shading"); imgui.next_column()
                        current_flat = getattr(target.drawable, 'use_flat_color', False)
                        changed_flat, new_flat = imgui.checkbox(f"Use Flat Shading##flat_{target.id}", current_flat)
                        if changed_flat:
                            actions['update_attr'] = {"obj": target.drawable, "attr": "use_flat_color", "val": new_flat}
                            if new_flat:
                                target.drawable.set_solid_color(target.color[:3])
                        imgui.next_column()
                        
                        # 5. SHININESS (Độ bóng cho specular)
                        imgui.text("Shininess"); imgui.next_column()
                        imgui.push_item_width(-1)
                        current_shininess = getattr(target.drawable, 'shininess', 100.0)
                        changed_shin, new_shininess = imgui.drag_float(f"##shininess_{target.id}", current_shininess, 1.0, 1.0, 500.0, "%.0f")
                        if changed_shin:
                            actions['update_attr'] = {"obj": target.drawable, "attr": "shininess", "val": new_shininess}
                        imgui.pop_item_width()
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
