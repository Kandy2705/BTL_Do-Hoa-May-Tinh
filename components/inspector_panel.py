import imgui
import glfw

class InspectorPanel:
    """Inspector Panel - giữ nguyên code gốc với texture UI"""
    
    @staticmethod
    def draw(model, cube_texture_id):
        """Draw inspector panel - giữ nguyên code gốc với texture UI"""
        actions = {}
        win_w, win_h = glfw.get_window_size(glfw.get_current_context())
        
        imgui.set_next_window_position(win_w - 320, 20)
        imgui.set_next_window_size(320, win_h - 20)
        imgui.begin("Inspector", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE)
        
        selected_obj = model.get_selected_hierarchy_object()
        
        if not selected_obj:
            imgui.text_disabled("No Object Selected")
        else:
            obj_id = selected_obj.get("id", 0)
            obj_type = selected_obj.get("type", "unknown")
            obj_name = selected_obj.get("name", "Unknown")
            obj_data = selected_obj  # Dùng chung dictionary
            
            # --- HEADER ---
            imgui.checkbox("##active", True); imgui.same_line()
            icon = cube_texture_id
            
            # Determine icon based on object type
            if obj_type == "light":
                icon = cube_texture_id  # TODO: Add light icon
            elif obj_type == "camera":
                icon = cube_texture_id  # TODO: Add camera icon
                
            imgui.image(icon, 16, 16)
            imgui.same_line()
            imgui.text(f"{obj_name}")
            imgui.same_line()
            
            # --- COMPONENT: TRANSFORM (Luôn có)
            transform = obj_data.get("transform", {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]})
            if imgui.collapsing_header("Transform", imgui.TREE_NODE_DEFAULT_OPEN):
                imgui.columns(2, "trans_layout", False)
                imgui.set_column_width(0, 80)
                
                # Position
                imgui.text("Position"); imgui.next_column()
                imgui.push_item_width(-1)
                changed_pos, new_pos = imgui.drag_float3(f"##pos_{obj_id}", *transform["position"], 0.1)
                if changed_pos: actions['update_transform_position'] = {"obj_id": obj_id, "value": list(new_pos)}
                imgui.pop_item_width(); imgui.next_column()
                
                # Rotation
                imgui.text("Rotation"); imgui.next_column()
                imgui.push_item_width(-1)
                changed_rot, new_rot = imgui.drag_float3(f"##rot_{obj_id}", *transform["rotation"], 1.0)
                if changed_rot: actions['update_transform_rotation'] = {"obj_id": obj_id, "value": list(new_rot)}
                imgui.pop_item_width(); imgui.next_column()
                
                # Scale
                imgui.text("Scale"); imgui.next_column()
                imgui.push_item_width(-1)
                changed_sca, new_sca = imgui.drag_float3(f"##sca_{obj_id}", *transform["scale"], 0.1)
                if changed_sca: actions['update_transform_scale'] = {"obj_id": obj_id, "value": list(new_sca)}
                imgui.pop_item_width(); imgui.next_column()
                
                imgui.columns(1)

            # --- DYNAMIC COMPONENTS: MESH RENDERER ---
            if obj_type in ["2d", "3d", "math", "custom_model", "mesh"]:
                mesh_renderer = obj_data.get("mesh_renderer", {"shader_idx": 0, "color": [1.0, 0.5, 0.0]})
                if imgui.collapsing_header("Mesh Renderer", imgui.TREE_NODE_DEFAULT_OPEN):
                    imgui.columns(2, "mesh_cols", False)
                    imgui.set_column_width(0, 80)
                    
                    imgui.text("Shader"); imgui.next_column()
                    imgui.push_item_width(-1)
                    current_shader = mesh_renderer.get("shader_idx", 0)
                    changed_shader, new_shader = imgui.combo(f"##shader_{obj_id}", current_shader, model.shader_names)
                    if changed_shader: 
                        actions['update_mesh_shader'] = {"obj_id": obj_id, "value": new_shader}
                    imgui.pop_item_width(); imgui.next_column()
                    
                    imgui.text("Color"); imgui.next_column()
                    imgui.push_item_width(-1)
                    current_color = mesh_renderer.get("color", [1.0, 0.5, 0.0])
                    changed_color, new_color = imgui.color_edit3(f"##color_{obj_id}", *current_color)
                    if changed_color: 
                        actions['update_mesh_color'] = {"obj_id": obj_id, "value": list(new_color)}
                    imgui.pop_item_width(); imgui.next_column()
                    
                    imgui.text("Texture"); imgui.next_column()
                    current_texture = mesh_renderer.get("texture_filename", "")
                    if imgui.button(f"Browse##texture_browse_{obj_id}"):
                        actions['browse_texture_for_object'] = {"obj_id": obj_id}
                    imgui.same_line()
                    if current_texture:
                        if imgui.button(f"Clear##texture_clear_{obj_id}"):
                            actions['clear_texture'] = {"obj_id": obj_id}
                        imgui.same_line()
                        imgui.text(current_texture)
                    imgui.next_column()
                    
                    imgui.columns(1)

            # --- DYNAMIC COMPONENTS: MATH SCRIPT ---
            if obj_type == "math":
                math_data = obj_data.get("math_data", {"function": "(x**2 + y - 11)**2"})
                if imgui.collapsing_header("Math Script", imgui.TREE_NODE_DEFAULT_OPEN):
                    imgui.text("z = f(x, y):")
                    imgui.push_item_width(-1)
                    changed_func, new_func = imgui.input_text(f"##fxy_{obj_id}", math_data.get("function", ""), 256)
                    if changed_func: actions['update_math_function'] = {"obj_id": obj_id, "value": new_func}
                    imgui.pop_item_width()
            
            # --- DYNAMIC COMPONENTS: LIGHT SETTINGS ---
            elif obj_type == "light":
                light_data = obj_data.get("light_data", {"intensity": 1.0, "color": [1.0, 1.0, 1.0]})
                if imgui.collapsing_header("Light Settings", imgui.TREE_NODE_DEFAULT_OPEN):
                    imgui.columns(2, "light_cols", False)
                    imgui.set_column_width(0, 80)
                    
                    imgui.text("Intensity"); imgui.next_column()
                    imgui.push_item_width(-1)
                    changed_intensity, new_intensity = imgui.drag_float(f"##intensity_{obj_id}", light_data.get("intensity", 1.0), 0.1, 0.0, 10.0)
                    if changed_intensity: actions['update_light_intensity'] = {"obj_id": obj_id, "value": new_intensity}
                    imgui.pop_item_width(); imgui.next_column()
                    
                    imgui.text("Color"); imgui.next_column()
                    imgui.push_item_width(-1)
                    current_light_color = light_data.get("color", [1.0, 1.0, 1.0])
                    changed_light_color, new_light_color = imgui.color_edit3(f"##light_color_{obj_id}", *current_light_color)
                    if changed_light_color: actions['update_light_color'] = {"obj_id": obj_id, "value": list(new_light_color)}
                    imgui.pop_item_width(); imgui.next_column()
                    
                    imgui.columns(1)

            # --- DYNAMIC COMPONENTS: CAMERA SETTINGS ---
            elif obj_type == "camera":
                camera_data = obj_data.get("camera_data", {"fov": 60.0, "near": 0.1, "far": 100.0})
                if imgui.collapsing_header("Camera Settings", imgui.TREE_NODE_DEFAULT_OPEN):
                    imgui.columns(2, "cam_cols", False)
                    imgui.set_column_width(0, 80)
                    
                    imgui.text("FOV"); imgui.next_column()
                    imgui.push_item_width(-1)
                    changed_fov, new_fov = imgui.drag_float(f"##fov_{obj_id}", camera_data.get("fov", 60.0), 1.0, 1.0, 179.0)
                    if changed_fov: actions['update_camera_fov'] = {"obj_id": obj_id, "value": new_fov}
                    imgui.pop_item_width(); imgui.next_column()
                    
                    imgui.text("Near"); imgui.next_column()
                    imgui.push_item_width(-1)
                    changed_near, new_near = imgui.drag_float(f"##near_{obj_id}", camera_data.get("near", 0.1), 0.01, 0.01, 10.0)
                    if changed_near: actions['update_camera_near'] = {"obj_id": obj_id, "value": new_near}
                    imgui.pop_item_width(); imgui.next_column()
                    
                    imgui.text("Far"); imgui.next_column()
                    imgui.push_item_width(-1)
                    changed_far, new_far = imgui.drag_float(f"##far_{obj_id}", camera_data.get("far", 100.0), 1.0, 1.0, 1000.0)
                    if changed_far: actions['update_camera_far'] = {"obj_id": obj_id, "value": new_far}
                    imgui.pop_item_width(); imgui.next_column()
                    
                    imgui.columns(1)

        imgui.end()
        return actions
