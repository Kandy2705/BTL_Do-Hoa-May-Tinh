import imgui
import glfw

class HierarchyPanel:
    """Hierarchy Panel - giữ nguyên code gốc"""
    
    @staticmethod
    def draw(model):
        """Draw hierarchy panel - giữ nguyên code gốc"""
        actions = {}
        win_w, win_h = glfw.get_window_size(glfw.get_current_context())
        
        # Ensure minimum window size
        win_w = max(win_w, 800)
        win_h = max(win_h, 600)
        
        imgui.set_next_window_position(0, 20)
        panel_h = max(win_h - 20, 100) if model.selected_category == 6 else max(win_h - 220, 100)
        imgui.set_next_window_size(275, panel_h)
        imgui.begin("Hierarchy", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE)
        
        # Check if in Normal mode (category 5)
        is_normal_mode = (model.selected_category == 5)
        
        if is_normal_mode:
            # Full hierarchy panel with add functionality
            if imgui.begin_popup_context_window():
                if imgui.begin_menu("Add 2D Object"):
                    original_cat = model.selected_category
                    model.selected_category = 0 
                    for idx, name in enumerate(model.menu_options()):
                        if imgui.menu_item(name)[0]:
                            obj_name = f"2D_{name}"
                            model.add_hierarchy_object(obj_name, "2d", shape_name=name)
                            model.select_hierarchy_object(len(model.hierarchy_objects) - 1)
                    model.selected_category = original_cat
                    imgui.end_menu()
                if imgui.begin_menu("Add 3D Object"):
                    original_cat = model.selected_category
                    model.selected_category = 1
                    for idx, name in enumerate(model.menu_options()):
                        if imgui.menu_item(name)[0]:
                            obj_name = f"3D_{name}"
                            model.add_hierarchy_object(obj_name, "3d", shape_name=name)
                            model.select_hierarchy_object(len(model.hierarchy_objects) - 1)
                    model.selected_category = original_cat
                    imgui.end_menu()
                if imgui.begin_menu("Add Mathematical Surface"):
                    if imgui.menu_item("Z = f(x,y)")[0]:
                        obj_name = "Math Surface"
                        model.add_hierarchy_object(obj_name, "math")
                        model.select_hierarchy_object(len(model.hierarchy_objects) - 1)
                    imgui.end_menu()
                if imgui.begin_menu("Add Model from file"):
                    if imgui.menu_item("Model from .obj/.ply file")[0]:
                        obj_name = "Custom Model"
                        model.add_hierarchy_object(obj_name, "custom_model")
                        model.select_hierarchy_object(len(model.hierarchy_objects) - 1)
                    imgui.end_menu()
                if imgui.begin_menu("Add Default Model"):
                    for key, spec in model.default_model_options().items():
                        if imgui.menu_item(spec["label"])[0]:
                            actions["add_default_model"] = key
                    imgui.end_menu()
                if imgui.begin_menu("Add Light"):
                    if imgui.menu_item("Light")[0]: 
                        actions['add_light'] = True
                    imgui.end_menu()
                if imgui.begin_menu("Add Camera"):
                    if imgui.menu_item("Camera")[0]: 
                        actions['add_camera'] = True
                    imgui.end_menu()
                imgui.end_popup()
        
        if imgui.tree_node("MainScene", imgui.TREE_NODE_DEFAULT_OPEN):
            for i, obj in enumerate(model.scene.objects):
                is_selected = obj in model.scene.selected_objects

                clicked, state = imgui.selectable(f"{obj.name}##{obj.id}", is_selected)
                if clicked:
                    io = imgui.get_io()
                    is_multi_select = io.key_super or io.key_ctrl 
                    
                    actions['select_object'] = {
                        'object': obj,
                        'multi_select': is_multi_select
                    }
                
                # Right-click context menu for each object
                if imgui.begin_popup_context_item(f"obj_menu_{obj.id}"):
                    if imgui.menu_item("Delete")[0]:
                        actions['delete_object'] = obj
                    imgui.end_popup()
            
            imgui.tree_pop()
            
        if imgui.is_window_hovered() and imgui.is_mouse_clicked(0) and not imgui.is_any_item_hovered():
            actions['clear_selection'] = True

        if not is_normal_mode:
            # Non-Normal mode - view only, show info about current mode
            imgui.separator()
            imgui.text_colored("View Only Mode", 0.7, 0.7, 0.7)
            
            mode_names = {
                0: "2D Shapes",
                1: "3D Shapes", 
                2: "Mathematical Surface",
                3: "Model from file",
                4: "SGD Visualization",
                6: "BTL 2 - Synthetic Road Scene"
            }
            mode_name = mode_names.get(model.selected_category, "Unknown")
            imgui.text(f"Mode: {mode_name}")
            
            if model.selected_category == 4 and model.sgd_visualizer:
                imgui.text("SGD Optimizers:")
                for name in model.sgd_visualizer.optimizers.keys():
                    imgui.text(f"  - {name}")
            elif model.selected_category == 6:
                imgui.text_wrapped("BTL2 mode: left panel stays as scene tree. Use camera objects + renderable objects, then run dataset export in right panel.")

        imgui.end()
        
        return actions
