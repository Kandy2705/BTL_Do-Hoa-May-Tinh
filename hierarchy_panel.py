import imgui
import glfw

class HierarchyPanel:
    """Hierarchy Panel - giữ nguyên code gốc"""
    
    @staticmethod
    def draw(model):
        """Draw hierarchy panel - giữ nguyên code gốc"""
        actions = {}
        win_w, win_h = glfw.get_window_size(glfw.get_current_context())
        
        imgui.set_next_window_position(0, 20)
        imgui.set_next_window_size(275, win_h - 220)
        imgui.begin("Hierarchy", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE)
        
        # Menu chuột phải cho toàn bộ Hierarchy window
        if imgui.begin_popup_context_window():
            if imgui.begin_menu("Add 2D Object"):
                original_cat = model.selected_category
                model.selected_category = 0 
                for idx, name in enumerate(model.menu_options):
                    if imgui.menu_item(name)[0]:
                        # Add hierarchy object instead of changing category
                        obj_name = f"2D_{name}"
                        model.add_hierarchy_object(obj_name, "2d")
                        model.select_hierarchy_object(len(model.hierarchy_objects) - 1)
                model.selected_category = original_cat
                imgui.end_menu()
            if imgui.begin_menu("Add 3D Object"):
                original_cat = model.selected_category
                model.selected_category = 1
                for idx, name in enumerate(model.menu_options):
                    if imgui.menu_item(name)[0]:
                        # Add hierarchy object instead of changing category
                        obj_name = f"3D_{name}"
                        model.add_hierarchy_object(obj_name, "3d")
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
            if imgui.begin_menu("Add Light"):
                if imgui.menu_item("Light")[0]: 
                    actions['add_light'] = True
                imgui.end_menu()
            if imgui.begin_menu("Add Camera"):
                if imgui.menu_item("Camera")[0]: 
                    actions['add_camera'] = True
                imgui.end_menu()
            imgui.separator()
            imgui.menu_item("Delete")
            imgui.end_popup()
        
        if imgui.tree_node("MainScene", imgui.TREE_NODE_DEFAULT_OPEN):
            for i, obj in enumerate(model.hierarchy_objects):
                # BẮT BUỘC PHẢI TÁCH TUPLE Ở ĐÂY để không bị lỗi dính object cuối cùng
                clicked, state = imgui.selectable(f"{obj['name']}##{obj['id']}", obj.get("selected", False))
                if clicked:
                    actions['select_hierarchy_object'] = i
            imgui.tree_pop()

        imgui.end()
        
        return actions
