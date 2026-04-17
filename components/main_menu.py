import imgui
import glfw

class MainMenu:
    """Main Menu Bar - giữ nguyên code gốc"""
    
    @staticmethod
    def draw(model):
        """Draw main menu bar - giữ nguyên code gốc"""
        actions = {}
        
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File"):
                if imgui.menu_item("Import Model")[0]: actions['browse_model_file'] = True
                if imgui.menu_item("Exit")[0]: 
                    # Will be handled by main loop
                    pass
                imgui.end_menu()

            if imgui.begin_menu("BTL 1"):
                if imgui.menu_item("Normal Mode")[0]:
                    actions['category_changed'] = 5
                if imgui.menu_item("Center Scene For Demo")[0]:
                    actions['center_scene_for_demo'] = True
                
                imgui.separator()
                
                if imgui.begin_menu("2D Shapes"):
                    original_cat = model.selected_category
                    model.selected_category = 0 
                    for idx, name in enumerate(model.menu_options()):
                        if imgui.menu_item(name)[0]:
                            actions['category_changed'] = 0
                            actions['shape_changed'] = idx
                    model.selected_category = original_cat
                    imgui.end_menu()
                
                if imgui.begin_menu("3D Shapes"):
                    original_cat = model.selected_category
                    model.selected_category = 1
                    for idx, name in enumerate(model.menu_options()):
                        if imgui.menu_item(name)[0]:
                            actions['category_changed'] = 1
                            actions['shape_changed'] = idx
                    model.selected_category = original_cat
                    imgui.end_menu()
                
                if imgui.begin_menu("Mathematical Surface"):
                    if imgui.menu_item("Z = f(x,y)")[0]:
                        actions['category_changed'] = 2
                        actions['shape_changed'] = 0
                    imgui.end_menu()
                
                if imgui.begin_menu("Model from file"):
                    if imgui.menu_item("Model from .obj/.ply file")[0]:
                        actions['category_changed'] = 3
                        actions['shape_changed'] = 0
                    imgui.end_menu()
                
                if imgui.begin_menu("Optimization (SGD)"):
                    original_cat = model.selected_category
                    model.selected_category = 4
                    for idx, name in enumerate(model.menu_options()):
                        if imgui.menu_item(name)[0]:
                            actions['category_changed'] = 4
                            actions['shape_changed'] = idx
                    model.selected_category = original_cat
                    imgui.end_menu()

                imgui.separator()
                if imgui.menu_item("Atom / Molecule Visualizer")[0]:
                    actions['category_changed'] = 5
                    actions['chemistry_show_panel'] = True
                    actions['chemistry_build_scene'] = "bohr"
                imgui.end_menu()

            if imgui.begin_menu("BTL 2"):
                if imgui.menu_item("Generate Demo Dataset")[0]:
                    actions['category_changed'] = 6
                    actions['btl2_generate'] = True
                imgui.end_menu()

            if imgui.begin_menu("Lab"):
                clicked, _ = imgui.menu_item(
                    "Sphere Z 180° SLERP Loop",
                    None,
                    bool(getattr(model, "lab_slerp_enabled", False)),
                    True,
                )
                if clicked:
                    actions['lab_toggle_slerp'] = not bool(getattr(model, "lab_slerp_enabled", False))

                if imgui.menu_item("Rescan Sphere Targets")[0]:
                    actions['lab_rescan_slerp_targets'] = True
                imgui.end_menu()
            imgui.end_main_menu_bar()
            
        return actions
