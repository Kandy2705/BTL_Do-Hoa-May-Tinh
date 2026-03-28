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
                    actions['shape_changed'] = -1
                
                imgui.separator()
                
                if imgui.begin_menu("2D Shapes"):
                    original_cat = model.selected_category
                    model.selected_category = 0 
                    for idx, name in enumerate(model.menu_options):
                        if imgui.menu_item(name)[0]:
                            actions['category_changed'] = 0
                            actions['shape_changed'] = idx
                    model.selected_category = original_cat
                    imgui.end_menu()
                
                if imgui.begin_menu("3D Shapes"):
                    original_cat = model.selected_category
                    model.selected_category = 1
                    for idx, name in enumerate(model.menu_options):
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
                    for idx, name in enumerate(model.menu_options):
                        if imgui.menu_item(name)[0]:
                            actions['category_changed'] = 4
                            actions['shape_changed'] = idx
                    model.selected_category = original_cat
                    imgui.end_menu()
                imgui.end_menu()

            if imgui.begin_menu("BTL 2"):
                if imgui.menu_item("Setup Road Scene")[0]: actions['category_changed'] = 4
                if imgui.begin_menu("Add Traffic Object"):
                    if imgui.menu_item("Main Vehicle")[0]: pass
                    if imgui.menu_item("Pedestrian")[0]: pass
                    if imgui.menu_item("Traffic Light")[0]: pass
                    imgui.end_menu()
                imgui.end_menu()
            imgui.end_main_menu_bar()
            
        return actions
