import imgui
import glfw


class SGDPanel:
    OPTIMIZER_COLORS = {
        'GD': (1.0, 0.2, 0.2),
        'SGD': (0.2, 1.0, 0.2),
        'MiniBatch': (0.2, 0.2, 1.0),
        'Momentum': (1.0, 1.0, 0.2),
        'Nesterov': (1.0, 0.5, 0.0),
        'Adam': (1.0, 0.2, 1.0),
    }
    
    @staticmethod
    def draw(model):
        actions = {}
        win_w, win_h = glfw.get_window_size(glfw.get_current_context())
        
        imgui.set_next_window_position(win_w - 320, 20)
        imgui.set_next_window_size(320, win_h - 20)
        imgui.begin("SGD Optimizer", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE)
        
        if imgui.collapsing_header("Loss Function", imgui.TREE_NODE_DEFAULT_OPEN):
            loss_functions = ["Himmelblau", "Rosenbrock", "Booth", "Quadratic 2D"]
            loss_current = loss_functions.index(model.sgd_loss_function) if model.sgd_loss_function in loss_functions else 0
            
            changed, new_loss = imgui.combo("##loss_func", loss_current, loss_functions)
            if changed:
                new_loss_name = loss_functions[new_loss]
                model.set_sgd_loss_function(new_loss_name)
                if 'sgd_loss_changed' not in actions:
                    actions['sgd_loss_changed'] = new_loss_name
            
            imgui.text(f"f(x,y) = {model.sgd_loss_function}")
        
        if imgui.collapsing_header("Parameters", imgui.TREE_NODE_DEFAULT_OPEN):
            changed_lr, new_lr = imgui.drag_float("Learning Rate", model.sgd_learning_rate, 0.005, 0.001, 0.2, "%.4f")
            if changed_lr:
                model.sgd_learning_rate = max(0.001, min(0.2, new_lr))
                actions['sgd_param_changed'] = True
            
            changed_mom, new_mom = imgui.drag_float("Momentum", model.sgd_momentum, 0.01, 0.0, 0.99, "%.2f")
            if changed_mom:
                model.sgd_momentum = max(0.0, min(0.99, new_mom))
                actions['sgd_param_changed'] = True
            
            changed_batch, new_batch = imgui.drag_int("Batch Size", model.sgd_batch_size, 1, 1, 1000)
            if changed_batch:
                model.sgd_batch_size = max(1, min(1000, new_batch))
                actions['sgd_param_changed'] = True
            
            changed_speed, new_speed = imgui.drag_int("Speed (steps/frame)", model.sgd_simulation_speed, 1, 1, 200)
            if changed_speed:
                model.sgd_simulation_speed = max(1, min(200, new_speed))
        
        if imgui.collapsing_header("Optimizers", imgui.TREE_NODE_DEFAULT_OPEN):
            imgui.text("Color Legend:")
            imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 0.2, 0.2); imgui.text("  GD - Red"); imgui.pop_style_color()
            imgui.push_style_color(imgui.COLOR_TEXT, 0.2, 1.0, 0.2); imgui.text("  SGD - Green"); imgui.pop_style_color()
            imgui.push_style_color(imgui.COLOR_TEXT, 0.2, 0.2, 1.0); imgui.text("  MiniBatch - Blue"); imgui.pop_style_color()
            imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 1.0, 0.2); imgui.text("  Momentum - Yellow"); imgui.pop_style_color()
            imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 0.5, 0.0); imgui.text("  Nesterov - Orange"); imgui.pop_style_color()
            imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 0.2, 1.0); imgui.text("  Adam - Pink"); imgui.pop_style_color()
            imgui.separator()
            imgui.text("Enable/Disable:")
            
            for opt_name in ['GD', 'SGD', 'MiniBatch', 'Momentum', 'Nesterov', 'Adam']:
                is_enabled = model.sgd_optimizers_enabled.get(opt_name, True)
                color = SGDPanel.OPTIMIZER_COLORS.get(opt_name, (1, 1, 1))
                
                imgui.push_style_color(imgui.COLOR_TEXT, color[0], color[1], color[2])
                changed, new_enabled = imgui.checkbox(f"##{opt_name}", is_enabled)
                imgui.pop_style_color()
                imgui.same_line()
                imgui.text(f" {opt_name}")
                
                if changed and new_enabled != is_enabled:
                    model.toggle_optimizer_enabled(opt_name)
                    actions['sgd_optimizer_toggled'] = opt_name
        
        if imgui.collapsing_header("Initial Positions", imgui.TREE_NODE_DEFAULT_OPEN):
            for opt_name in ['GD', 'SGD', 'MiniBatch', 'Momentum', 'Nesterov', 'Adam']:
                pos = model.sgd_initial_positions.get(opt_name, [0.0, 0.0])
                color = SGDPanel.OPTIMIZER_COLORS.get(opt_name, (1, 1, 1))
                
                imgui.push_style_color(imgui.COLOR_TEXT, color[0], color[1], color[2])
                imgui.text(f" {opt_name}:")
                imgui.same_line()
                imgui.push_item_width(60)
                changed_x, new_x = imgui.drag_float(f"##x_{opt_name}", pos[0], 0.1, -10, 10, "%.1f")
                imgui.pop_item_width()
                imgui.same_line()
                imgui.push_item_width(60)
                changed_y, new_y = imgui.drag_float(f"##y_{opt_name}", pos[1], 0.1, -10, 10, "%.1f")
                imgui.pop_item_width()
                imgui.pop_style_color()
                
                if changed_x:
                    model.sgd_initial_positions[opt_name][0] = new_x
                    actions['sgd_pos_changed'] = True
                if changed_y:
                    model.sgd_initial_positions[opt_name][1] = new_y
                    actions['sgd_pos_changed'] = True
        
        if imgui.collapsing_header("Simulation", imgui.TREE_NODE_DEFAULT_OPEN):
            imgui.text(f"Steps: {model.sgd_step_count}")
            
            if imgui.button("Start/Stop", 140, 30):
                model.sgd_simulation_running = not model.sgd_simulation_running
                actions['sgd_toggle_simulation'] = True
            
            imgui.same_line()
            if imgui.button("Reset", 140, 30):
                model.reset_sgd()
                actions['sgd_reset'] = True
        
        if imgui.collapsing_header("Statistics", imgui.TREE_NODE_DEFAULT_OPEN):
            stats = model.get_sgd_stats()
            
            for opt_name, opt_stats in stats.items():
                color = SGDPanel.OPTIMIZER_COLORS.get(opt_name, (1, 1, 1))
                imgui.push_style_color(imgui.COLOR_TEXT, color[0], color[1], color[2])
                imgui.text(f"=== {opt_name} ===")
                imgui.pop_style_color()
                imgui.text(f"  Loss: {opt_stats['loss']:.6f}")
                imgui.text(f"  |Grad|: {opt_stats['gradient_mag']:.6f}")
                imgui.text(f"  Position: ({opt_stats['position'][0]:.4f}, {opt_stats['position'][1]:.4f})")
                imgui.text(f"  Step: {opt_stats['step']}")
                imgui.spacing()
        
        imgui.separator()
        imgui.text("Display:")
        wireframe_modes = ["Solid", "Wireframe", "Points"]
        current_mode = model.sgd_wireframe_mode
        changed, new_mode = imgui.combo("Render Mode", current_mode, wireframe_modes)
        if changed:
            model.sgd_wireframe_mode = new_mode
        
        imgui.separator()
        imgui.text("Controls:")
        imgui.text("- Right Mouse + Drag: Rotate")
        imgui.text("- Scroll: Zoom")
        imgui.text("- W: Toggle Wireframe")
        imgui.text("- Space: Start/Stop")
        imgui.text("- R: Reset")
        
        imgui.end()
        return actions
