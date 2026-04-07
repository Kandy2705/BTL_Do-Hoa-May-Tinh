import glfw
import imgui
import numpy as np


class SGDPanel:
    ORDER = ['GD', 'SGD', 'MiniBatch', 'Momentum', 'Nesterov', 'Adam']
    OPTIMIZER_COLORS = {
        'GD': (0.90, 0.28, 0.24),
        'SGD': (0.18, 0.72, 0.44),
        'MiniBatch': (0.20, 0.46, 0.90),
        'Momentum': (0.95, 0.78, 0.20),
        'Nesterov': (0.95, 0.52, 0.16),
        'Adam': (0.72, 0.24, 0.92),
    }
    COLORBLIND_COLORS = {
        'GD': (0.00, 0.45, 0.70),
        'SGD': (0.80, 0.47, 0.65),
        'MiniBatch': (0.00, 0.62, 0.45),
        'Momentum': (0.95, 0.90, 0.25),
        'Nesterov': (0.90, 0.60, 0.00),
        'Adam': (0.35, 0.35, 0.35),
    }

    @staticmethod
    def _current_optimizer_colors(model):
        if model.sgd_visualizer and getattr(model.sgd_visualizer, 'optimizer_colors', None):
            palette = model.sgd_visualizer.optimizer_colors
            return {k: tuple(palette.get(k, (1.0, 1.0, 1.0))) for k in SGDPanel.ORDER}
        if getattr(model, 'sgd_colorblind_mode', False):
            return SGDPanel.COLORBLIND_COLORS
        return SGDPanel.OPTIMIZER_COLORS

    @staticmethod
    def _mark_custom_preset(model):
        model.sgd_selected_preset = "Custom"

    @staticmethod
    def draw_convergence_overlay(model):
        if model.selected_category != 4:
            return

        win_w, win_h = glfw.get_window_size(glfw.get_current_context())
        win_w = max(win_w, 800)
        win_h = max(win_h, 600)

        # Dock riêng ở cột trái (thay cho Hierarchy khi ở SGD mode), không chồng cửa sổ.
        x = 0
        y = 20
        w = 275
        h = max(win_h - 220, 100)

        imgui.set_next_window_position(x, y)
        imgui.set_next_window_size(w, h)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0.17, 0.17, 0.17, 0.96)
        imgui.begin("Convergence Charts", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE)

        colors = SGDPanel._current_optimizer_colors(model)
        replay_step = model.sgd_replay_step if model.sgd_replay_enabled else None
        if model.sgd_replay_enabled:
            imgui.text(f"Replay step: {int(model.sgd_replay_step)}")

        imgui.text("Visible series:")
        for opt_name in SGDPanel.ORDER:
            visible = model.sgd_chart_visible.get(opt_name, True)
            changed_vis, new_vis = imgui.checkbox(f"##chart_overlay_visible_{opt_name}", visible)
            if changed_vis:
                model.sgd_chart_visible[opt_name] = new_vis
            imgui.same_line()
            c = colors.get(opt_name, (1.0, 1.0, 1.0))
            imgui.push_style_color(imgui.COLOR_TEXT, c[0], c[1], c[2])
            imgui.text(opt_name)
            imgui.pop_style_color()

        chart_data = model.get_sgd_metric_series(replay_step=replay_step, max_points=300)
        shown_any = False
        graph_w = max(imgui.get_window_width() - 36.0, 130.0)
        for opt_name in SGDPanel.ORDER:
            if not model.sgd_chart_visible.get(opt_name, True):
                continue
            series = chart_data.get(opt_name)
            if not series:
                continue

            loss_vals = np.array(series['loss'], dtype=np.float32)
            grad_vals = np.array(series['grad'], dtype=np.float32)
            if len(loss_vals) < 2 or len(grad_vals) < 2:
                continue
            shown_any = True

            c = colors.get(opt_name, (1.0, 1.0, 1.0))
            imgui.push_style_color(imgui.COLOR_TEXT, c[0], c[1], c[2])
            imgui.text(f"{opt_name} ({len(loss_vals)} pts)")
            imgui.pop_style_color()
            imgui.plot_lines(f"Loss##overlay_{opt_name}", loss_vals, overlay_text=f"{loss_vals[-1]:.5f}", graph_size=(graph_w, 62))
            imgui.plot_lines(f"|Grad|##overlay_{opt_name}", grad_vals, overlay_text=f"{grad_vals[-1]:.5f}", graph_size=(graph_w, 52))
            imgui.spacing()

        if not shown_any:
            imgui.text_wrapped("Chay simulation de sinh du lieu chart, hoac bat them optimizer/chart series.")

        imgui.end()
        imgui.pop_style_color()

    @staticmethod
    def draw(model):
        actions = {}
        win_w, win_h = glfw.get_window_size(glfw.get_current_context())

        win_w = max(win_w, 800)
        win_h = max(win_h, 600)

        imgui.set_next_window_position(win_w - 320, 20)
        imgui.set_next_window_size(320, max(win_h - 20, 100))
        imgui.begin("SGD Optimizer", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE)

        colors = SGDPanel._current_optimizer_colors(model)

        if imgui.collapsing_header("Loss Function", imgui.TREE_NODE_DEFAULT_OPEN):
            loss_functions = ["Himmelblau", "Rosenbrock", "Booth", "Quadratic 2D"]
            loss_current = loss_functions.index(model.sgd_loss_function) if model.sgd_loss_function in loss_functions else 0

            changed, new_loss = imgui.combo("##loss_func", loss_current, loss_functions)
            if changed:
                new_loss_name = loss_functions[new_loss]
                model.set_sgd_loss_function(new_loss_name)
                SGDPanel._mark_custom_preset(model)
                if 'sgd_loss_changed' not in actions:
                    actions['sgd_loss_changed'] = new_loss_name

            imgui.text(f"f(x,y) = {model.sgd_loss_function}")

        if imgui.collapsing_header("Parameters", imgui.TREE_NODE_DEFAULT_OPEN):
            preset_names = ["Custom"] + list(model.SGD_PRESETS.keys())
            current_preset = getattr(model, 'sgd_selected_preset', "Custom")
            current_idx = preset_names.index(current_preset) if current_preset in preset_names else 0
            changed_preset, new_preset_idx = imgui.combo("Teaching Preset", current_idx, preset_names)
            if changed_preset:
                selected_preset = preset_names[new_preset_idx]
                if selected_preset == "Custom":
                    model.sgd_selected_preset = "Custom"
                else:
                    model.apply_sgd_preset(selected_preset)
                    actions['sgd_preset_applied'] = selected_preset

            changed_lr, new_lr = imgui.drag_float("Learning Rate", model.sgd_learning_rate, 0.005, 0.001, 0.2, "%.4f")
            if changed_lr:
                model.sgd_learning_rate = max(0.001, min(0.2, new_lr))
                SGDPanel._mark_custom_preset(model)
                actions['sgd_param_changed'] = True

            changed_mom, new_mom = imgui.drag_float("Momentum", model.sgd_momentum, 0.01, 0.0, 0.99, "%.2f")
            if changed_mom:
                model.sgd_momentum = max(0.0, min(0.99, new_mom))
                SGDPanel._mark_custom_preset(model)
                actions['sgd_param_changed'] = True

            changed_batch, new_batch = imgui.drag_int("Batch Size", model.sgd_batch_size, 1, 1, 1000)
            if changed_batch:
                model.sgd_batch_size = max(1, min(1000, new_batch))
                SGDPanel._mark_custom_preset(model)
                actions['sgd_param_changed'] = True

            changed_steps, new_steps = imgui.drag_int("Max Steps", model.sgd_max_iterations, 10, 10, 100000)
            if changed_steps:
                model.sgd_max_iterations = max(10, min(100000, new_steps))
                SGDPanel._mark_custom_preset(model)
                actions['sgd_param_changed'] = True

            changed_speed, new_speed = imgui.drag_int("Speed (steps/frame)", model.sgd_simulation_speed, 1, 1, 200)
            if changed_speed:
                model.sgd_simulation_speed = max(1, min(200, new_speed))
                SGDPanel._mark_custom_preset(model)

        if imgui.collapsing_header("Optimizers", imgui.TREE_NODE_DEFAULT_OPEN):
            imgui.text("Color Legend:")
            for opt_name in SGDPanel.ORDER:
                c = colors.get(opt_name, (1.0, 1.0, 1.0))
                imgui.push_style_color(imgui.COLOR_TEXT, c[0], c[1], c[2])
                imgui.text(f"  {opt_name}")
                imgui.pop_style_color()

            imgui.separator()
            imgui.text("Enable/Disable:")

            for opt_name in SGDPanel.ORDER:
                is_enabled = model.sgd_optimizers_enabled.get(opt_name, True)
                c = colors.get(opt_name, (1.0, 1.0, 1.0))
                imgui.push_style_color(imgui.COLOR_TEXT, c[0], c[1], c[2])
                changed, new_enabled = imgui.checkbox(f"##{opt_name}", is_enabled)
                imgui.pop_style_color()
                imgui.same_line()
                imgui.text(f" {opt_name}")

                if changed and new_enabled != is_enabled:
                    model.toggle_optimizer_enabled(opt_name)
                    actions['sgd_optimizer_toggled'] = opt_name

        if imgui.collapsing_header("Initial Positions", imgui.TREE_NODE_DEFAULT_OPEN):
            for opt_name in SGDPanel.ORDER:
                pos = model.sgd_initial_positions.get(opt_name, [0.0, 0.0])
                c = colors.get(opt_name, (1.0, 1.0, 1.0))

                imgui.push_style_color(imgui.COLOR_TEXT, c[0], c[1], c[2])
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
                    SGDPanel._mark_custom_preset(model)
                    actions['sgd_pos_changed'] = True
                if changed_y:
                    model.sgd_initial_positions[opt_name][1] = new_y
                    SGDPanel._mark_custom_preset(model)
                    actions['sgd_pos_changed'] = True

        if imgui.collapsing_header("Simulation", imgui.TREE_NODE_DEFAULT_OPEN):
            imgui.text(f"Steps: {model.sgd_step_count}")
            imgui.text(f"Max: {model.sgd_max_iterations}")
            status = "Running" if model.sgd_simulation_running else "Paused"
            imgui.text(f"Status: {status}")

            toggle_label = "Pause" if model.sgd_simulation_running else "Start"
            if imgui.button(toggle_label, 140, 30):
                if not model.sgd_simulation_running and model.sgd_replay_enabled:
                    model.sgd_replay_enabled = False
                    model.sgd_replay_step = model.sgd_step_count
                model.sgd_simulation_running = not model.sgd_simulation_running
                actions['sgd_toggle_simulation'] = True

            imgui.same_line()
            if imgui.button("Reset", 140, 30):
                model.reset_sgd()
                actions['sgd_reset'] = True

            changed_replay, new_replay = imgui.checkbox("Replay Timeline", model.sgd_replay_enabled)
            if changed_replay:
                model.sgd_replay_enabled = new_replay
                if new_replay:
                    model.sgd_simulation_running = False
                    model.sgd_replay_step = min(model.sgd_replay_step, model.sgd_step_count)
                else:
                    model.sgd_replay_step = model.sgd_step_count

            timeline_max = max(int(model.sgd_step_count), 0)
            if timeline_max > 0:
                changed_t, new_t = imgui.slider_int("Replay Step", int(model.sgd_replay_step), 0, timeline_max, "%d")
                if changed_t:
                    model.sgd_replay_step = new_t
                if imgui.button("Latest", 100, 22):
                    model.sgd_replay_step = timeline_max
                imgui.same_line()
                imgui.text(f"{int(model.sgd_replay_step)}/{timeline_max}")
            else:
                imgui.text_disabled("Replay Step: chua co du lieu")

            changed_traj, new_traj = imgui.checkbox("Show Trajectory", model.sgd_show_trajectory)
            if changed_traj:
                model.sgd_show_trajectory = new_traj

            changed_proj, new_proj = imgui.checkbox("Show Projected Path", model.sgd_show_projected_trajectory)
            if changed_proj:
                model.sgd_show_projected_trajectory = new_proj

            changed_drop, new_drop = imgui.checkbox("Show Vertical Drop Lines", model.sgd_show_drop_lines)
            if changed_drop:
                model.sgd_show_drop_lines = new_drop

            changed_contours, new_contours = imgui.checkbox("Show Contours", model.sgd_show_contours)
            if changed_contours:
                model.sgd_show_contours = new_contours

            changed_trail_w, new_trail_w = imgui.drag_float("Trail Width", model.sgd_trail_width, 0.02, 0.3, 3.0, "%.2f")
            if changed_trail_w:
                model.sgd_trail_width = max(0.3, min(3.0, new_trail_w))
                SGDPanel._mark_custom_preset(model)
                if model.sgd_visualizer:
                    model.sgd_visualizer.trail_width_scale = model.sgd_trail_width

        if imgui.collapsing_header("Visualization", imgui.TREE_NODE_DEFAULT_OPEN):
            view_modes = ["Surface 3D", "Contour Map", "Combined"]
            internal_modes = ["surface", "contour", "combined", "interactive"]
            current_view = internal_modes.index(model.sgd_view_mode) if model.sgd_view_mode in internal_modes else 2
            changed_view, new_view = imgui.combo("View Mode", current_view, view_modes)
            if changed_view:
                model.sgd_view_mode = internal_modes[new_view]

            changed_hover, new_hover = imgui.checkbox("Hover readout x,y,z", getattr(model, 'sgd_hover_enabled', True))
            if changed_hover:
                model.sgd_hover_enabled = new_hover

            changed_cb, new_cb = imgui.checkbox("Colorblind-safe palette", getattr(model, 'sgd_colorblind_mode', False))
            if changed_cb:
                model.set_sgd_colorblind_mode(new_cb)
                colors = SGDPanel._current_optimizer_colors(model)

            imgui.text_colored("Loss Color Map", 0.80, 0.85, 0.95)
            imgui.text_wrapped("Low loss = xanh, high loss = do. Contour mode cho thay duong muc va duong di tung optimizer de phan tich hoi tu theo huong ung dung hon.")

            hover = getattr(model, 'sgd_hover_info', None)
            if hover:
                imgui.separator()
                imgui.text(f"Hover x: {hover['x']:.3f}")
                imgui.text(f"Hover y: {hover['y']:.3f}")
                imgui.text(f"Hover z: {hover['z']:.5f}")

        if imgui.collapsing_header("Statistics", imgui.TREE_NODE_DEFAULT_OPEN):
            replay_step = model.sgd_replay_step if model.sgd_replay_enabled else None
            stats = model.get_sgd_stats(replay_step=replay_step)

            for opt_name in SGDPanel.ORDER:
                if opt_name not in stats:
                    continue
                opt_stats = stats[opt_name]
                c = colors.get(opt_name, (1, 1, 1))
                imgui.push_style_color(imgui.COLOR_TEXT, c[0], c[1], c[2])
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
        imgui.text("- Combined view: nhin ca mat loss va contour")

        imgui.end()
        return actions
