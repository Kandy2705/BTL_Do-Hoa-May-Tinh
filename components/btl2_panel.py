import os

import glfw
import imgui


class BTL2Panel:
    """BTL2 workflow panel: source -> config -> validate -> generate -> result."""

    @staticmethod
    def _status_meta(status_text):
        text = (status_text or "").strip()
        low = text.lower()
        # Ưu tiên nhận diện trạng thái kết thúc trước để tránh bị dính nhãn RUNNING sai.
        if (
            low.startswith("done")
            or "generated" in low
            or "exported" in low
            or "loaded procedural preview" in low
            or " da xong" in low
            or "hoan tat" in low
        ):
            return "DONE", (0.36, 0.84, 0.55), text
        if "loi" in low or "failed" in low or "validation" in low:
            return "FAILED", (0.95, 0.35, 0.33), text
        if "dang chay" in low or "running" in low or "in progress" in low:
            return "RUNNING", (0.96, 0.77, 0.30), text
        if not text:
            text = "Idle: chua chay BTL2."
        return "IDLE", (0.72, 0.75, 0.80), text

    @staticmethod
    def _validate(model):
        issues = []
        cfg = (model.btl2_config_path or "").strip()
        out_dir = (model.btl2_output_dir or "").strip()
        frames = int(max(0, model.btl2_num_frames))
        seed = int(model.btl2_seed)

        if not cfg:
            issues.append("Config path is empty.")
        elif not os.path.exists(cfg):
            issues.append(f"Config not found: {cfg}")

        if not out_dir:
            issues.append("Output folder is empty.")
        if frames <= 0:
            issues.append("Frames must be > 0.")
        if seed < 0:
            issues.append("Seed must be >= 0.")

        if model.btl2_source_mode == "current_scene":
            if getattr(model, 'btl2_scene_camera_count', 0) <= 0:
                issues.append("Current scene needs at least 1 camera object.")
            if getattr(model, 'btl2_scene_renderable_count', 0) <= 0:
                issues.append("Current scene needs at least 1 renderable object.")

        return issues

    @staticmethod
    def draw(model):
        actions = {}
        win_w, win_h = glfw.get_window_size(glfw.get_current_context())
        win_w = max(win_w, 800)
        win_h = max(win_h, 600)

        panel_w = 380
        imgui.set_next_window_position(win_w - panel_w, 20)
        imgui.set_next_window_size(panel_w, max(win_h - 20, 100))
        imgui.begin("BTL2 Dataset Builder", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE)

        if model.btl2_source_mode == "current_scene":
            model.refresh_btl2_scene_summary()

        status_name, status_color, status_text = BTL2Panel._status_meta(getattr(model, "btl2_last_status", ""))
        imgui.text("Pipeline status:")
        imgui.same_line()
        imgui.push_style_color(imgui.COLOR_TEXT, status_color[0], status_color[1], status_color[2])
        imgui.text(status_name)
        imgui.pop_style_color()
        imgui.text_wrapped(status_text)
        imgui.separator()

        imgui.text("1) Source")
        source_modes = ["current_scene", "procedural_demo"]
        source_labels = ["Use current BTL1 scene", "Use procedural road demo"]
        current_mode = source_modes.index(model.btl2_source_mode) if model.btl2_source_mode in source_modes else 0
        changed_mode, new_mode = imgui.combo("Source", current_mode, source_labels)
        if changed_mode:
            model.btl2_source_mode = source_modes[new_mode]
            if model.btl2_source_mode == "current_scene":
                model.refresh_btl2_scene_summary()
                model.btl2_last_status = "Source set: current BTL1 scene."
            else:
                model.btl2_last_status = (
                    "Source set: procedural demo (offscreen). "
                    "Viewport still shows BTL1 scene."
                )

        if model.btl2_source_mode == "current_scene":
            if imgui.button("Refresh Scene Summary"):
                actions["btl2_refresh_scene"] = True
            imgui.text(f"Cameras: {getattr(model, 'btl2_scene_camera_count', 0)}")
            imgui.text(f"Renderables: {getattr(model, 'btl2_scene_renderable_count', 0)}")
            imgui.text_wrapped("Tip: add camera objects in Hierarchy. Viewer default camera (index 0) is excluded.")
        else:
            imgui.text_wrapped("Procedural mode creates a demo road scene directly from BTL2 package.")
            imgui.text_wrapped("Note: by default it renders offscreen for dataset export.")
            imgui.text_wrapped("Use the button below if you want to load one procedural preview frame into BTL1 scene.")
            if imgui.button("Load Procedural Preview To BTL1 Scene"):
                actions["btl2_load_preview_scene"] = True
        imgui.separator()

        imgui.text("2) Config & Output")
        changed_cfg, new_cfg = imgui.input_text("Config file", model.btl2_config_path, 256)
        if changed_cfg:
            model.btl2_config_path = new_cfg
        changed_out, new_out = imgui.input_text("Output folder", model.btl2_output_dir, 256)
        if changed_out:
            model.btl2_output_dir = new_out
        changed_frames, new_frames = imgui.drag_int("Frames", model.btl2_num_frames, 1, 1, 10000)
        if changed_frames:
            model.btl2_num_frames = max(1, new_frames)
        changed_seed, new_seed = imgui.drag_int("Seed", model.btl2_seed, 1, 0, 1000000)
        if changed_seed:
            model.btl2_seed = max(0, new_seed)

        if imgui.button("Sync From YAML"):
            actions["btl2_sync_config"] = True
        imgui.same_line()

        issues = BTL2Panel._validate(model)
        can_generate = len(issues) == 0
        if not can_generate:
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.30, 0.30, 0.30, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.35, 0.35, 0.35, 1.0)
            clicked_generate = imgui.button("Generate Dataset")
            imgui.pop_style_color()
            imgui.pop_style_color()
            if clicked_generate:
                model.btl2_last_status = "Validation failed: fix items in section 3 before generate."
        else:
            if imgui.button("Generate Dataset"):
                actions["btl2_generate"] = True
        imgui.separator()

        imgui.text("3) Validation")
        if can_generate:
            imgui.push_style_color(imgui.COLOR_TEXT, 0.36, 0.84, 0.55)
            imgui.text("Ready to generate.")
            imgui.pop_style_color()
        else:
            imgui.push_style_color(imgui.COLOR_TEXT, 0.95, 0.55, 0.33)
            imgui.text("Fix these items:")
            imgui.pop_style_color()
            for item in issues:
                imgui.bullet_text(item)
        imgui.separator()

        imgui.text("4) Result")
        result = model.btl2_last_result or {}
        imgui.text(f"Frames generated: {result.get('generated_frames', 0)}")
        imgui.text_wrapped(f"Output: {result.get('output_dir', model.btl2_output_dir)}")
        first_frame = result.get("first_frame")
        if first_frame:
            imgui.text_wrapped(f"Sample RGB: {first_frame.get('rgb', '')}")
            if first_frame.get('depth'):
                imgui.text_wrapped(f"Sample depth: {first_frame.get('depth', '')}")

        imgui.end()
        return actions
