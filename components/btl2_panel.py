import os

import glfw
import imgui


class BTL2Panel:
    """Panel UI điều khiển workflow BTL 2.

    Luồng trên giao diện đi theo thứ tự: chọn nguồn scene -> cấu hình output ->
    validate -> generate dataset -> xem preview -> chạy inference thử.
    Hàm `draw` chỉ tạo UI và trả về `actions`; phần xử lý thật nằm ở model/controller.
    """

    IMAGE_VIEWER_POPUP = "BTL2 Image Preview##btl2_image_viewer"

    @staticmethod
    def _vec2_xy(value, default=(0.0, 0.0)):
        """Đổi nhiều kiểu vec2 của imgui/glfw về tuple `(x, y)` an toàn."""
        if value is None:
            return default
        if hasattr(value, "x") and hasattr(value, "y"):
            return float(value.x), float(value.y)
        try:
            return float(value[0]), float(value[1])
        except (TypeError, IndexError, ValueError):
            return default

    @staticmethod
    def _clamp(value, min_value, max_value):
        """Giới hạn một giá trị trong khoảng min/max."""
        return max(min_value, min(max_value, value))

    @staticmethod
    def _image_viewer_state(model):
        """Lấy hoặc khởi tạo state cho modal xem ảnh preview."""
        state = getattr(model, "btl2_image_viewer_state", None)
        if not isinstance(state, dict):
            state = {
                "request_open": False,
                "title": "",
                "path": "",
                "texture_id": None,
                "width": 1.0,
                "height": 1.0,
                "zoom": 1.0,
                "offset_x": 0.0,
                "offset_y": 0.0,
                "fit_next": True,
            }
            model.btl2_image_viewer_state = state
        return state

    @staticmethod
    def _open_image_viewer(model, title, image_path, preview):
        """Nạp thông tin ảnh vào state và yêu cầu mở popup preview lớn."""
        state = BTL2Panel._image_viewer_state(model)
        state.update(
            {
                "request_open": True,
                "title": title,
                "path": image_path or preview.get("path", ""),
                "texture_id": preview.get("texture_id"),
                "width": max(float(preview.get("width", 1)), 1.0),
                "height": max(float(preview.get("height", 1)), 1.0),
                "zoom": 1.0,
                "offset_x": 0.0,
                "offset_y": 0.0,
                "fit_next": True,
            }
        )

    @staticmethod
    def _draw_clickable_preview(model, preview, title, image_path, widget_id):
        """Vẽ thumbnail preview có thể click để mở modal zoom/pan."""
        if not preview or not preview.get("texture_id"):
            return

        texture_id = preview["texture_id"]
        width = max(float(preview.get("width", 1)), 1.0)
        height = max(float(preview.get("height", 1)), 1.0)
        available_width = max(imgui.get_window_width() - 36.0, 120.0)
        display_width = min(available_width, width)
        display_height = display_width * (height / width)

        imgui.text_disabled("Click image to zoom / inspect.")
        imgui.push_id(widget_id)
        clicked = imgui.image_button(
            texture_id,
            display_width,
            display_height,
            frame_padding=1,
            border_color=(0.18, 0.52, 0.82, 0.70),
        )
        if clicked:
            BTL2Panel._open_image_viewer(model, title, image_path, preview)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Open large preview")
        imgui.pop_id()

    @staticmethod
    def _fit_zoom(width, height, canvas_w, canvas_h):
        """Tính mức zoom vừa khít ảnh trong vùng canvas."""
        fit = min(canvas_w / max(width, 1.0), canvas_h / max(height, 1.0))
        return BTL2Panel._clamp(min(fit, 1.0), 0.02, 10.0)

    @staticmethod
    def _clamp_pan(state, canvas_w, canvas_h, scaled_w, scaled_h):
        """Giới hạn pan để ảnh không bị kéo trôi hoàn toàn khỏi canvas."""
        if scaled_w <= canvas_w:
            state["offset_x"] = 0.0
        else:
            limit_x = (scaled_w - canvas_w) * 0.5
            state["offset_x"] = BTL2Panel._clamp(float(state.get("offset_x", 0.0)), -limit_x, limit_x)

        if scaled_h <= canvas_h:
            state["offset_y"] = 0.0
        else:
            limit_y = (scaled_h - canvas_h) * 0.5
            state["offset_y"] = BTL2Panel._clamp(float(state.get("offset_y", 0.0)), -limit_y, limit_y)

    @staticmethod
    def _zoom_image_viewer(state, new_zoom, canvas_pos=None, canvas_size=None, mouse_pos=None):
        """Zoom ảnh preview, giữ điểm dưới con trỏ ổn định nếu có tọa độ chuột."""
        old_zoom = max(float(state.get("zoom", 1.0)), 0.02)
        new_zoom = BTL2Panel._clamp(float(new_zoom), 0.02, 10.0)
        if abs(new_zoom - old_zoom) < 1e-6:
            return

        if canvas_pos and canvas_size and mouse_pos:
            canvas_x, canvas_y = canvas_pos
            canvas_w, canvas_h = canvas_size
            mouse_x, mouse_y = mouse_pos
            center_x = canvas_x + canvas_w * 0.5
            center_y = canvas_y + canvas_h * 0.5
            rel_x = mouse_x - center_x
            rel_y = mouse_y - center_y
            offset_x = float(state.get("offset_x", 0.0))
            offset_y = float(state.get("offset_y", 0.0))
            scale = new_zoom / old_zoom
            state["offset_x"] = rel_x - ((rel_x - offset_x) * scale)
            state["offset_y"] = rel_y - ((rel_y - offset_y) * scale)

        state["zoom"] = new_zoom

    @staticmethod
    def _draw_image_viewer_modal(model):
        """Vẽ modal xem ảnh lớn với fit, zoom, pan và đóng bằng Escape."""
        state = BTL2Panel._image_viewer_state(model)
        if state.pop("request_open", False):
            imgui.open_popup(BTL2Panel.IMAGE_VIEWER_POPUP)

        io = imgui.get_io()
        display_w, display_h = BTL2Panel._vec2_xy(getattr(io, "display_size", None), (1280.0, 720.0))
        modal_w = BTL2Panel._clamp(display_w * 0.78, 640.0, max(display_w - 40.0, 640.0))
        modal_h = BTL2Panel._clamp(display_h * 0.82, 460.0, max(display_h - 40.0, 460.0))
        imgui.set_next_window_position(display_w * 0.5, display_h * 0.5, pivot_x=0.5, pivot_y=0.5)
        imgui.set_next_window_size(modal_w, modal_h)

        flags = imgui.WINDOW_NO_COLLAPSE
        popup = imgui.begin_popup_modal(BTL2Panel.IMAGE_VIEWER_POPUP, True, flags=flags)
        if not popup.opened:
            return

        texture_id = state.get("texture_id")
        width = max(float(state.get("width", 1.0)), 1.0)
        height = max(float(state.get("height", 1.0)), 1.0)
        zoom = max(float(state.get("zoom", 1.0)), 0.02)

        imgui.text(state.get("title", "Image Preview"))
        image_path = state.get("path", "")
        if image_path:
            imgui.text_wrapped(image_path)

        if imgui.button("Fit"):
            state["fit_next"] = True
        imgui.same_line()
        if imgui.button("100%"):
            state["zoom"] = 1.0
            state["offset_x"] = 0.0
            state["offset_y"] = 0.0
        imgui.same_line()
        if imgui.button("-"):
            BTL2Panel._zoom_image_viewer(state, zoom / 1.25)
        imgui.same_line()
        if imgui.button("+"):
            BTL2Panel._zoom_image_viewer(state, zoom * 1.25)
        imgui.same_line()
        changed_zoom, slider_zoom = imgui.slider_float("Zoom", float(state.get("zoom", 1.0)), 0.02, 10.0, "%.2fx")
        if changed_zoom:
            BTL2Panel._zoom_image_viewer(state, slider_zoom)
        imgui.same_line()
        if imgui.button("Close") or imgui.is_key_pressed(imgui.KEY_ESCAPE):
            imgui.close_current_popup()

        imgui.text_disabled("Drag to pan. Mouse wheel over the image zooms in/out.")
        imgui.separator()

        avail_w, avail_h = BTL2Panel._vec2_xy(imgui.get_content_region_available(), (modal_w - 30.0, modal_h - 130.0))
        canvas_w = max(avail_w, 160.0)
        canvas_h = max(avail_h, 160.0)

        imgui.begin_child(
            "##btl2_image_viewer_canvas",
            canvas_w,
            canvas_h,
            border=True,
            flags=imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SCROLL_WITH_MOUSE,
        )

        canvas_x, canvas_y = BTL2Panel._vec2_xy(imgui.get_cursor_screen_pos())
        imgui.invisible_button("##btl2_image_canvas_hitbox", canvas_w, canvas_h)
        hovered = imgui.is_item_hovered()
        active = imgui.is_item_active()

        fit_zoom = BTL2Panel._fit_zoom(width, height, canvas_w, canvas_h)
        if state.pop("fit_next", False):
            state["zoom"] = fit_zoom
            state["offset_x"] = 0.0
            state["offset_y"] = 0.0

        zoom = max(float(state.get("zoom", fit_zoom)), 0.02)
        if hovered:
            wheel = float(getattr(io, "mouse_wheel", 0.0))
            if abs(wheel) > 1e-6:
                # Zoom bằng wheel quanh vị trí chuột để người dùng inspect bbox/mask dễ hơn.
                mouse_x, mouse_y = BTL2Panel._vec2_xy(imgui.get_mouse_pos())
                BTL2Panel._zoom_image_viewer(
                    state,
                    zoom * (1.12 ** wheel),
                    canvas_pos=(canvas_x, canvas_y),
                    canvas_size=(canvas_w, canvas_h),
                    mouse_pos=(mouse_x, mouse_y),
                )
                zoom = max(float(state.get("zoom", zoom)), 0.02)

        scaled_w = width * zoom
        scaled_h = height * zoom
        if imgui.is_item_clicked(0):
            state["drag_start_x"] = float(state.get("offset_x", 0.0))
            state["drag_start_y"] = float(state.get("offset_y", 0.0))
        if active and imgui.is_mouse_dragging(0):
            drag_x, drag_y = BTL2Panel._vec2_xy(imgui.get_mouse_drag_delta(0))
            state["offset_x"] = float(state.get("drag_start_x", 0.0)) + drag_x
            state["offset_y"] = float(state.get("drag_start_y", 0.0)) + drag_y

        BTL2Panel._clamp_pan(state, canvas_w, canvas_h, scaled_w, scaled_h)
        image_x = canvas_x + (canvas_w - scaled_w) * 0.5 + float(state.get("offset_x", 0.0))
        image_y = canvas_y + (canvas_h - scaled_h) * 0.5 + float(state.get("offset_y", 0.0))

        draw_list = imgui.get_window_draw_list()
        draw_list.add_rect_filled(
            canvas_x,
            canvas_y,
            canvas_x + canvas_w,
            canvas_y + canvas_h,
            imgui.get_color_u32_rgba(0.10, 0.10, 0.10, 1.0),
        )
        if texture_id:
            draw_list.push_clip_rect(canvas_x, canvas_y, canvas_x + canvas_w, canvas_y + canvas_h, True)
            draw_list.add_image(
                texture_id,
                (image_x, image_y),
                (image_x + scaled_w, image_y + scaled_h),
            )
            draw_list.pop_clip_rect()

        imgui.end_child()
        imgui.end_popup()

    @staticmethod
    def _status_meta(status_text):
        text = (status_text or "").strip()
        low = text.lower()
        # Ưu tiên nhận diện trạng thái kết thúc trước để tránh bị dính nhãn RUNNING sai.
        if (
            low.startswith("done")
            or low.startswith("validation ok")
            or "generated" in low
            or "exported" in low
            or "loaded procedural preview" in low
            or " da xong" in low
            or "hoan tat" in low
        ):
            return "DONE", (0.36, 0.84, 0.55), text
        if "loi" in low or "failed" in low or "validation failed" in low:
            return "FAILED", (0.95, 0.35, 0.33), text
        if "dang chay" in low or "running" in low or "in progress" in low:
            return "RUNNING", (0.96, 0.77, 0.30), text
        if not text:
            text = "Idle: chua chay BTL2."
        return "IDLE", (0.72, 0.75, 0.80), text

    @staticmethod
    def _validate(model):
        """Kiểm tra input tối thiểu trước khi cho phép generate dataset."""
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
            # Chế độ current_scene cần ít nhất một camera do người dùng đặt và một
            # object có drawable; camera viewport mặc định không được tính.
            if getattr(model, 'btl2_scene_camera_count', 0) <= 0:
                issues.append("Current scene needs at least 1 camera object.")
            if getattr(model, 'btl2_scene_renderable_count', 0) <= 0:
                issues.append("Current scene needs at least 1 renderable object.")

        return issues

    @staticmethod
    def draw(model, dataset_preview=None, inference_preview=None):
        """Vẽ toàn bộ panel BTL 2 và trả về dict action cho controller xử lý."""
        actions = {}
        win_w, win_h = glfw.get_window_size(glfw.get_current_context())
        win_w = max(win_w, 800)
        win_h = max(win_h, 600)

        panel_w = 420
        imgui.set_next_window_position(win_w - panel_w, 20)
        imgui.set_next_window_size(panel_w, max(win_h - 20, 100))
        imgui.begin("BTL2 Dataset Builder", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE)

        if model.btl2_source_mode == "current_scene":
            # Summary có thể thay đổi khi người dùng thêm/xóa object ở Hierarchy.
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
                # Khi quay lại current scene, cập nhật ngay số camera/renderable để
                # phần validation phản hồi đúng.
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
            # Button vẫn bấm được để báo lý do fail trong status, nhưng không gửi
            # action generate cho controller.
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
        if imgui.button("Validate Output Dataset"):
            actions["btl2_validate_output"] = True
        imgui.same_line()
        imgui.text_disabled("checks RGB/mask/depth/labels/metadata/COCO")

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

        imgui.separator()
        imgui.text("5) Preview")
        sample_frame = getattr(model, "btl2_preview_source_image_path", "")
        if sample_frame:
            imgui.text_wrapped(f"Sample frame: {sample_frame}")

        preview_tabs = [
            ("RGB", "rgb"),
            ("Depth", "depth"),
            ("Mask", "mask"),
            ("GT Boxes", "boxes"),
        ]
        # Preview tabs chỉ đổi mode hiển thị; ảnh thật được controller render/load lại.
        for idx, (label, mode) in enumerate(preview_tabs):
            is_active = getattr(model, "btl2_preview_mode", "rgb") == mode
            if is_active:
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.24, 0.53, 0.78, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.28, 0.58, 0.84, 1.0)
            if imgui.button(label):
                actions["btl2_preview_mode"] = mode
            if is_active:
                imgui.pop_style_color()
                imgui.pop_style_color()
            if idx < len(preview_tabs) - 1:
                imgui.same_line()

        if imgui.button("Refresh Preview"):
            actions["btl2_refresh_preview"] = True

        preview_status = getattr(model, "btl2_preview_status", "")
        if preview_status:
            imgui.text_wrapped(preview_status)
        preview_path = getattr(model, "btl2_preview_path", "")
        if preview_path:
            imgui.text_wrapped(f"Preview file: {preview_path}")

        BTL2Panel._draw_clickable_preview(
            model,
            dataset_preview,
            "Dataset Preview",
            preview_path,
            "dataset_preview",
        )

        imgui.separator()
        imgui.text("6) YOLO Inference")
        inf_status_name, inf_status_color, inf_status_text = BTL2Panel._status_meta(
            getattr(model, "btl2_inference_status", "")
        )
        imgui.text("Inference status:")
        imgui.same_line()
        imgui.push_style_color(imgui.COLOR_TEXT, inf_status_color[0], inf_status_color[1], inf_status_color[2])
        imgui.text(inf_status_name)
        imgui.pop_style_color()
        imgui.text_wrapped(inf_status_text)

        current_backend = getattr(model, "btl2_inference_backend", "local_yolo")
        imgui.text("Backend")
        if current_backend == "local_yolo":
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.20, 0.55, 0.90)
        if imgui.button("Local YOLO"):
            model.btl2_inference_backend = "local_yolo"
            model.btl2_inference_status = "Selected backend: Local YOLO."
        if current_backend == "local_yolo":
            imgui.pop_style_color()
        imgui.same_line()
        if current_backend == "roboflow":
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.20, 0.70, 0.35)
        if imgui.button("Roboflow Workflow"):
            model.btl2_inference_backend = "roboflow"
            model.btl2_inference_status = "Selected backend: Roboflow Workflow."
        if current_backend == "roboflow":
            imgui.pop_style_color()

        if getattr(model, "btl2_inference_backend", "local_yolo") == "roboflow":
            # Roboflow là backend tùy chọn, cần đủ thông tin API/workspace/workflow.
            imgui.text_wrapped("Roboflow uses inference-sdk and sends the selected image to your configured Workflow.")
            changed_url, new_url = imgui.input_text(
                "API URL##roboflow",
                getattr(model, "btl2_roboflow_api_url", "https://detect.roboflow.com"),
                256,
            )
            if changed_url:
                model.btl2_roboflow_api_url = new_url
            changed_key, new_key = imgui.input_text(
                "API Key##roboflow",
                getattr(model, "btl2_roboflow_api_key", ""),
                256,
            )
            if changed_key:
                model.btl2_roboflow_api_key = new_key
            changed_workspace, new_workspace = imgui.input_text(
                "Workspace##roboflow",
                getattr(model, "btl2_roboflow_workspace", ""),
                256,
            )
            if changed_workspace:
                model.btl2_roboflow_workspace = new_workspace
            changed_workflow, new_workflow = imgui.input_text(
                "Workflow ID##roboflow",
                getattr(model, "btl2_roboflow_workflow_id", ""),
                256,
            )
            if changed_workflow:
                model.btl2_roboflow_workflow_id = new_workflow

            changed_image, new_image = imgui.input_text(
                "Image file##roboflow",
                getattr(model, "btl2_inference_image_path", ""),
                512,
            )
            if changed_image:
                model.btl2_inference_image_path = new_image
            if imgui.button("Use Sample Image##roboflow"):
                actions["btl2_pick_sample_image"] = True
            imgui.same_line()
            if imgui.button("Browse Image##roboflow"):
                actions["btl2_browse_image"] = True

            changed_conf, new_conf = imgui.slider_float(
                "Min Confidence##roboflow",
                float(getattr(model, "btl2_inference_conf", 0.25)),
                0.01,
                0.90,
            )
            if changed_conf:
                model.btl2_inference_conf = new_conf

            if imgui.button("Run Roboflow Workflow"):
                actions["btl2_run_roboflow_inference"] = True

            json_path = getattr(model, "btl2_roboflow_last_json_path", "")
            csv_path = getattr(model, "btl2_roboflow_last_csv_path", "")
            if json_path:
                imgui.text_wrapped(f"JSON: {json_path}")
            if csv_path:
                imgui.text_wrapped(f"CSV: {csv_path}")
            summary = getattr(model, "btl2_inference_summary", "")
            if summary:
                imgui.text_wrapped(f"Detections: {summary}")
            preview_path = getattr(model, "btl2_inference_preview_path", "")
            if preview_path:
                imgui.text_wrapped(f"Preview: {preview_path}")
            BTL2Panel._draw_clickable_preview(
                model,
                inference_preview,
                "Roboflow Inference Preview",
                preview_path,
                "roboflow_inference_preview",
            )

            BTL2Panel._draw_image_viewer_modal(model)
            imgui.end()
            return actions

        changed_weight, new_weight = imgui.input_text(
            "Weight file",
            getattr(model, "btl2_detector_weight_path", ""),
            512,
        )
        if changed_weight:
            model.btl2_detector_weight_path = new_weight
            model.btl2_detector_weight_preset = "custom"
        current_preset = str(getattr(model, "btl2_detector_weight_preset", "custom"))
        imgui.text("Quick Select")
        # Các preset chỉ chọn đường dẫn/model; load thật diễn ra khi bấm Load Detector.
        if current_preset == "yolov8s":
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.20, 0.55, 0.90)
        if imgui.button("YOLOv8s"):
            actions["btl2_use_yolov8s_weight"] = True
        if current_preset == "yolov8s":
            imgui.pop_style_color()
        imgui.same_line()
        if current_preset == "yolov8m":
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.25, 0.60, 0.95)
        if imgui.button("YOLOv8m"):
            actions["btl2_use_yolov8m_weight"] = True
        if current_preset == "yolov8m":
            imgui.pop_style_color()
        imgui.same_line()
        if current_preset == "yolov8x":
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.30, 0.65, 1.00)
        if imgui.button("YOLOv8x"):
            actions["btl2_use_yolov8x_weight"] = True
        if current_preset == "yolov8x":
            imgui.pop_style_color()
        imgui.same_line()
        if current_preset == "yolo26s":
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.55, 0.45, 0.95)
        if imgui.button("YOLO26s"):
            actions["btl2_use_yolo26s_weight"] = True
        if current_preset == "yolo26s":
            imgui.pop_style_color()
        imgui.same_line()
        if current_preset == "fine_tuned":
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.20, 0.70, 0.35)
        if imgui.button("Fine-tuned"):
            actions["btl2_use_finetuned_weight"] = True
        if current_preset == "fine_tuned":
            imgui.pop_style_color()
        imgui.same_line()
        imgui.text_disabled(f"Current: {current_preset}")
        if imgui.button("Latest best.pt"):
            actions["btl2_pick_latest_weight"] = True
        imgui.same_line()
        if imgui.button("Browse Weight"):
            actions["btl2_browse_weight"] = True

        changed_image, new_image = imgui.input_text(
            "Image file",
            getattr(model, "btl2_inference_image_path", ""),
            512,
        )
        if changed_image:
            model.btl2_inference_image_path = new_image
        if imgui.button("Use Sample Image"):
            actions["btl2_pick_sample_image"] = True
        imgui.same_line()
        if imgui.button("Browse Image"):
            actions["btl2_browse_image"] = True

        changed_conf, new_conf = imgui.slider_float(
            "Confidence",
            float(getattr(model, "btl2_inference_conf", 0.25)),
            0.05,
            0.90,
        )
        if changed_conf:
            model.btl2_inference_conf = new_conf

        changed_imgsz, new_imgsz = imgui.input_int(
            "Image size",
            int(getattr(model, "btl2_inference_imgsz", 640)),
            32,
            128,
        )
        if changed_imgsz:
            model.btl2_inference_imgsz = max(320, min(1280, int(new_imgsz)))
        imgui.text_disabled(f"Device: {getattr(model, 'btl2_inference_device', 'cpu')}")
        imgui.same_line()
        if imgui.button("Use CPU"):
            model.btl2_inference_device = "cpu"
            model.btl2_inference_status = "Selected device: CPU (safer, slower)."
        imgui.same_line()
        if imgui.button("Use MPS"):
            model.btl2_inference_device = "mps"
            model.btl2_inference_status = "Selected device: MPS (faster, may run out of memory)."

        if imgui.button("Load Detector"):
            actions["btl2_load_detector"] = True
        imgui.same_line()
        if imgui.button("Run Inference"):
            actions["btl2_run_inference"] = True

        loaded_weight = getattr(model, "btl2_detector_loaded_path", "")
        if loaded_weight:
            imgui.text_wrapped(f"Loaded: {loaded_weight}")
        summary = getattr(model, "btl2_inference_summary", "")
        if summary:
            imgui.text_wrapped(f"Detections: {summary}")

        preview_path = getattr(model, "btl2_inference_preview_path", "")
        if preview_path:
            imgui.text_wrapped(f"Preview: {preview_path}")

        BTL2Panel._draw_clickable_preview(
            model,
            inference_preview,
            "YOLO Inference Preview",
            preview_path,
            "local_inference_preview",
        )

        BTL2Panel._draw_image_viewer_modal(model)
        imgui.end()
        return actions
