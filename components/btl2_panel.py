import imgui


class BTL2Panel:
    """Panel nối giao diện BTL 1 với pipeline synthetic dataset của BTL 2."""

    @staticmethod
    def draw(model):
        actions = {}

        imgui.begin("BTL 2 - Road Scene Generator")
        imgui.text_wrapped(
            "BTL 2 kế thừa nền tảng đồ họa của BTL 1: vẫn là camera phối cảnh, object placement, shader-based rendering, "
            "nhưng mở rộng sang render nhiều pass để xuất RGB, depth, segmentation, YOLO, COCO và metadata."
        )
        imgui.separator()

        source_modes = ["current_scene", "procedural_demo"]
        source_labels = ["Use current BTL1 scene", "Use procedural road demo"]
        current_mode = source_modes.index(model.btl2_source_mode) if model.btl2_source_mode in source_modes else 0
        changed_mode, new_mode = imgui.combo("Source", current_mode, source_labels)
        if changed_mode:
            model.btl2_source_mode = source_modes[new_mode]

        if model.btl2_source_mode == "current_scene":
            imgui.text_wrapped(
                "Che do nay se lay scene dang dat trong BTL 1 lam moi truong goc. "
                "Tat ca camera ban them trong Hierarchy se duoc dung de xuat frame; "
                "camera mac dinh cua viewer (index 0) khong nam trong danh sach nay nen tu dong bi bo qua."
            )
            imgui.text_wrapped(
                "Meo: dat ten object co chua car / pedestrian / sign / light de BTL 2 gan nhan lop hop ly hon."
            )
            if imgui.button("Refresh Scene Summary"):
                actions["btl2_refresh_scene"] = True
            imgui.text(f"Scene cameras: {getattr(model, 'btl2_scene_camera_count', 0)}")
            imgui.text(f"Renderable objects: {getattr(model, 'btl2_scene_renderable_count', 0)}")
        else:
            imgui.text_wrapped("Che do nay dung road scene procedural trong package btl2/ de tao nhanh dataset demo.")

        changed_cfg, new_cfg = imgui.input_text("Config", model.btl2_config_path, 256)
        if changed_cfg:
            model.btl2_config_path = new_cfg

        changed_out, new_out = imgui.input_text("Output", model.btl2_output_dir, 256)
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
        if imgui.button("Generate Dataset"):
            actions["btl2_generate"] = True

        imgui.separator()
        imgui.text("Status:")
        imgui.text_wrapped(model.btl2_last_status)
        imgui.text_wrapped(f"Output folder: {model.btl2_output_dir}")

        if model.btl2_last_result:
            imgui.separator()
            imgui.text(f"Frames: {model.btl2_last_result.get('generated_frames', 0)}")
            imgui.text_wrapped(f"Output: {model.btl2_last_result.get('output_dir', '')}")
            first_frame = model.btl2_last_result.get("first_frame")
            if first_frame:
                imgui.text_wrapped(f"Ví dụ frame đầu: {first_frame.get('rgb', '')}")

        imgui.end()
        return actions
