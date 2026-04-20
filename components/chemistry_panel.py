import imgui
import glfw


class ChemistryPanel:
    """Small BTL1 Part 3 control panel for atom and molecule visualization."""

    @staticmethod
    def draw(model):
        actions = {}
        if not getattr(model, "chemistry_panel_visible", False):
            return actions

        win_w, win_h = glfw.get_window_size(glfw.get_current_context())
        win_w = max(win_w, 800)
        win_h = max(win_h, 600)
        panel_w = 320
        inspector_h = max(300, min(int(win_h * 0.52), win_h - 285))
        panel_y = 24 + inspector_h
        panel_h = max(win_h - panel_y - 12, 250)
        imgui.set_next_window_position(win_w - panel_w, panel_y)
        imgui.set_next_window_size(panel_w, panel_h)
        imgui.begin(
            "BTL1 Part 3 - Atom/Molecule",
            flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE,
        )


        current = getattr(model, "chemistry_scene_kind", "none")
        imgui.text(f"Current scene: {current}")

        if imgui.button("Bohr", 82, 26):
            actions["chemistry_build_scene"] = "bohr"
        imgui.same_line()
        if imgui.button("H2O", 82, 26):
            actions["chemistry_build_scene"] = "h2o"
        imgui.same_line()
        if imgui.button("CO2", 82, 26):
            actions["chemistry_build_scene"] = "co2"

        imgui.spacing()
        changed_anim, animate = imgui.checkbox("Animate electrons / molecule", bool(getattr(model, "chemistry_animate", True)))
        if changed_anim:
            actions["chemistry_set_animate"] = animate

        changed_orbit, show_orbits = imgui.checkbox("Show electron orbits", bool(getattr(model, "chemistry_show_orbits", True)))
        if changed_orbit:
            actions["chemistry_set_show_orbits"] = show_orbits

        imgui.text("Animation speed")
        changed_speed, speed = imgui.slider_float(
            "##chemistry_animation_speed",
            float(getattr(model, "chemistry_animation_speed", 1.0)),
            0.1,
            5.0,
            "%.2f",
        )
        if changed_speed:
            actions["chemistry_set_speed"] = speed

        imgui.spacing()
        if imgui.button("Reset Scene", 140, 26):
            actions["chemistry_clear_scene"] = True
        imgui.same_line()
        if imgui.button("Center", 100, 26):
            actions["center_scene_for_demo"] = True

        imgui.separator()

        if imgui.button("Close", 90, 24):
            actions["chemistry_hide_panel"] = True

        imgui.end()
        return actions
