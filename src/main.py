import taichi as ti
import numpy as np
import dearpygui.dearpygui as dpg
from src.config import *
from src.config import DEFAULT_NUM_PARTICLES, MIN_NUM_PARTICLES, MAX_NUM_PARTICLES
from src.solver import FluidSolver

# Initialize Taichi
try:
    ti.init(arch=ti.gpu)
except:
    ti.init(arch=ti.cpu)

# Global State
class AppState:
    def __init__(self):
        self.tank_width = 1000.0
        self.tank_height = 500.0
        self.tank_depth = 1000.0
        
        self.inlet_y_mm = 100.0
        self.inlet_z_mm = 500.0
        self.inlet_radius_mm = 60.0
        
        self.outlet_y_mm = 400.0
        self.outlet_z_mm = 500.0
        self.outlet_radius_mm = 60.0
        
        self.inlet_flow = 500.0
        self.outlet_flow = 500.0
        self.is_sync = True
        
        self.current_num_particles = DEFAULT_NUM_PARTICLES
        self.target_num_particles = DEFAULT_NUM_PARTICLES
        
        self.needs_dimension_update = False
        self.needs_particle_reset = False
        self.needs_particle_count_update = False

state = AppState()

# Resolution Scale
SCALE = 10.0 

# Scene Data
box_vertices = ti.Vector.field(3, dtype=float, shape=8)
box_indices = ti.field(dtype=int, shape=24)

# Pipe Data
MAX_PIPE_VERTS = 1000
pipe_v_field = ti.Vector.field(3, dtype=float, shape=MAX_PIPE_VERTS)
pipe_i_field = ti.field(dtype=int, shape=MAX_PIPE_VERTS * 2)
num_pipe_indices = ti.field(dtype=int, shape=())

def update_box_geometry(res_x, res_y, res_z):
    corners = np.array([
        [0, 0, 0], [res_x, 0, 0], [res_x, 0, res_z], [0, 0, res_z],
        [0, res_y, 0], [res_x, res_y, 0], [res_x, res_y, res_z], [0, res_y, res_z]
    ], dtype=np.float32)
    
    indices = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ], dtype=np.int32)
    
    for i in range(8):
        box_vertices[i] = ti.Vector(corners[i])
    for i in range(12):
        box_indices[2*i] = indices[i, 0]
        box_indices[2*i+1] = indices[i, 1]

def update_pipe_geometry(in_y, out_y, in_rad, out_rad, in_z, out_z, res_y, res_z):
    verts = []
    inds = []
    num_pipe_segs = 16
    
    def add_pipe_mesh(start_pos, end_pos, radius):
        start_idx = len(verts)
        for i in range(num_pipe_segs):
            theta = (i / num_pipe_segs) * 2 * np.pi
            y = np.cos(theta) * radius
            z = np.sin(theta) * radius
            verts.append([start_pos[0], start_pos[1] + y, start_pos[2] + z])
            verts.append([end_pos[0], end_pos[1] + y, end_pos[2] + z])
            
            curr = start_idx + 2*i
            next_seg = start_idx + 2*((i+1)%num_pipe_segs)
            
            inds.append([curr, next_seg])
            inds.append([curr+1, next_seg+1])
            inds.append([curr, curr+1])

    add_pipe_mesh([0, in_y, in_z], [-20, in_y, in_z], in_rad)
    add_pipe_mesh([0, out_y, out_z], [-20, out_y, out_z], out_rad)
    
    count = min(len(verts), MAX_PIPE_VERTS)
    for i in range(count):
        pipe_v_field[i] = ti.Vector(verts[i])
        
    ind_count = min(len(inds), MAX_PIPE_VERTS)
    for i in range(ind_count):
        pipe_i_field[2*i] = inds[i][0]
        pipe_i_field[2*i+1] = inds[i][1]
        
    num_pipe_indices[None] = ind_count * 2


# ============ Dear PyGui Callbacks ============

def sync_slider_to_input(sender, app_data, user_data):
    """スライダー値をインプットボックスに同期"""
    input_tag = user_data
    dpg.set_value(input_tag, app_data)
    update_state_from_ui()

def sync_input_to_slider(sender, app_data, user_data):
    """インプットボックス値をスライダーに同期"""
    slider_tag = user_data
    dpg.set_value(slider_tag, app_data)
    update_state_from_ui()

def update_state_from_ui():
    """UIから状態を更新"""
    state.tank_width = dpg.get_value("tank_width_slider")
    state.tank_height = dpg.get_value("tank_height_slider")
    state.tank_depth = dpg.get_value("tank_depth_slider")
    
    state.inlet_y_mm = dpg.get_value("inlet_y_slider")
    state.inlet_z_mm = dpg.get_value("inlet_z_slider")
    state.inlet_radius_mm = dpg.get_value("inlet_radius_slider")
    state.inlet_flow = dpg.get_value("inlet_flow_slider")
    
    state.outlet_y_mm = dpg.get_value("outlet_y_slider")
    state.outlet_z_mm = dpg.get_value("outlet_z_slider")
    state.outlet_radius_mm = dpg.get_value("outlet_radius_slider")
    
    state.is_sync = dpg.get_value("sync_checkbox")
    if state.is_sync:
        state.outlet_flow = state.inlet_flow
        dpg.set_value("outlet_flow_slider", state.outlet_flow)
        dpg.set_value("outlet_flow_input", state.outlet_flow)
    else:
        state.outlet_flow = dpg.get_value("outlet_flow_slider")
    
    state.target_num_particles = int(dpg.get_value("particle_slider"))

def on_apply_dimensions():
    state.needs_dimension_update = True

def on_reset_particles():
    state.needs_particle_reset = True

def on_apply_particles():
    state.needs_particle_count_update = True

def create_labeled_slider_with_input(label, tag_base, default_val, min_val, max_val, format_str="%.1f"):
    """ラベル + スライダー + インプットボックスのセットを作成"""
    slider_tag = f"{tag_base}_slider"
    input_tag = f"{tag_base}_input"
    
    with dpg.group(horizontal=True):
        dpg.add_text(label, indent=10)
        dpg.add_spacer(width=10)
        dpg.add_slider_float(
            tag=slider_tag,
            default_value=default_val,
            min_value=min_val,
            max_value=max_val,
            width=180,
            format=format_str,
            callback=sync_slider_to_input,
            user_data=input_tag
        )
        dpg.add_input_float(
            tag=input_tag,
            default_value=default_val,
            width=70,
            step=0,
            callback=sync_input_to_slider,
            user_data=slider_tag
        )

def setup_dpg_ui():
    dpg.create_context()
    
    # フォント設定（日本語対応）
    with dpg.font_registry():
        font_path = "C:/Windows/Fonts/meiryo.ttc"
        try:
            with dpg.font(font_path, 16) as default_font:
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Japanese)
            dpg.bind_font(default_font)
        except Exception as e:
            print(f"Font loading failed: {e}")
    
    # テーマ設定（Windows風ダークテーマ）
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (45, 45, 48, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (30, 30, 30, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (60, 60, 65, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (62, 62, 66, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (75, 75, 80, 255))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (0, 122, 204, 255))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (0, 150, 230, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Button, (62, 62, 66, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (80, 80, 85, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (0, 122, 204, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Header, (60, 60, 65, 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (80, 80, 85, 255))
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (0, 180, 100, 255))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 4)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 6)
    
    dpg.bind_theme(global_theme)
    
    # メインウィンドウ
    with dpg.window(label="コントロールパネル", tag="main_window", width=400, height=680, no_close=True):
        
        # タンク寸法セクション
        with dpg.collapsing_header(label="タンク寸法", default_open=True):
            dpg.add_spacer(height=5)
            create_labeled_slider_with_input("幅 (mm)", "tank_width", state.tank_width, 500.0, 2000.0)
            create_labeled_slider_with_input("高さ (mm)", "tank_height", state.tank_height, 200.0, 1000.0)
            create_labeled_slider_with_input("奥行 (mm)", "tank_depth", state.tank_depth, 500.0, 2000.0)
            dpg.add_spacer(height=5)
            dpg.add_button(label="寸法を適用", callback=on_apply_dimensions, width=150)
            dpg.add_spacer(height=10)
        
        dpg.add_separator()
        
        # 流入口セクション
        with dpg.collapsing_header(label="流入口 (Inlet)", default_open=True):
            dpg.add_spacer(height=5)
            create_labeled_slider_with_input("Y位置 (mm)", "inlet_y", state.inlet_y_mm, 20.0, 450.0)
            create_labeled_slider_with_input("Z位置 (mm)", "inlet_z", state.inlet_z_mm, 20.0, 980.0)
            create_labeled_slider_with_input("半径 (mm)", "inlet_radius", state.inlet_radius_mm, 20.0, 150.0)
            create_labeled_slider_with_input("流量 (L/min)", "inlet_flow", state.inlet_flow, 0.0, 2000.0)
            dpg.add_spacer(height=10)
        
        dpg.add_separator()
        
        # 流出口セクション
        with dpg.collapsing_header(label="流出口 (Outlet)", default_open=True):
            dpg.add_spacer(height=5)
            dpg.add_checkbox(label="流入口と同期", tag="sync_checkbox", default_value=state.is_sync, callback=lambda: update_state_from_ui())
            dpg.add_spacer(height=5)
            create_labeled_slider_with_input("Y位置 (mm)", "outlet_y", state.outlet_y_mm, 20.0, 450.0)
            create_labeled_slider_with_input("Z位置 (mm)", "outlet_z", state.outlet_z_mm, 20.0, 980.0)
            create_labeled_slider_with_input("半径 (mm)", "outlet_radius", state.outlet_radius_mm, 20.0, 150.0)
            create_labeled_slider_with_input("流量 (L/min)", "outlet_flow", state.outlet_flow, 0.0, 2000.0)
            dpg.add_spacer(height=10)
        
        dpg.add_separator()
        
        # 粒子セクション
        with dpg.collapsing_header(label="粒子設定", default_open=True):
            dpg.add_spacer(height=5)
            with dpg.group(horizontal=True):
                dpg.add_text("粒子数", indent=10)
                dpg.add_spacer(width=10)
                dpg.add_slider_int(
                    tag="particle_slider",
                    default_value=state.current_num_particles,
                    min_value=MIN_NUM_PARTICLES,
                    max_value=MAX_NUM_PARTICLES,
                    width=180,
                    callback=lambda: update_state_from_ui()
                )
                dpg.add_input_int(
                    tag="particle_input",
                    default_value=state.current_num_particles,
                    width=70,
                    step=0,
                    callback=lambda s, a: dpg.set_value("particle_slider", a)
                )
            
            dpg.add_text(f"現在: {state.current_num_particles:,}", tag="particle_count_text", indent=10)
            dpg.add_spacer(height=5)
            with dpg.group(horizontal=True):
                dpg.add_button(label="粒子数を適用", callback=on_apply_particles, width=120)
                dpg.add_spacer(width=10)
                dpg.add_button(label="リセット", callback=on_reset_particles, width=80)
            dpg.add_spacer(height=10)
        
        dpg.add_separator()
        dpg.add_spacer(height=10)
        
        # 操作説明
        with dpg.collapsing_header(label="操作方法", default_open=False):
            dpg.add_text("右クリック+ドラッグ: 視点回転", indent=10)
            dpg.add_text("Shift+右クリック: ズーム", indent=10)
            dpg.add_text("中クリック+ドラッグ: パン", indent=10)
    
    dpg.create_viewport(title='Fluid Simulation - Control Panel', width=420, height=700, x_pos=50, y_pos=50)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)


def main():
    # DearPyGui UIをセットアップ
    setup_dpg_ui()
    
    # Physics Init
    res_x = int(state.tank_width / SCALE)
    res_y = int(state.tank_height / SCALE)
    res_z = int(state.tank_depth / SCALE)
    
    solver = FluidSolver(res_x, res_y, res_z, state.current_num_particles)
    update_box_geometry(res_x, res_y, res_z)
    update_pipe_geometry(
        state.inlet_y_mm/SCALE, state.outlet_y_mm/SCALE, 
        state.inlet_radius_mm/SCALE, state.outlet_radius_mm/SCALE, 
        state.inlet_z_mm/SCALE, state.outlet_z_mm/SCALE, 
        res_y, res_z
    )
    
    solver.update_params(
        state.inlet_y_mm/SCALE, state.outlet_y_mm/SCALE, 
        state.inlet_radius_mm/SCALE, state.outlet_radius_mm/SCALE,
        state.inlet_z_mm/SCALE, state.outlet_z_mm/SCALE,
        state.inlet_flow, state.outlet_flow
    )

    # Taichi GUI Setup
    window = ti.ui.Window("3D View", (1000, 800), pos=(500, 50))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    
    # Camera State
    camera_pos = np.array([res_x * 1.5, res_y * 1.5, res_z * 2.0], dtype=np.float32)
    camera_target = np.array([res_x / 2.0, res_y / 2.0, res_z / 2.0], dtype=np.float32)
    camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    last_mouse_pos = None
    
    # メインループ
    while window.running and dpg.is_dearpygui_running():
        # DearPyGuiのフレームを処理
        dpg.render_dearpygui_frame()
        
        # 寸法更新チェック
        if state.needs_dimension_update:
            res_x = int(state.tank_width / SCALE)
            res_y = int(state.tank_height / SCALE)
            res_z = int(state.tank_depth / SCALE)
            solver = FluidSolver(res_x, res_y, res_z, state.current_num_particles)
            update_box_geometry(res_x, res_y, res_z)
            camera_target = np.array([res_x/2, res_y/2, res_z/2], dtype=np.float32)
            camera_pos = camera_target + np.array([res_x, res_y/2, res_z], dtype=np.float32)
            state.needs_dimension_update = False
        
        # 粒子リセットチェック
        if state.needs_particle_reset:
            solver.init_particles()
            state.needs_particle_reset = False
        
        # 粒子数更新チェック
        if state.needs_particle_count_update:
            if state.target_num_particles != state.current_num_particles:
                state.current_num_particles = state.target_num_particles
                solver = FluidSolver(res_x, res_y, res_z, state.current_num_particles)
                try:
                    dpg.set_value("particle_count_text", f"現在: {state.current_num_particles:,}")
                except:
                    pass
            state.needs_particle_count_update = False
        
        # パイプジオメトリ更新
        update_pipe_geometry(
            state.inlet_y_mm/SCALE, state.outlet_y_mm/SCALE, 
            state.inlet_radius_mm/SCALE, state.outlet_radius_mm/SCALE, 
            state.inlet_z_mm/SCALE, state.outlet_z_mm/SCALE, 
            res_y, res_z
        )
        
        # ソルバーパラメータ更新
        solver.update_params(
            state.inlet_y_mm/SCALE, state.outlet_y_mm/SCALE, 
            state.inlet_radius_mm/SCALE, state.outlet_radius_mm/SCALE, 
            state.inlet_z_mm/SCALE, state.outlet_z_mm/SCALE, 
            state.inlet_flow, state.outlet_flow
        )
        
        solver.step()
        
        # Camera Control
        curr_mouse_pos = np.array(window.get_cursor_pos())
        
        if last_mouse_pos is not None:
            delta = curr_mouse_pos - last_mouse_pos
            is_shift = window.is_pressed(ti.ui.SHIFT)
            
            if window.is_pressed(ti.ui.RMB):
                if is_shift:  # Zoom
                    zoom_speed = 5.0
                    diff = camera_pos - camera_target
                    dist = np.linalg.norm(diff)
                    dir_vec = diff / dist
                    new_dist = max(1.0, dist + delta[1] * zoom_speed)
                    camera_pos = camera_target + dir_vec * new_dist
                else:  # Orbit
                    diff = camera_pos - camera_target
                    radius = np.linalg.norm(diff)
                    theta = np.arctan2(diff[2], diff[0])
                    phi = np.arccos(diff[1] / radius)
                    theta += delta[0] * 5.0 
                    phi += delta[1] * 5.0  # 上下反転
                    phi = np.clip(phi, 0.01, 3.14)
                    camera_pos[0] = camera_target[0] + radius * np.sin(phi) * np.cos(theta)
                    camera_pos[1] = camera_target[1] + radius * np.cos(phi)
                    camera_pos[2] = camera_target[2] + radius * np.sin(phi) * np.sin(theta)
            
            elif window.is_pressed(ti.ui.MMB):  # Pan
                forward = (camera_target - camera_pos) / np.linalg.norm(camera_target - camera_pos)
                right = np.cross(forward, camera_up) / np.linalg.norm(np.cross(forward, camera_up))
                up = np.cross(right, forward)
                displacement = (right * -delta[0] + up * -delta[1]) * 100.0
                camera_pos += displacement
                camera_target += displacement
                
        last_mouse_pos = curr_mouse_pos
        
        # Rendering
        camera.position(camera_pos[0], camera_pos[1], camera_pos[2])
        camera.lookat(camera_target[0], camera_target[1], camera_target[2])
        camera.up(0, 1, 0)
        scene.set_camera(camera)
        
        scene.point_light(pos=(res_x/2, res_y*1.5, res_z/2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))
        
        scene.particles(solver.particle_pos, radius=0.5, per_vertex_color=solver.particle_color)
        scene.lines(box_vertices, indices=box_indices, color=(1, 1, 1), width=2.0)
        scene.lines(pipe_v_field, indices=pipe_i_field, color=(0.8, 0.8, 0.8), width=2.0, vertex_count=num_pipe_indices[None])
        
        canvas.scene(scene)
        window.show()
    
    dpg.destroy_context()

if __name__ == "__main__":
    main()
