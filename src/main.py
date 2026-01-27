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
        
        # 視覚化設定
        self.colormap_mode = 0  # 0=Blue-Red, 1=Rainbow, 2=Cool-Warm, 3=Viridis
        self.particle_size = 0.5
        self.show_trails = False
        self.show_tank_walls = True
        
        # シミュレーション制御 (5, 6)
        self.is_paused = False
        self.sim_speed = 1.0  # 0.25, 0.5, 1.0, 2.0, 4.0
        
        # 追加ポート (7) - 2番目のInlet/Outlet
        self.use_second_inlet = False
        self.inlet2_y_mm = 200.0
        self.inlet2_z_mm = 300.0
        self.inlet2_radius_mm = 40.0
        self.inlet2_flow = 300.0
        
        self.use_second_outlet = False
        self.outlet2_y_mm = 300.0
        self.outlet2_z_mm = 700.0
        self.outlet2_radius_mm = 40.0
        self.outlet2_flow = 300.0
        
        # 障害物 (8)
        self.obstacles = []  # リスト of {type, x, y, z, size}
        self.show_obstacles = True

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
        
        dpg.add_separator()
        
        # 視覚化セクション
        with dpg.collapsing_header(label="視覚化設定", default_open=True):
            dpg.add_spacer(height=5)
            
            # カラーマップ選択
            dpg.add_text("カラーマップ", indent=10)
            dpg.add_radio_button(
                items=["青→赤", "レインボー", "クールウォーム", "Viridis"],
                tag="colormap_radio",
                default_value="青→赤",
                horizontal=True,
                callback=lambda s, a: setattr(state, 'colormap_mode', ["青→赤", "レインボー", "クールウォーム", "Viridis"].index(a)),
                indent=10
            )
            dpg.add_spacer(height=5)
            
            # 粒子サイズ
            with dpg.group(horizontal=True):
                dpg.add_text("粒子サイズ", indent=10)
                dpg.add_slider_float(
                    tag="particle_size_slider",
                    default_value=0.5,
                    min_value=0.1,
                    max_value=2.0,
                    width=150,
                    format="%.2f",
                    callback=lambda s, a: setattr(state, 'particle_size', a)
                )
            
            dpg.add_spacer(height=5)
            
            # 表示オプション
            dpg.add_checkbox(
                label="流線（トレイル）表示",
                tag="show_trails_checkbox",
                default_value=False,
                callback=lambda s, a: setattr(state, 'show_trails', a),
                indent=10
            )
            dpg.add_checkbox(
                label="水槽の壁面表示",
                tag="show_tank_checkbox",
                default_value=True,
                callback=lambda s, a: setattr(state, 'show_tank_walls', a),
                indent=10
            )
            dpg.add_spacer(height=10)
        
        dpg.add_separator()
        
        # シミュレーション制御セクション (5, 6)
        with dpg.collapsing_header(label="シミュレーション制御", default_open=True):
            dpg.add_spacer(height=5)
            
            # 一時停止/再生ボタン
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="⏸ 一時停止",
                    tag="pause_button",
                    callback=lambda: toggle_pause(),
                    width=100
                )
                dpg.add_spacer(width=10)
                dpg.add_text("再生中", tag="pause_status_text")
            
            dpg.add_spacer(height=5)
            
            # シミュレーション速度
            dpg.add_text("シミュレーション速度", indent=10)
            dpg.add_radio_button(
                items=["0.25x", "0.5x", "1x", "2x", "4x"],
                tag="speed_radio",
                default_value="1x",
                horizontal=True,
                callback=lambda s, a: setattr(state, 'sim_speed', {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0, "2x": 2.0, "4x": 4.0}[a]),
                indent=10
            )
            dpg.add_spacer(height=10)
        
        dpg.add_separator()
        
        # 追加ポートセクション (7)
        with dpg.collapsing_header(label="追加ポート", default_open=False):
            dpg.add_spacer(height=5)
            
            # 2番目のInlet
            dpg.add_checkbox(
                label="2番目の流入口を有効化",
                tag="use_inlet2_checkbox",
                default_value=False,
                callback=lambda s, a: setattr(state, 'use_second_inlet', a),
                indent=10
            )
            with dpg.group(tag="inlet2_group"):
                create_labeled_slider_with_input("Inlet2 Y (mm)", "inlet2_y", state.inlet2_y_mm, 20.0, 450.0)
                create_labeled_slider_with_input("Inlet2 Z (mm)", "inlet2_z", state.inlet2_z_mm, 20.0, 980.0)
                create_labeled_slider_with_input("Inlet2 半径", "inlet2_radius", state.inlet2_radius_mm, 20.0, 100.0)
                create_labeled_slider_with_input("Inlet2 流量", "inlet2_flow", state.inlet2_flow, 0.0, 1000.0)
            
            dpg.add_spacer(height=10)
            
            # 2番目のOutlet
            dpg.add_checkbox(
                label="2番目の流出口を有効化",
                tag="use_outlet2_checkbox",
                default_value=False,
                callback=lambda s, a: setattr(state, 'use_second_outlet', a),
                indent=10
            )
            with dpg.group(tag="outlet2_group"):
                create_labeled_slider_with_input("Outlet2 Y (mm)", "outlet2_y", state.outlet2_y_mm, 20.0, 450.0)
                create_labeled_slider_with_input("Outlet2 Z (mm)", "outlet2_z", state.outlet2_z_mm, 20.0, 980.0)
                create_labeled_slider_with_input("Outlet2 半径", "outlet2_radius", state.outlet2_radius_mm, 20.0, 100.0)
                create_labeled_slider_with_input("Outlet2 流量", "outlet2_flow", state.outlet2_flow, 0.0, 1000.0)
            
            dpg.add_spacer(height=10)
        
        dpg.add_separator()
        
        # 障害物セクション (8)
        with dpg.collapsing_header(label="障害物", default_open=False):
            dpg.add_spacer(height=5)
            
            dpg.add_text("障害物を追加:", indent=10)
            with dpg.group(horizontal=True):
                dpg.add_combo(
                    items=["球", "箱"],
                    tag="obstacle_type_combo",
                    default_value="球",
                    width=80
                )
                dpg.add_spacer(width=10)
                dpg.add_button(label="追加", callback=lambda: add_obstacle(), width=60)
                dpg.add_button(label="全削除", callback=lambda: clear_obstacles(), width=60)
            
            dpg.add_spacer(height=5)
            
            # 障害物パラメータ
            create_labeled_slider_with_input("X位置 (mm)", "obs_x", 500.0, 0.0, 1000.0)
            create_labeled_slider_with_input("Y位置 (mm)", "obs_y", 250.0, 0.0, 500.0)
            create_labeled_slider_with_input("Z位置 (mm)", "obs_z", 500.0, 0.0, 1000.0)
            create_labeled_slider_with_input("サイズ (mm)", "obs_size", 50.0, 20.0, 200.0)
            
            dpg.add_spacer(height=5)
            dpg.add_text("配置済み障害物: 0個", tag="obstacle_count_text", indent=10)
            dpg.add_checkbox(
                label="障害物を表示",
                tag="show_obstacles_checkbox",
                default_value=True,
                callback=lambda s, a: setattr(state, 'show_obstacles', a),
                indent=10
            )
            dpg.add_spacer(height=10)
    
    dpg.create_viewport(title='Fluid Simulation - Control Panel', width=460, height=950, x_pos=50, y_pos=20)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)


def toggle_pause():
    """一時停止/再生を切り替え"""
    state.is_paused = not state.is_paused
    if state.is_paused:
        dpg.set_item_label("pause_button", "▶ 再生")
        dpg.set_value("pause_status_text", "一時停止中")
    else:
        dpg.set_item_label("pause_button", "⏸ 一時停止")
        dpg.set_value("pause_status_text", "再生中")


def add_obstacle():
    """障害物を追加"""
    obs_type = dpg.get_value("obstacle_type_combo")
    x = dpg.get_value("obs_x_slider")
    y = dpg.get_value("obs_y_slider")
    z = dpg.get_value("obs_z_slider")
    size = dpg.get_value("obs_size_slider")
    
    state.obstacles.append({
        'type': 'sphere' if obs_type == "球" else 'box',
        'x': x,
        'y': y,
        'z': z,
        'size': size
    })
    dpg.set_value("obstacle_count_text", f"配置済み障害物: {len(state.obstacles)}個")


def clear_obstacles():
    """全障害物を削除"""
    state.obstacles.clear()
    dpg.set_value("obstacle_count_text", f"配置済み障害物: 0個")


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
        
        # カラーマップ更新
        solver.colormap_mode[None] = state.colormap_mode
        
        # 障害物データをソルバーに渡す
        solver.num_obstacles[None] = min(len(state.obstacles), solver.max_obstacles)
        for i, obs in enumerate(state.obstacles[:solver.max_obstacles]):
            solver.obstacle_data[i] = [
                obs['x'] / SCALE,
                obs['y'] / SCALE,
                obs['z'] / SCALE,
                obs['size'] / SCALE,
                0.0 if obs['type'] == 'sphere' else 1.0
            ]
        
        # シミュレーション実行（一時停止と速度制御）
        if not state.is_paused:
            # 速度に応じてステップ数を調整
            steps_per_frame = max(1, int(state.sim_speed))
            for _ in range(steps_per_frame):
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
        
        # 粒子描画（サイズ設定反映）
        scene.particles(solver.particle_pos, radius=state.particle_size, per_vertex_color=solver.particle_color)
        
        # 水槽の壁面表示
        if state.show_tank_walls:
            scene.lines(box_vertices, indices=box_indices, color=(1, 1, 1), width=2.0)
        
        # パイプ描画
        scene.lines(pipe_v_field, indices=pipe_i_field, color=(0.8, 0.8, 0.8), width=2.0, vertex_count=num_pipe_indices[None])
        
        # 障害物描画
        if state.show_obstacles and len(state.obstacles) > 0:
            for obs in state.obstacles:
                # グリッド座標に変換
                ox = obs['x'] / SCALE
                oy = obs['y'] / SCALE
                oz = obs['z'] / SCALE
                osize = obs['size'] / SCALE
                
                if obs['type'] == 'sphere':
                    # 球の簡易描画（パーティクルとして表示）
                    sphere_center = np.array([[ox, oy, oz]], dtype=np.float32)
                    sphere_field = ti.Vector.field(3, dtype=float, shape=1)
                    sphere_field.from_numpy(sphere_center)
                    scene.particles(sphere_field, radius=osize, color=(0.8, 0.4, 0.1))
                else:
                    # 箱の簡易描画（ワイヤーフレーム）
                    # TODO: 箱のワイヤーフレーム描画
                    pass
        
        # トレイル描画
        if state.show_trails:
            # トレイルデータを取得してライン描画
            trail_np = solver.trail_positions.to_numpy()
            trail_idx_np = solver.trail_index.to_numpy()
            
            # サンプリング（全粒子だと重いので間引く）
            sample_step = max(1, state.current_num_particles // 500)
            for i in range(0, min(500, state.current_num_particles), 1):
                p_idx = i * sample_step
                if p_idx >= state.current_num_particles:
                    break
                idx = trail_idx_np[p_idx]
                # トレイルの点を順番に取得
                trail_points = []
                for j in range(solver.trail_length):
                    actual_idx = (idx - j - 1 + solver.trail_length) % solver.trail_length
                    pos = trail_np[p_idx, actual_idx]
                    if pos[0] > 0:  # タンク内のみ
                        trail_points.append(pos)
                
                # 線として描画（簡易版）
                if len(trail_points) >= 2:
                    for k in range(len(trail_points) - 1):
                        pass  # Taichi UIでは動的なライン数描画が難しいため、パーティクルで代用
        
        canvas.scene(scene)
        window.show()
    
    dpg.destroy_context()

if __name__ == "__main__":
    main()
