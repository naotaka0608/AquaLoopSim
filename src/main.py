import numpy as np
import dearpygui.dearpygui as dpg
import os
import json
import datetime
import pyvista as pv
from src.config import *
from src.config import DEFAULT_NUM_PARTICLES, MIN_NUM_PARTICLES, MAX_NUM_PARTICLES
from src.solver import FluidSolver

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
        self.background_color_mode = "Black" # Black, Dark Gray, Light Gray, White, Paraview Blue
        self.particle_size = 8.0 # PyVista uses point size, not radius
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
        
        # 分析ツール (9, 10, 11)
        # 流量計表示
        self.show_flow_meter = True
        self.inlet_particle_count = 0
        self.outlet_particle_count = 0
        
        # 断面ビュー
        self.show_cross_section = False
        self.cross_section_axis = 'X'  # X, Y, Z
        self.cross_section_pos = 50.0  # % (0-100)
        
        # スクリーンショット/録画
        self.screenshot_dir = "./screenshots"
        self.is_recording = False
        self.recording_frames = []
        self.frame_count = 0
        
        # シミュレーション経過時間
        self.sim_elapsed_time = 0.0  # 秒
    
    def to_dict(self):
        """保存用の辞書に変換"""
        return {
            'tank_width': self.tank_width,
            'tank_height': self.tank_height,
            'tank_depth': self.tank_depth,
            'inlet_y_mm': self.inlet_y_mm,
            'inlet_z_mm': self.inlet_z_mm,
            'inlet_radius_mm': self.inlet_radius_mm,
            'outlet_y_mm': self.outlet_y_mm,
            'outlet_z_mm': self.outlet_z_mm,
            'outlet_radius_mm': self.outlet_radius_mm,
            'inlet_flow': self.inlet_flow,
            'outlet_flow': self.outlet_flow,
            'is_sync': self.is_sync,
            'target_num_particles': self.target_num_particles,
            'colormap_mode': self.colormap_mode,
            'background_color_mode': self.background_color_mode,
            'particle_size': self.particle_size,
            'show_tank_walls': self.show_tank_walls,
            'sim_speed': self.sim_speed,
            'use_second_inlet': self.use_second_inlet,
            'inlet2_y_mm': self.inlet2_y_mm,
            'inlet2_z_mm': self.inlet2_z_mm,
            'inlet2_radius_mm': self.inlet2_radius_mm,
            'inlet2_flow': self.inlet2_flow,
            'use_second_outlet': self.use_second_outlet,
            'outlet2_y_mm': self.outlet2_y_mm,
            'outlet2_z_mm': self.outlet2_z_mm,
            'outlet2_radius_mm': self.outlet2_radius_mm,
            'outlet2_flow': self.outlet2_flow,
            'obstacles': self.obstacles,
        }
    
    def from_dict(self, data):
        """辞書からパラメータを読み込み"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

# 設定ファイルパス
CONFIG_FILE = "./config.json"

def save_config():
    """パラメータをJSONに保存"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"設定を保存しました: {CONFIG_FILE}")
        return True
    except Exception as e:
        print(f"設定の保存に失敗: {e}")
        return False

def load_config():
    """JSONからパラメータを読み込み"""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            state.from_dict(data)
            print(f"設定を読み込みました: {CONFIG_FILE}")
            return True
    except Exception as e:
        print(f"設定の読み込みに失敗: {e}")
    return False

state = AppState()

# 起動時に設定を読み込み
load_config()

# Resolution Scale
SCALE = 10.0 

# Scene Data
box_mesh = None
pipe_mesh = None


def update_box_geometry(res_x, res_y, res_z):
    global box_mesh
    # 8 corners
    corners = np.array([
        [0, 0, 0], [res_x, 0, 0], [res_x, 0, res_z], [0, 0, res_z],
        [0, res_y, 0], [res_x, res_y, 0], [res_x, res_y, res_z], [0, res_y, res_z]
    ], dtype=np.float32)
    
    # 12 edges (2 indices per edge)
    # PyVista lines format: [2, i0, i1, 2, i2, i3, ...]
    indices = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ], dtype=np.int32)
    
    # Construct PyVista PolyData
    # padding lines array with connection count (2)
    lines_flat = np.hstack([[2, i0, i1] for i0, i1 in indices])
    
    box_mesh = pv.PolyData(corners)
    box_mesh.lines = lines_flat

def update_pipe_geometry(in_y, out_y, in_rad, out_rad, in_z, out_z, res_y, res_z):
    global pipe_mesh
    verts = []
    lines_list = []
    num_pipe_segs = 16
    
    def add_pipe_mesh(start_pos, end_pos, radius):
        start_idx = len(verts)
        for i in range(num_pipe_segs):
            theta = (i / num_pipe_segs) * 2 * np.pi
            y = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            p1 = [start_pos[0], start_pos[1] + y, start_pos[2] + z]
            p2 = [end_pos[0], end_pos[1] + y, end_pos[2] + z]
            
            verts.append(p1)
            verts.append(p2)
            
            curr = start_idx + 2*i
            next_seg = start_idx + 2*((i+1)%num_pipe_segs)
            
            # Lines: p1-p2 (length of pipe), p1-next_p1 (ring), p2-next_p2 (ring)
            lines_list.append([2, curr, next_seg])       # Ring 1 segment
            lines_list.append([2, curr+1, next_seg+1])   # Ring 2 segment
            lines_list.append([2, curr, curr+1])         # Length segment

    add_pipe_mesh([0, in_y, in_z], [-20, in_y, in_z], in_rad)
    add_pipe_mesh([0, out_y, out_z], [-20, out_y, out_z], out_rad)
    
    if verts:
        points = np.array(verts, dtype=np.float32)
        lines_flat = np.hstack(lines_list)
        pipe_mesh = pv.PolyData(points)
        pipe_mesh.lines = lines_flat
    else:
        pipe_mesh = None


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

def on_save_config():
    """設定保存ボタンのコールバック"""
    update_state_from_ui()  # UIから最新の値を取得
    if save_config():
        dpg.set_value("config_status_text", "✓ 設定を保存しました")
    else:
        dpg.set_value("config_status_text", "✗ 保存に失敗しました")

def on_load_config():
    """設定読み込みボタンのコールバック"""
    if load_config():
        # UIを更新
        try:
            dpg.set_value("tank_width_slider", state.tank_width)
            dpg.set_value("tank_height_slider", state.tank_height)
            dpg.set_value("tank_depth_slider", state.tank_depth)
            dpg.set_value("inlet_y_slider", state.inlet_y_mm)
            dpg.set_value("inlet_z_slider", state.inlet_z_mm)
            dpg.set_value("inlet_radius_slider", state.inlet_radius_mm)
            dpg.set_value("inlet_flow_slider", state.inlet_flow)
            dpg.set_value("outlet_y_slider", state.outlet_y_mm)
            dpg.set_value("outlet_z_slider", state.outlet_z_mm)
            dpg.set_value("outlet_radius_slider", state.outlet_radius_mm)
            dpg.set_value("outlet_flow_slider", state.outlet_flow)
            dpg.set_value("sync_checkbox", state.is_sync)
            dpg.set_value("particle_slider", state.target_num_particles)
            dpg.set_value("particle_size_slider", state.particle_size)
            dpg.set_value("show_tank_checkbox", state.show_tank_walls)
            dpg.set_value("config_status_text", "✓ 設定を読み込みました")
        except Exception as e:
            print(f"UI更新エラー: {e}")
    else:
        dpg.set_value("config_status_text", "✗ 読み込みに失敗しました")

def create_labeled_slider_with_input(label, tag_base, default_val, min_val, max_val, format_str="%.1f"):
    """ラベル + スライダー + インプットボックスのセットを作成"""
    slider_tag = f"{tag_base}_slider"
    input_tag = f"{tag_base}_input"
    
    dpg.add_text(label)
    with dpg.group(horizontal=True):
        dpg.add_slider_float(
            tag=slider_tag,
            default_value=default_val,
            min_value=min_val,
            max_value=max_val,
            width=280,
            format=format_str,
            callback=sync_slider_to_input,
            user_data=input_tag
        )
        dpg.add_input_float(
            tag=input_tag,
            default_value=default_val,
            width=90,
            step=0,
            callback=sync_input_to_slider,
            user_data=slider_tag
        )
    dpg.add_spacer(height=3)

def _create_config_section():
    """設定保存/読み込みセクション"""
    with dpg.group(horizontal=True):
        dpg.add_button(label="Save", callback=lambda: on_save_config(), width=100)
        dpg.add_spacer(width=5)
        dpg.add_button(label="Load", callback=lambda: on_load_config(), width=100)
        dpg.add_spacer(width=5)
        dpg.add_button(label="Reset", callback=on_reset_particles, width=100)
    dpg.add_text("", tag="config_status_text")
    dpg.add_spacer(height=5)
    dpg.add_separator()


def _create_tank_section():
    """タンク寸法セクション"""
    with dpg.collapsing_header(label="タンク寸法", default_open=False):
        dpg.add_spacer(height=5)
        create_labeled_slider_with_input("幅 (mm)", "tank_width", state.tank_width, 500.0, 2000.0)
        create_labeled_slider_with_input("高さ (mm)", "tank_height", state.tank_height, 200.0, 1000.0)
        create_labeled_slider_with_input("奥行 (mm)", "tank_depth", state.tank_depth, 500.0, 2000.0)
        dpg.add_spacer(height=5)
        dpg.add_button(label="寸法を適用", callback=on_apply_dimensions, width=150)
        dpg.add_spacer(height=10)
    dpg.add_separator()


def _create_inlet_section():
    """流入口セクション"""
    with dpg.collapsing_header(label="流入口 (Inlet)", default_open=False):
        dpg.add_spacer(height=5)
        create_labeled_slider_with_input("Y位置 (mm)", "inlet_y", state.inlet_y_mm, 20.0, 450.0)
        create_labeled_slider_with_input("Z位置 (mm)", "inlet_z", state.inlet_z_mm, 20.0, 980.0)
        create_labeled_slider_with_input("半径 (mm)", "inlet_radius", state.inlet_radius_mm, 20.0, 150.0)
        create_labeled_slider_with_input("流量 (L/min)", "inlet_flow", state.inlet_flow, 0.0, 2000.0)
        dpg.add_spacer(height=10)
    dpg.add_separator()


def _create_outlet_section():
    """流出口セクション"""
    with dpg.collapsing_header(label="流出口 (Outlet)", default_open=False):
        dpg.add_spacer(height=5)
        dpg.add_checkbox(label="流入口と同期", tag="sync_checkbox", default_value=state.is_sync, callback=lambda: update_state_from_ui())
        dpg.add_spacer(height=5)
        create_labeled_slider_with_input("Y位置 (mm)", "outlet_y", state.outlet_y_mm, 20.0, 450.0)
        create_labeled_slider_with_input("Z位置 (mm)", "outlet_z", state.outlet_z_mm, 20.0, 980.0)
        create_labeled_slider_with_input("半径 (mm)", "outlet_radius", state.outlet_radius_mm, 20.0, 150.0)
        create_labeled_slider_with_input("流量 (L/min)", "outlet_flow", state.outlet_flow, 0.0, 2000.0)
        dpg.add_spacer(height=10)
    dpg.add_separator()


def _create_particle_section():
    """粒子設定セクション"""
    with dpg.collapsing_header(label="粒子設定", default_open=False):
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
        dpg.add_button(label="粒子数を適用", callback=on_apply_particles, width=120)
        dpg.add_spacer(height=10)
    dpg.add_separator()
    dpg.add_spacer(height=10)


def _create_visualization_section():
    """視覚化設定セクション"""
    with dpg.collapsing_header(label="視覚化設定", default_open=False):
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
        # 背景色選択
        dpg.add_text("背景色", indent=10)
        dpg.add_combo(
            items=["Black", "Dark Gray", "Light Gray", "White", "Paraview Blue"],
            tag="background_color_combo",
            default_value=state.background_color_mode,
            callback=lambda s, a: setattr(state, 'background_color_mode', a),
            width=200,
            indent=10
        )
        dpg.add_spacer(height=5)
        # 粒子サイズ
        with dpg.group(horizontal=True):
            dpg.add_text("粒子サイズ", indent=10)
            dpg.add_slider_float(
                tag="particle_size_slider",
                default_value=state.particle_size,
                min_value=1.0,
                max_value=30.0,
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


def _create_simulation_control_section():
    """シミュレーション制御セクション"""
    with dpg.collapsing_header(label="シミュレーション制御", default_open=False):
        dpg.add_spacer(height=5)
        # 一時停止/再生ボタン
        with dpg.group(horizontal=True):
            dpg.add_button(label="Pause", tag="pause_button", callback=lambda: toggle_pause(), width=120)
            dpg.add_spacer(width=15)
            dpg.add_text("Playing", tag="pause_status_text")
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
        dpg.add_spacer(height=5)
        # 経過時間表示
        dpg.add_text("経過時間: 0.0 秒", tag="elapsed_time_text", indent=10)
        dpg.add_spacer(height=10)
    dpg.add_separator()


def _create_additional_ports_section():
    """追加ポートセクション"""
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


def _create_obstacles_section():
    """障害物セクション"""
    with dpg.collapsing_header(label="障害物", default_open=False):
        dpg.add_spacer(height=5)
        dpg.add_text("障害物を追加:", indent=10)
        with dpg.group(horizontal=True):
            dpg.add_combo(items=["球", "箱"], tag="obstacle_type_combo", default_value="球", width=80)
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
    dpg.add_separator()


def _create_analysis_section():
    """分析ツールセクション"""
    with dpg.collapsing_header(label="分析ツール", default_open=False):
        dpg.add_spacer(height=5)
        # 流量計表示
        dpg.add_checkbox(
            label="流量計を表示",
            tag="show_flow_meter_checkbox",
            default_value=True,
            callback=lambda s, a: setattr(state, 'show_flow_meter', a),
            indent=10
        )
        dpg.add_text("流入口: 0 粒子/秒", tag="inlet_flow_text", indent=10)
        dpg.add_text("流出口: 0 粒子/秒", tag="outlet_flow_text", indent=10)
        dpg.add_text("平均速度: 0.0 mm/s", tag="avg_speed_text", indent=10)
        dpg.add_spacer(height=10)
        dpg.add_separator()
        # 断面ビュー
        dpg.add_checkbox(
            label="断面ビューを表示",
            tag="show_cross_section_checkbox",
            default_value=False,
            callback=lambda s, a: setattr(state, 'show_cross_section', a),
            indent=10
        )
        dpg.add_text("断面軸:", indent=10)
        dpg.add_radio_button(
            items=["X", "Y", "Z"],
            tag="cross_section_axis_radio",
            default_value="X",
            horizontal=True,
            callback=lambda s, a: setattr(state, 'cross_section_axis', a),
            indent=10
        )
        with dpg.group(horizontal=True):
            dpg.add_text("位置 (%)", indent=10)
            dpg.add_slider_float(
                tag="cross_section_pos_slider",
                default_value=50.0,
                min_value=0.0,
                max_value=100.0,
                width=150,
                callback=lambda s, a: setattr(state, 'cross_section_pos', a)
            )
        dpg.add_spacer(height=10)
        dpg.add_separator()
        # Screenshot / Recording
        dpg.add_text("Screenshot / Recording")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Screenshot", callback=lambda: take_screenshot(), width=150)
            dpg.add_button(label="Start Record", tag="record_button", callback=lambda: toggle_recording(), width=150)
        dpg.add_text("Save to: ./screenshots", tag="save_path_text")
        dpg.add_text("Frames: 0", tag="frame_count_text")
        dpg.add_spacer(height=10)


def setup_dpg_ui():
    """Dear PyGui UIをセットアップ"""
    dpg.create_context()
    
    # フォント設定（日本語対応・大きめサイズ）
    with dpg.font_registry():
        font_path = "C:/Windows/Fonts/meiryo.ttc"
        try:
            with dpg.font(font_path, 20) as default_font:
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Japanese)
            dpg.bind_font(default_font)
        except Exception as e:
            print(f"Font loading failed: {e}")
    
    # テーマ設定（見やすいダークテーマ）
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (40, 40, 45, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (30, 30, 30, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (50, 80, 120, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (55, 55, 60, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (70, 70, 80, 255))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (80, 150, 220, 255))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (100, 180, 255, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Button, (60, 90, 130, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (80, 120, 170, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (100, 150, 200, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Header, (50, 80, 120, 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (70, 100, 150, 255))
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (100, 200, 150, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text, (230, 230, 235, 255))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 10, 8)
            dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing, 8, 6)
    
    dpg.bind_theme(global_theme)
    
    # メインウィンドウ
    with dpg.window(label="Control Panel", tag="main_window", width=500, height=800, no_close=True):
        _create_config_section()
        _create_tank_section()
        _create_inlet_section()
        _create_outlet_section()
        _create_particle_section()
        _create_visualization_section()
        _create_simulation_control_section()
        _create_additional_ports_section()
        _create_obstacles_section()
        _create_analysis_section()
    
    dpg.create_viewport(title='Fluid Simulation', width=520, height=1050, x_pos=50, y_pos=10)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)


def toggle_pause():
    """一時停止/再生を切り替え"""
    state.is_paused = not state.is_paused
    if state.is_paused:
        dpg.set_item_label("pause_button", "Play")
        dpg.set_value("pause_status_text", "Paused")
    else:
        dpg.set_item_label("pause_button", "Pause")
        dpg.set_value("pause_status_text", "Playing")


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


# グローバル変数（録画用）
_window_ref = None


def take_screenshot():
    """スクリーンショットを保存"""
    global _window_ref
    if _window_ref is None:
        print("ウィンドウが初期化されていません")
        return
    
    # ディレクトリを確保
    os.makedirs(state.screenshot_dir, exist_ok=True)
    
    # ファイル名を生成
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{state.screenshot_dir}/screenshot_{timestamp}.png"
    
    try:
        _window_ref.save_image(filename)
        dpg.set_value("save_path_text", f"Saved: {filename}")
        print(f"Screenshot saved: {filename}")
    except Exception as e:
        print(f"Screenshot error: {e}")


def toggle_recording():
    """録画開始/停止を切り替え"""
    state.is_recording = not state.is_recording
    
    if state.is_recording:
        dpg.set_item_label("record_button", "Stop Record")
        state.frame_count = 0
        os.makedirs(state.screenshot_dir, exist_ok=True)
        dpg.set_value("save_path_text", "Recording...")
    else:
        dpg.set_item_label("record_button", "Start Record")
        dpg.set_value("save_path_text", f"Done: {state.frame_count} frames")


def save_recording_frame():
    """録画中のフレームを保存"""
    global _window_ref
    if _window_ref is None or not state.is_recording:
        return
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{state.screenshot_dir}/frame_{timestamp}_{state.frame_count:05d}.png"
    
    try:
        _window_ref.save_image(filename)
        state.frame_count += 1
        dpg.set_value("frame_count_text", f"Frames: {state.frame_count}")
    except:
        pass


def calculate_flow_stats(solver, res_x, res_y, res_z, 
                         inlet_y, inlet_z, inlet_r,
                         outlet_y, outlet_z, outlet_r):
    """流量統計を計算"""
    positions = solver.particle_pos
    velocities = solver.particle_vel
    
    # X座標チェック (壁際)
    x_mask = positions[:, 0] < 5.0
    
    # YZ距離の二乗計算
    # Inlet
    dy_in = positions[:, 1] - inlet_y
    dz_in = positions[:, 2] - inlet_z
    dist_sq_in = dy_in*dy_in + dz_in*dz_in
    inlet_mask = np.logical_and(x_mask, dist_sq_in < (inlet_r * inlet_r))
    
    # Outlet
    dy_out = positions[:, 1] - outlet_y
    dz_out = positions[:, 2] - outlet_z
    dist_sq_out = dy_out*dy_out + dz_out*dz_out
    outlet_mask = np.logical_and(x_mask, dist_sq_out < (outlet_r * outlet_r))
    
    inlet_count = np.sum(inlet_mask)
    outlet_count = np.sum(outlet_mask)
    
    # 平均速度
    speeds = np.linalg.norm(velocities, axis=1)
    avg_speed = np.mean(speeds) * SCALE  # mm/s に変換
    
    return inlet_count, outlet_count, avg_speed

def main():
    # DearPyGui UIをセットアップ
    setup_dpg_ui()
    
    # ローディング画面を表示（最前面）
    loading_tag = "loading_overlay"
    with dpg.window(label="Loading", tag=loading_tag, modal=True, no_title_bar=True, no_move=True, width=400, height=150, pos=(60, 300)):
        dpg.add_text("システム初期化中...", indent=130)
        dpg.add_text("※ 初回起動は最適化処理のため時間がかかります。\n(10秒〜20秒程度お待ちください)", indent=30)
        dpg.add_spacer(height=10)
        dpg.add_loading_indicator(style=1, radius=6.0, color=(100, 200, 255, 255), indent=185)
    
    # ローディング画面を確実に描画させる
    for _ in range(10):
        dpg.render_dearpygui_frame()

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
    
    # Numba JITコンパイルのウォームアップ
    # これを行うことで、メインループ開始時のフリーズを防ぐ
    print("Warming up solver (JIT compilation)...")
    try:
        solver.step()
    except Exception as e:
        print(f"Warmup failed (non-fatal): {e}")

    # PyVista GUI Setup
    plotter = pv.Plotter(title="3D View", window_size=(1000, 800))
    # plotter.app_window.position = (500, 50) # Removed as it causes AttributeError
    
    # 粒子メッシュ初期化
    particles_mesh = pv.PolyData(solver.particle_pos)
    particles_mesh.point_data["rgb"] = solver.particle_color
    particles_actor = plotter.add_mesh(particles_mesh, scalars="rgb", rgb=True, point_size=state.particle_size, style='points', render_points_as_spheres=False, name="particles")
    
    # 壁面メッシュ初期化
    if box_mesh:
        plotter.add_mesh(box_mesh, color="white", style="wireframe", line_width=2, name="walls")
        
    # パイプメッシュ初期化
    if pipe_mesh:
        plotter.add_mesh(pipe_mesh, color="gray", style="wireframe", line_width=2, name="pipes")
    
    # ライトとカメラ設定
    plotter.add_light(pv.Light(position=(res_x/2, res_y*1.5, res_z/2), color='white', intensity=0.8))
    plotter.set_background('black')  # 背景を黒に設定
    plotter.camera_position = [
        (res_x * 1.5, res_y * 1.5, res_z * 2.0),  # Position
        (res_x / 2.0, res_y / 2.0, res_z / 2.0),  # Focal point
        (0.0, 1.0, 0.0)  # Up vector
    ]
    plotter.show_axes()
    
    # 非ブロッキングで表示
    plotter.show(interactive_update=True, auto_close=False)
    
    # グローバル参照を設定（スクリーンショット用）
    global _window_ref
    _window_ref = plotter
    
    # 流量計用のカウンター
    flow_update_counter = 0

    # 前回のパーティクルサイズを記録
    last_particle_size = state.particle_size
    last_background_mode = state.background_color_mode
    
    # ローディング画面を削除
    dpg.delete_item(loading_tag)
    
    # メインループ
    try:
        while not getattr(plotter, 'closed', getattr(plotter, '_closed', False)) and dpg.is_dearpygui_running():
            # DearPyGuiのフレームを処理
            dpg.render_dearpygui_frame()
            
            # PyVistaのイベント処理（描画更新含む）
            plotter.update()

            # 背景色の更新反映
            if state.background_color_mode != last_background_mode:
                bg_map = {
                    "Black": "black",
                    "Dark Gray": (0.2, 0.2, 0.2),
                    "Light Gray": (0.8, 0.8, 0.8),
                    "White": "white",
                    "Paraview Blue": (0.3, 0.3, 0.4)
                }
                new_color = bg_map.get(state.background_color_mode, "black")
                plotter.set_background(new_color)
                # 白背景の場合は文字色を黒にすると親切だが、今回は背景のみ
                last_background_mode = state.background_color_mode

            # パーティクルサイズの更新反映
            if state.particle_size != last_particle_size:
                if particles_actor:
                     particles_actor.prop.point_size = state.particle_size
                last_particle_size = state.particle_size
            
            # 寸法更新チェック
            if state.needs_dimension_update:
                res_x = int(state.tank_width / SCALE)
                res_y = int(state.tank_height / SCALE)
                res_z = int(state.tank_depth / SCALE)
                solver = FluidSolver(res_x, res_y, res_z, state.current_num_particles)
                update_box_geometry(res_x, res_y, res_z)
                
                # メッシュ再登録
                particles_mesh = pv.PolyData(solver.particle_pos)
                particles_mesh.point_data["rgb"] = solver.particle_color
                particles_actor = plotter.add_mesh(particles_mesh, scalars="rgb", rgb=True, point_size=state.particle_size, style='points', render_points_as_spheres=False, name="particles")
                
                if box_mesh:
                    plotter.add_mesh(box_mesh, color="white", style="wireframe", line_width=2, name="walls")
                
                # カメラリセット
                plotter.camera_position = [
                    (res_x * 1.5, res_y * 1.5, res_z * 2.0),
                    (res_x / 2.0, res_y / 2.0, res_z / 2.0),
                    (0.0, 1.0, 0.0)
                ]
                state.needs_dimension_update = False
            
            # 粒子リセットチェック
            if state.needs_particle_reset:
                solver.init_particles()
                state.sim_elapsed_time = 0.0
                state.needs_particle_reset = False
            
            # 粒子数更新チェック
            if state.needs_particle_count_update:
                if state.target_num_particles != state.current_num_particles:
                    state.current_num_particles = state.target_num_particles
                    solver = FluidSolver(res_x, res_y, res_z, state.current_num_particles)
                    
                    # メッシュ更新
                    particles_mesh = pv.PolyData(solver.particle_pos)
                    particles_mesh.point_data["rgb"] = solver.particle_color
                    particles_actor = plotter.add_mesh(particles_mesh, scalars="rgb", rgb=True, point_size=state.particle_size, style='points', render_points_as_spheres=False, name="particles")
                    
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
            if pipe_mesh:
                 plotter.add_mesh(pipe_mesh, color="gray", style="wireframe", line_width=2, name="pipes")
            
            # ソルバーパラメータ更新
            solver.update_params(
                state.inlet_y_mm/SCALE, state.outlet_y_mm/SCALE, 
                state.inlet_radius_mm/SCALE, state.outlet_radius_mm/SCALE, 
                state.inlet_z_mm/SCALE, state.outlet_z_mm/SCALE, 
                state.inlet_flow, state.outlet_flow
            )
            
            # カラーマップ更新 (Numba版では配列ではなくスカラ)
            solver.colormap_mode = state.colormap_mode
            
            # 障害物データをソルバーに渡す (Numba版では配列)
            solver.num_obstacles = min(len(state.obstacles), solver.max_obstacles)
            for i, obs in enumerate(state.obstacles[:solver.max_obstacles]):
                solver.obstacle_data[i] = [
                    obs['x'] / SCALE,
                    obs['y'] / SCALE,
                    obs['z'] / SCALE,
                    obs['size'] / SCALE,
                    0.0 if obs['type'] == 'sphere' else 1.0
                ]
            
            # シミュレーション実行
            if not state.is_paused:
                steps_per_frame = max(1, int(state.sim_speed))
                for _ in range(steps_per_frame):
                    solver.step()
                state.sim_elapsed_time += (1.0 / 60.0) * state.sim_speed
            
            # PyVistaメッシュ更新 (インプレース更新で高速化)
            particles_mesh.points[:] = solver.particle_pos
            particles_mesh.point_data["rgb"][:] = solver.particle_color
            
            # 断面ビュー更新
            if particles_actor:
                if state.show_cross_section:
                    # クリッピングプレーンの設定
                    # 中心位置計算
                    center = [
                        res_x/2 * SCALE, res_y/2 * SCALE, res_z/2 * SCALE
                    ]
                    # GUIの%指定から座標オフセット計算
                    offset_pct = (state.cross_section_pos - 50.0) / 100.0
                    
                    origin = [res_x/2, res_y/2, res_z/2]
                    normal = [1, 0, 0]
                    
                    if state.cross_section_axis == 'X':
                        origin[0] += res_x * offset_pct
                        normal = [1, 0, 0]
                    elif state.cross_section_axis == 'Y':
                        origin[1] += res_y * offset_pct
                        normal = [0, 1, 0]
                    elif state.cross_section_axis == 'Z':
                        origin[2] += res_z * offset_pct
                        normal = [0, 0, 1]
                        
                    # PyVistaのクリッピング適用
                    # Mapperに直接適用することで高速に反映
                    plane = pv.Plane(center=origin, direction=normal)
                    particles_actor.mapper.RemoveAllClippingPlanes()
                    particles_actor.mapper.AddClippingPlane(plane)
                else:
                     particles_actor.mapper.RemoveAllClippingPlanes()

            # 強制再描画
            plotter.render()
        
            # 経過時間表示更新
            try:
                minutes = int(state.sim_elapsed_time // 60)
                seconds = state.sim_elapsed_time % 60
                if minutes > 0:
                    dpg.set_value("elapsed_time_text", f"経過時間: {minutes}分 {seconds:.1f}秒")
                else:
                    dpg.set_value("elapsed_time_text", f"経過時間: {seconds:.1f} 秒")
            except:
                pass
            
            # 流量計更新
            flow_update_counter += 1
            if state.show_flow_meter and flow_update_counter >= 10:
                flow_update_counter = 0
                inlet_count, outlet_count, avg_speed = calculate_flow_stats(
                    solver, res_x, res_y, res_z,
                    state.inlet_y_mm/SCALE, state.inlet_z_mm/SCALE, state.inlet_radius_mm/SCALE,
                    state.outlet_y_mm/SCALE, state.outlet_z_mm/SCALE, state.outlet_radius_mm/SCALE
                )
                try:
                     dpg.set_value("inlet_flow_text", f"流入口: {inlet_count} 粒子")
                     dpg.set_value("outlet_flow_text", f"流出口: {outlet_count} 粒子")
                     dpg.set_value("avg_speed_text", f"平均速度: {avg_speed:.1f} mm/s")
                except:
                     pass
            
            # 録画フレーム保存
            if state.is_recording:
                # PyVistaのスクリーンショット機能を使用
                 timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                 filename = f"{state.screenshot_dir}/frame_{timestamp}_{state.frame_count:05d}.png"
                 plotter.screenshot(filename)
                 state.frame_count += 1
                 dpg.set_value("frame_count_text", f"Frames: {state.frame_count}")
            
            # 障害物描画 (簡易)
            vis_obstacles_name = "obstacles_vis"
            if state.show_obstacles and len(state.obstacles) > 0:
                 # 障害物の可視化用マルチブロックを作成（毎フレーム再作成は重いが一旦これで行く）
                 mb = pv.MultiBlock()
                 for obs in state.obstacles:
                     ox, oy, oz = obs['x']/SCALE, obs['y']/SCALE, obs['z']/SCALE
                     osize = obs['size']/SCALE
                     if obs['type'] == 'sphere':
                         mesh = pv.Sphere(radius=osize, center=(ox, oy, oz))
                     else:
                         mesh = pv.Box(bounds=(ox-osize, ox+osize, oy-osize, oy+osize, oz-osize, oz+osize))
                     mb.append(mesh)
                 
                 # 名前を指定して追加（上書き更新される）
                 plotter.add_mesh(mb, color="orange", opacity=0.5, name=vis_obstacles_name, reset_camera=False)
            elif not state.show_obstacles:
                 # 非表示の場合は削除
                 plotter.remove_actor(vis_obstacles_name)


        # 断面プロットなどはPyVistaのクリッピング機能を使うといいが、今回は省略または後で追加
        
    except Exception as e:
        print(f"Main loop error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            plotter.close()
        except:
            pass
        dpg.destroy_context()


if __name__ == "__main__":
    main()
