import taichi as ti
import numpy as np
import dearpygui.dearpygui as dpg
import os
import json
import datetime
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
        dpg.add_button(label="Save Config", callback=lambda: on_save_config(), width=150)
        dpg.add_spacer(width=10)
        dpg.add_button(label="Load Config", callback=lambda: on_load_config(), width=150)
    dpg.add_text("", tag="config_status_text")
    dpg.add_spacer(height=5)
    dpg.add_separator()


def _create_tank_section():
    """タンク寸法セクション"""
    with dpg.collapsing_header(label="タンク寸法", default_open=True):
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
    with dpg.collapsing_header(label="流入口 (Inlet)", default_open=True):
        dpg.add_spacer(height=5)
        create_labeled_slider_with_input("Y位置 (mm)", "inlet_y", state.inlet_y_mm, 20.0, 450.0)
        create_labeled_slider_with_input("Z位置 (mm)", "inlet_z", state.inlet_z_mm, 20.0, 980.0)
        create_labeled_slider_with_input("半径 (mm)", "inlet_radius", state.inlet_radius_mm, 20.0, 150.0)
        create_labeled_slider_with_input("流量 (L/min)", "inlet_flow", state.inlet_flow, 0.0, 2000.0)
        dpg.add_spacer(height=10)
    dpg.add_separator()


def _create_outlet_section():
    """流出口セクション"""
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


def _create_particle_section():
    """粒子設定セクション"""
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


def _create_controls_section():
    """操作説明セクション"""
    with dpg.collapsing_header(label="操作方法", default_open=False):
        dpg.add_text("右クリック+ドラッグ: 視点回転", indent=10)
        dpg.add_text("Shift+右クリック: ズーム", indent=10)
        dpg.add_text("中クリック+ドラッグ: パン", indent=10)
    dpg.add_separator()


def _create_visualization_section():
    """視覚化設定セクション"""
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


def _create_simulation_control_section():
    """シミュレーション制御セクション"""
    with dpg.collapsing_header(label="シミュレーション制御", default_open=True):
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
    with dpg.collapsing_header(label="分析ツール", default_open=True):
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
        _create_controls_section()
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


def calculate_flow_stats(solver, res_x, res_y, res_z):
    """流量統計を計算"""
    positions = solver.particle_pos.to_numpy()
    velocities = solver.particle_vel.to_numpy()
    
    # 入口近くの粒子数（X < 5）
    inlet_mask = positions[:, 0] < 5
    inlet_count = np.sum(inlet_mask)
    
    # 出口近くの粒子数（X < 0）
    outlet_mask = positions[:, 0] < 0
    outlet_count = np.sum(outlet_mask)
    
    # 平均速度
    speeds = np.linalg.norm(velocities, axis=1)
    avg_speed = np.mean(speeds) * SCALE  # mm/s に変換
    
    return inlet_count, outlet_count, avg_speed

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
    
    # グローバル参照を設定
    global _window_ref
    _window_ref = window
    
    # 流量計用のカウンター
    flow_update_counter = 0
    
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
            state.sim_elapsed_time = 0.0  # 経過時間もリセット
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
            # 経過時間を更新（1フレーム = 約1/60秒として概算）
            state.sim_elapsed_time += (1.0 / 60.0) * state.sim_speed
        
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
        
        # 流量計更新（10フレームごと）
        flow_update_counter += 1
        if state.show_flow_meter and flow_update_counter >= 10:
            flow_update_counter = 0
            inlet_count, outlet_count, avg_speed = calculate_flow_stats(solver, res_x, res_y, res_z)
            try:
                dpg.set_value("inlet_flow_text", f"流入口: {inlet_count} 粒子")
                dpg.set_value("outlet_flow_text", f"流出口: {outlet_count} 粒子")
                dpg.set_value("avg_speed_text", f"平均速度: {avg_speed:.1f} mm/s")
            except:
                pass
        
        # 録画フレーム保存
        if state.is_recording:
            save_recording_frame()

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
        
        # 断面ビュー描画
        if state.show_cross_section:
            # 断面位置を計算
            if state.cross_section_axis == 'X':
                pos = res_x * state.cross_section_pos / 100.0
                # YZ平面の矩形
                section_verts = np.array([
                    [pos, 0, 0],
                    [pos, res_y, 0],
                    [pos, res_y, res_z],
                    [pos, 0, res_z],
                    [pos, 0, 0]
                ], dtype=np.float32)
            elif state.cross_section_axis == 'Y':
                pos = res_y * state.cross_section_pos / 100.0
                # XZ平面の矩形
                section_verts = np.array([
                    [0, pos, 0],
                    [res_x, pos, 0],
                    [res_x, pos, res_z],
                    [0, pos, res_z],
                    [0, pos, 0]
                ], dtype=np.float32)
            else:  # Z
                pos = res_z * state.cross_section_pos / 100.0
                # XY平面の矩形
                section_verts = np.array([
                    [0, 0, pos],
                    [res_x, 0, pos],
                    [res_x, res_y, pos],
                    [0, res_y, pos],
                    [0, 0, pos]
                ], dtype=np.float32)
            
            section_field = ti.Vector.field(3, dtype=float, shape=4)
            section_field.from_numpy(section_verts[:4])
            section_indices = ti.field(dtype=int, shape=8)
            # 閉じた矩形: 0-1, 1-2, 2-3, 3-0
            section_indices[0] = 0
            section_indices[1] = 1
            section_indices[2] = 1
            section_indices[3] = 2
            section_indices[4] = 2
            section_indices[5] = 3
            section_indices[6] = 3
            section_indices[7] = 0
            scene.lines(section_field, indices=section_indices, color=(1.0, 1.0, 0.0), width=3.0)
        
        canvas.scene(scene)
        window.show()
    
    dpg.destroy_context()

if __name__ == "__main__":
    main()
