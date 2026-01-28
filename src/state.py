
import json
import os
from src.config import *
from src.config import DEFAULT_NUM_PARTICLES, MIN_NUM_PARTICLES, MAX_NUM_PARTICLES

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
        
        # シミュレーション制御
        self.is_paused = False
        self.sim_speed = 1.0  # 0.25, 0.5, 1.0, 2.0, 4.0
        
        # 追加ポート - 2番目のInlet/Outlet
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
        
        # 障害物
        self.obstacles = []  # リスト of {type, x, y, z, size}
        self.show_obstacles = True
        
        # 分析ツール
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
            'show_obstacles': self.show_obstacles
        }
    
    def from_dict(self, data):
        """辞書から設定を読み込み"""
        self.tank_width = data.get('tank_width', self.tank_width)
        self.tank_height = data.get('tank_height', self.tank_height)
        self.tank_depth = data.get('tank_depth', self.tank_depth)
        self.inlet_y_mm = data.get('inlet_y_mm', self.inlet_y_mm)
        self.inlet_z_mm = data.get('inlet_z_mm', self.inlet_z_mm)
        self.inlet_radius_mm = data.get('inlet_radius_mm', self.inlet_radius_mm)
        self.outlet_y_mm = data.get('outlet_y_mm', self.outlet_y_mm)
        self.outlet_z_mm = data.get('outlet_z_mm', self.outlet_z_mm)
        self.outlet_radius_mm = data.get('outlet_radius_mm', self.outlet_radius_mm)
        self.inlet_flow = data.get('inlet_flow', self.inlet_flow)
        self.outlet_flow = data.get('outlet_flow', self.outlet_flow)
        self.is_sync = data.get('is_sync', self.is_sync)
        self.target_num_particles = data.get('target_num_particles', self.target_num_particles)
        
        self.colormap_mode = data.get('colormap_mode', self.colormap_mode)
        self.background_color_mode = data.get('background_color_mode', self.background_color_mode)
        self.particle_size = data.get('particle_size', self.particle_size)
        self.show_tank_walls = data.get('show_tank_walls', self.show_tank_walls)
        
        self.sim_speed = data.get('sim_speed', self.sim_speed)
        
        self.use_second_inlet = data.get('use_second_inlet', self.use_second_inlet)
        self.inlet2_y_mm = data.get('inlet2_y_mm', self.inlet2_y_mm)
        self.inlet2_z_mm = data.get('inlet2_z_mm', self.inlet2_z_mm)
        self.inlet2_radius_mm = data.get('inlet2_radius_mm', self.inlet2_radius_mm)
        self.inlet2_flow = data.get('inlet2_flow', self.inlet2_flow)
        
        self.use_second_outlet = data.get('use_second_outlet', self.use_second_outlet)
        self.outlet2_y_mm = data.get('outlet2_y_mm', self.outlet2_y_mm)
        self.outlet2_z_mm = data.get('outlet2_z_mm', self.outlet2_z_mm)
        self.outlet2_radius_mm = data.get('outlet2_radius_mm', self.outlet2_radius_mm)
        self.outlet2_flow = data.get('outlet2_flow', self.outlet2_flow)
        
        self.obstacles = data.get('obstacles', self.obstacles)
        self.show_obstacles = data.get('show_obstacles', self.show_obstacles)
        
        self.needs_dimension_update = True
        self.needs_particle_count_update = True

    def save_to_file(self, filename="./config.json"):
        """設定をファイルに保存"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.to_dict(), f, indent=4)
            return True, f"設定を保存しました: {filename}"
        except Exception as e:
            return False, f"保存エラー: {e}"

    def load_from_file(self, filename="./config.json"):
        """設定をファイルから読み込み"""
        if not os.path.exists(filename):
            return False, "設定ファイルが見つかりません"
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.from_dict(data)
            return True, f"設定を読み込みました: {filename}"
        except Exception as e:
            return False, f"読み込みエラー: {e}"
