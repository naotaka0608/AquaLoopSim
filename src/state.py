from .config import DEFAULT_NUM_PARTICLES

class AppState:
    def __init__(self):
        self.tank_width = 1000.0
        self.tank_height = 500.0
        self.tank_depth = 1000.0
        
        # 0=Left(X-), 1=Right(X+), 2=Bottom(Y-), 3=Top(Y+)
        self.inlet_face = 0
        self.outlet_face = 1
        
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
        self.needs_device_update = False
        
        # GPU Settings
        self.compute_device = "CPU"
        
        # 視覚化設定
        self.colormap_mode = 0  # 0=Blue-Red, 1=Rainbow, 2=Cool-Warm, 3=Viridis
        self.background_color_mode = "Black" # Black, Dark Gray, Light Gray, White, Paraview Blue
        self.particle_size = 8.0 # PyVista uses point size, not radius
        self.show_trails = False
        self.show_tank_walls = True
        self.viz_mode = "Particles" # Particles, Streamlines, Both
        self.streamline_radius = 0.5
        self.streamline_count = 50
        
        # シミュレーション制御
        self.is_paused = False
        self.sim_speed = 1.0
        
        # 追加ポート
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
        self.show_flow_meter = True
        self.inlet_particle_count = 0
        self.outlet_particle_count = 0
        
        self.show_cross_section = False
        self.cross_section_axis = 'X'
        self.cross_section_pos = 50.0
        
        # スクリーンショット/録画
        self.screenshot_dir = "./screenshots"
        self.is_recording = False
        self.recording_frames = []
        self.frame_count = 0
        
        # シミュレーション経過時間
        self.sim_elapsed_time = 0.0
    
    def to_dict(self):
        """保存用の辞書に変換"""
        return {
            'tank_width': self.tank_width,
            'tank_height': self.tank_height,
            'tank_depth': self.tank_depth,
            'inlet_face': self.inlet_face,
            'outlet_face': self.outlet_face,
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
            'compute_device': self.compute_device,
            'viz_mode': self.viz_mode,
            'streamline_radius': self.streamline_radius,
            'streamline_count': self.streamline_count,
        }
    
    def from_dict(self, data):
        """辞書からパラメータを読み込み"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
