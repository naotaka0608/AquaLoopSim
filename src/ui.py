
import dearpygui.dearpygui as dpg
from src.config import MIN_NUM_PARTICLES, MAX_NUM_PARTICLES

def create_ui(state, callbacks=None):
    """Dear PyGui UIをセットアップ"""
    if callbacks is None:
        callbacks = {}
    
    dpg.create_context()
    
    # フォント設定
    with dpg.font_registry():
        font_path = "C:/Windows/Fonts/meiryo.ttc"
        try:
            with dpg.font(font_path, 20) as default_font:
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Japanese)
            dpg.bind_font(default_font)
        except Exception as e:
            print(f"Font loading failed: {e}")
    
    # テーマ設定
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

    # 内部ヘルパー関数: スライダーとインプットの同期 + State更新
    def sync_slider_to_input(sender, app_data, user_data):
        input_tag, attr_name = user_data
        dpg.set_value(input_tag, app_data)
        setattr(state, attr_name, app_data)

    def sync_input_to_slider(sender, app_data, user_data):
        slider_tag, attr_name = user_data
        dpg.set_value(slider_tag, app_data)
        setattr(state, attr_name, app_data)

    def create_labeled_slider_with_input(label, attr_name, default_val, min_val, max_val, format_str="%.1f"):
        """ラベル + スライダー + インプットボックスのセットを作成"""
        # タグの一意性を保証するため attr_name を使用
        slider_tag = f"{attr_name}_slider"
        input_tag = f"{attr_name}_input"
        label_tag = f"{attr_name}_label"
        
        dpg.add_text(label, tag=label_tag)
        with dpg.group(horizontal=True):
            dpg.add_slider_float(
                tag=slider_tag,
                default_value=default_val,
                min_value=min_val,
                max_value=max_val,
                width=280,
                format=format_str,
                callback=sync_slider_to_input,
                user_data=(input_tag, attr_name)
            )
            dpg.add_input_float(
                tag=input_tag,
                default_value=default_val,
                width=90,
                step=0,
                callback=sync_input_to_slider,
                user_data=(slider_tag, attr_name)
            )
        dpg.add_spacer(height=3)

    def update_inlet_ui_labels():
        face = state.inlet_face
        # 0:Left(X-), 1:Right(X+), 2:Bottom(Y-), 3:Top(Y+)
        # If X face (0,1): Pos1=Y, Pos2=Z
        # If Y face (2,3): Pos1=X, Pos2=Z
        if face in [0, 1]:
            dpg.configure_item("inlet_y_mm_label", label="Y位置 (mm)")
            dpg.configure_item("inlet_z_mm_label", label="Z位置 (mm)")
        else:
            dpg.configure_item("inlet_y_mm_label", label="X位置 (mm)") # mapped to inlet_y_mm var but represents X
            dpg.configure_item("inlet_z_mm_label", label="Z位置 (mm)")

    def update_outlet_ui_labels():
        face = state.outlet_face
        if face in [0, 1]:
            dpg.configure_item("outlet_y_mm_label", label="Y位置 (mm)")
            dpg.configure_item("outlet_z_mm_label", label="Z位置 (mm)")
        else:
            dpg.configure_item("outlet_y_mm_label", label="X位置 (mm)")
            dpg.configure_item("outlet_z_mm_label", label="Z位置 (mm)")

    # 内部コールバック
    def on_reset_particles(sender=None, app_data=None, user_data=None):
        state.needs_particle_reset = True
        dpg.set_value("config_status_text", "粒子をリセットしました")

    def toggle_pause(sender=None, app_data=None, user_data=None):
        state.is_paused = not state.is_paused
        status = "Paused" if state.is_paused else "Playing"
        dpg.set_value("pause_status_text", status)
        label = "Resume" if state.is_paused else "Pause"
        dpg.configure_item("pause_button", label=label)

    def on_save_config(sender=None, app_data=None, user_data=None):
        success, msg = state.save_to_file()
        dpg.set_value("config_status_text", msg)

    def on_load_config(sender=None, app_data=None, user_data=None):
        success, msg = state.load_from_file()
        dpg.set_value("config_status_text", msg)
        if success:
            # 簡易UI更新: Slider等をUpdate
            # 本来は全スライダーに set_value が必要だが、主要なものだけ
            items_to_update = {
                "tank_width_slider": state.tank_width,
                "tank_height_slider": state.tank_height,
                "tank_depth_slider": state.tank_depth,
                "inlet_y_slider": state.inlet_y_mm,
                "inlet_z_slider": state.inlet_z_mm,
                "inlet_radius_slider": state.inlet_radius_mm,
                "inlet_flow_slider": state.inlet_flow,
                "outlet_y_slider": state.outlet_y_mm,
                "outlet_z_slider": state.outlet_z_mm,
                "outlet_radius_slider": state.outlet_radius_mm,
                "outlet_flow_slider": state.outlet_flow,
                "particle_slider": state.target_num_particles,
                "particle_size_slider": state.particle_size,
                 # Inputも更新
                "tank_width_input": state.tank_width,
                "tank_height_input": state.tank_height,
                "tank_depth_input": state.tank_depth,
                # ... 他多数
            }
            try:
                for tag, val in items_to_update.items():
                    if dpg.does_item_exist(tag):
                        dpg.set_value(tag, val)
                
                dpg.set_value("bg_color_combo", state.background_color_mode)
                
                # Face Combo update
                faces = ["0:左(X-)", "1:右(X+)", "2:底(Y-)", "3:上(Y+)"]
                dpg.set_value("inlet_face_combo", faces[state.inlet_face])
                dpg.set_value("outlet_face_combo", faces[state.outlet_face])
                update_inlet_ui_labels()
                update_outlet_ui_labels()
                
                if state.is_paused != (dpg.get_item_label("pause_button") == "Resume"):
                    toggle_pause()
            except:
                pass


    def on_apply_dimensions(sender=None, app_data=None, user_data=None):
        state.needs_dimension_update = True
        
    def on_apply_particles(sender=None, app_data=None, user_data=None):
        state.needs_particle_count_update = True

    # 外部コールバックラッパー
    def call_cb(name):
        if name in callbacks:
            callbacks[name]()

    # Obstacle helper
    def update_obstacles_list():
        if dpg.does_item_exist("obstacles_list"):
            dpg.delete_item("obstacles_list")
        with dpg.group(tag="obstacles_list", parent="obstacles_container"):
            for i, obs in enumerate(state.obstacles):
                with dpg.group(horizontal=True):
                    dpg.add_text(f"#{i+1} {obs['type']} size={obs['size']}")
                    dpg.add_button(label="x", width=20, callback=lambda s, a, idx=i: remove_obstacle(idx))

    def add_obstacle(sender=None, app_data=None, user_data=None, type_str=None):
        if type_str is None:
             type_str = 'sphere' if dpg.get_value("obstacle_type_combo") == "球" else 'box'
        try:
            x = dpg.get_value("obs_x_slider")
            y = dpg.get_value("obs_y_slider")
            z = dpg.get_value("obs_z_slider")
            size = dpg.get_value("obs_size_slider")
        except:
            x, y, z, size = 500, 250, 500, 50
        
        new_obs = {'type': type_str, 'x': x, 'y': y, 'z': z, 'size': size}
        state.obstacles.append(new_obs)
        update_obstacles_list()
        dpg.set_value("obstacle_count_text", f"配置済み障害物: {len(state.obstacles)}個")

    def remove_obstacle(idx):
        # idx is passed via lambda, so normal signature is fine if called correctly
        if 0 <= idx < len(state.obstacles):
            state.obstacles.pop(idx)
            update_obstacles_list()
            dpg.set_value("obstacle_count_text", f"配置済み障害物: {len(state.obstacles)}個")
            
    def clear_obstacles(sender=None, app_data=None, user_data=None):
        state.obstacles.clear()
        update_obstacles_list()
        dpg.set_value("obstacle_count_text", f"配置済み障害物: 0個")


    # 各セクション作成関数
    def _create_tank_section():
        """水槽寸法セクション"""
        with dpg.collapsing_header(label="水槽設定", default_open=False):
            dpg.add_spacer(height=5)
            # 以前はスライダー+Inputだった
            create_labeled_slider_with_input("幅 (mm)", "tank_width", state.tank_width, 500.0, 2000.0)
            create_labeled_slider_with_input("高さ (mm)", "tank_height", state.tank_height, 200.0, 1000.0)
            create_labeled_slider_with_input("奥行 (mm)", "tank_depth", state.tank_depth, 500.0, 2000.0)
            
            dpg.add_spacer(height=5)
            dpg.add_button(label="寸法を適用", callback=on_apply_dimensions, width=150)
            dpg.add_checkbox(label="水槽の壁を表示", default_value=state.show_tank_walls,
                            callback=lambda s, a: setattr(state, 'show_tank_walls', a) or setattr(state, 'needs_dimension_update', True))
            dpg.add_spacer(height=5)
        dpg.add_separator()

    def _create_inlet_section():
        """流入口設定セクション"""
        with dpg.collapsing_header(label="流入口設定 (Inlet)", default_open=False):
            dpg.add_spacer(height=5)
            
            faces = ["0:左(X-)", "1:右(X+)", "2:底(Y-)", "3:上(Y+)"]
            def set_inlet_face(s, a, u):
                idx = faces.index(a)
                state.inlet_face = idx
                update_inlet_ui_labels()
            
            dpg.add_text("設置面 (Face)")
            dpg.add_combo(items=faces, tag="inlet_face_combo", default_value=faces[state.inlet_face],
                         callback=set_inlet_face, width=200)

            # 以前は slider + input
            create_labeled_slider_with_input("Y位置 (mm)", "inlet_y_mm", state.inlet_y_mm, 20.0, 1000.0) # Expanded max
            create_labeled_slider_with_input("Z位置 (mm)", "inlet_z_mm", state.inlet_z_mm, 20.0, 1000.0) # Expanded max
            create_labeled_slider_with_input("半径 (mm)", "inlet_radius_mm", state.inlet_radius_mm, 20.0, 150.0)
            create_labeled_slider_with_input("流入量", "inlet_flow", state.inlet_flow, 0.0, 2000.0)
            dpg.add_spacer(height=5)
            
            # 初期ラベル更新
            update_inlet_ui_labels()
            
        dpg.add_separator()

    def _create_outlet_section():
        """流出口設定セクション"""
        with dpg.collapsing_header(label="流出口設定 (Outlet)", default_open=False):
            dpg.add_spacer(height=5)
            dpg.add_checkbox(label="流入口と同期", tag="sync_checkbox", default_value=state.is_sync,
                           callback=lambda s, a: setattr(state, 'is_sync', a))

            faces = ["0:左(X-)", "1:右(X+)", "2:底(Y-)", "3:上(Y+)"]
            def set_outlet_face(s, a, u):
                idx = faces.index(a)
                state.outlet_face = idx
                update_outlet_ui_labels()

            dpg.add_text("設置面 (Face)")
            dpg.add_combo(items=faces, tag="outlet_face_combo", default_value=faces[state.outlet_face],
                         callback=set_outlet_face, width=200)

            create_labeled_slider_with_input("Y位置 (mm)", "outlet_y_mm", state.outlet_y_mm, 20.0, 1000.0)
            create_labeled_slider_with_input("Z位置 (mm)", "outlet_z_mm", state.outlet_z_mm, 20.0, 1000.0)
            create_labeled_slider_with_input("半径 (mm)", "outlet_radius_mm", state.outlet_radius_mm, 20.0, 150.0)
            create_labeled_slider_with_input("流出量", "outlet_flow", state.outlet_flow, 0.0, 2000.0)
            dpg.add_spacer(height=5)
            
            # 初期ラベル更新
            update_outlet_ui_labels()
        dpg.add_separator()

    def _create_particle_section():
        """粒子設定セクション"""
        with dpg.collapsing_header(label="粒子設定", default_open=False):
            dpg.add_spacer(height=5)
            
            dpg.add_text("粒子数", indent=10)
            with dpg.group(horizontal=True):
                # 粒子設定だけ特殊（Int slider）
                dpg.add_slider_int(
                    tag="particle_slider",
                    default_value=state.target_num_particles,
                    min_value=MIN_NUM_PARTICLES,
                    max_value=MAX_NUM_PARTICLES,
                    width=180,
                    callback=lambda s, a: (dpg.set_value("particle_input", a), setattr(state, 'target_num_particles', a))
                )
                dpg.add_input_int(
                    tag="particle_input",
                    default_value=state.target_num_particles,
                    width=70,
                    step=0,
                    callback=lambda s, a: (dpg.set_value("particle_slider", a), setattr(state, 'target_num_particles', a))
                )
            
            dpg.add_text(f"現在: {state.current_num_particles:,}", tag="particle_count_text", indent=10)
            dpg.add_spacer(height=5)
            dpg.add_button(label="粒子数を適用", callback=on_apply_particles, width=120)
            dpg.add_spacer(height=5)
        dpg.add_separator()

    def _create_visualization_section():
        """視覚化設定セクション"""
        with dpg.collapsing_header(label="視覚化設定", default_open=False):
            dpg.add_spacer(height=5)
            # カラーマップ
            dpg.add_text("カラーマップ", indent=10)
            dpg.add_combo(
                items=["青→赤", "レインボー", "クールウォーム", "Viridis"],
                default_value="青→赤",
                width=200,
                callback=lambda s, a: setattr(state, 'colormap_mode', ["青→赤", "レインボー", "クールウォーム", "Viridis"].index(a)),
                indent=10
            )
            dpg.add_spacer(height=5)
            # 背景色選択
            dpg.add_text("背景色", indent=10)
            dpg.add_combo(
                items=["Black", "Dark Gray", "Light Gray", "White", "Paraview Blue"],
                tag="bg_color_combo",
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
        dpg.add_separator()

    def _create_simulation_control_section():
        """シミュレーション制御セクション"""
        with dpg.collapsing_header(label="シミュレーション制御", default_open=True):
            dpg.add_spacer(height=5)
            # 一時停止/再生ボタン + リセットボタン
            with dpg.group(horizontal=True):
                dpg.add_button(label="Pause", tag="pause_button", callback=lambda: toggle_pause(), width=100)
                dpg.add_spacer(width=10)
                dpg.add_button(label="Reset", callback=on_reset_particles, width=100)
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
            
            # 設定保存 / 読み込み (経過時間の下に配置)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Save", callback=lambda: on_save_config(), width=100, indent=10)
                dpg.add_spacer(width=10)
                dpg.add_button(label="Load", callback=lambda: on_load_config(), width=100)
            dpg.add_text("", tag="config_status_text", indent=10)
            dpg.add_spacer(height=5)

        dpg.add_separator()

    def _create_additional_ports_section():
        """追加ポートセクション"""
        with dpg.collapsing_header(label="追加ポート設定", default_open=False):
            dpg.add_checkbox(label="追加流入口 (Inlet 2)", default_value=state.use_second_inlet,
                           callback=lambda s, a: setattr(state, 'use_second_inlet', a))
            with dpg.group(tag="inlet2_group"):
                create_labeled_slider_with_input("Y位置", "inlet2_y_mm", state.inlet2_y_mm, 20.0, 450.0)
                create_labeled_slider_with_input("Z位置", "inlet2_z_mm", state.inlet2_z_mm, 20.0, 980.0)
                create_labeled_slider_with_input("半径", "inlet2_radius_mm", state.inlet2_radius_mm, 20.0, 100.0)
                create_labeled_slider_with_input("流入量", "inlet2_flow", state.inlet2_flow, 0.0, 1000.0)
            
            dpg.add_spacer(height=5)
            dpg.add_checkbox(label="追加流出口 (Outlet 2)", default_value=state.use_second_outlet,
                           callback=lambda s, a: setattr(state, 'use_second_outlet', a))
            with dpg.group(tag="outlet2_group"):
                create_labeled_slider_with_input("Y位置", "outlet2_y_mm", state.outlet2_y_mm, 20.0, 450.0)
                create_labeled_slider_with_input("Z位置", "outlet2_z_mm", state.outlet2_z_mm, 20.0, 980.0)
                create_labeled_slider_with_input("半径", "outlet2_radius_mm", state.outlet2_radius_mm, 20.0, 100.0)
                create_labeled_slider_with_input("流出量", "outlet2_flow", state.outlet2_flow, 0.0, 1000.0)
            dpg.add_spacer(height=5)
        dpg.add_separator()

    def _create_obstacles_section():
        """障害物設定セクション"""
        with dpg.collapsing_header(label="障害物設定", default_open=False):
            dpg.add_checkbox(label="障害物を表示", default_value=state.show_obstacles,
                           callback=lambda s, a: setattr(state, 'show_obstacles', a))
            with dpg.group(horizontal=True):
                dpg.add_combo(items=["球", "箱"], tag="obstacle_type_combo", default_value="球", width=80)
                dpg.add_spacer(width=5)
            
            # create_labeled_slider_with_input を使用（ただしTag管理が必要）
            # helper関数内では tag = f"{attr_name}_slider" となっているので注意
            # ここでは state に obs_... という名前の属性はないので、一時的な変数名を使うか、
            # 単純にヘルパーを使わずに実装を手書きで綺麗にするか。
            # 今回はヘルパーを使いたいが、setattr(state, ...) が走ってしまうのが問題（obstacle追加前なのでStateに保存する場所がない）
            # なので、Input+Sliderの構成を手書きで統一する。
            
            def add_slider_input_pair(label, tag_base, default_val, min_val, max_val, width=280):
                 with dpg.group(horizontal=True):
                    dpg.add_text(label)
                    dpg.add_slider_float(tag=f"{tag_base}_slider", default_value=default_val, min_value=min_val, max_value=max_val, width=width,
                                         callback=lambda s, a, u=f"{tag_base}_input": dpg.set_value(u, a))
                    dpg.add_input_float(tag=f"{tag_base}_input", default_value=default_val, width=90, step=0,
                                         callback=lambda s, a, u=f"{tag_base}_slider": dpg.set_value(u, a))

            add_slider_input_pair("X(mm)", "obs_x", 500.0, 0.0, 1000.0)
            add_slider_input_pair("Y(mm)", "obs_y", 250.0, 0.0, 500.0)
            add_slider_input_pair("Z(mm)", "obs_z", 500.0, 0.0, 1000.0)
            add_slider_input_pair("サイズ", "obs_size", 50.0, 10.0, 200.0)

            dpg.add_spacer(height=5)
            with dpg.group(horizontal=True):
                dpg.add_button(label="追加", callback=lambda: add_obstacle())
                dpg.add_spacer(width=5)
                dpg.add_button(label="全削除", callback=lambda: clear_obstacles())
                
            dpg.add_group(tag="obstacles_container")
            update_obstacles_list()

    def _create_analysis_section():
        """分析ツールセクション"""
        with dpg.collapsing_header(label="分析ツール", default_open=False):
            dpg.add_checkbox(label="流量計を表示", default_value=state.show_flow_meter,
                           callback=lambda s, a: setattr(state, 'show_flow_meter', a))
            if state.show_flow_meter:
                dpg.add_text("流入口: 0 粒子", tag="inlet_flow_text", indent=10)
                dpg.add_text("流出口: 0 粒子", tag="outlet_flow_text", indent=10)
                dpg.add_text("平均速度: 0.0 mm/s", tag="avg_speed_text", indent=10)
            
            dpg.add_spacer(height=5)
            dpg.add_checkbox(label="断面ビュー有効", default_value=state.show_cross_section,
                           callback=lambda s, a: setattr(state, 'show_cross_section', a))
            
            dpg.add_radio_button(items=["X軸", "Y軸", "Z軸"], default_value="X軸", horizontal=True,
                               callback=lambda s, a: setattr(state, 'cross_section_axis', a[0])) 
            
            dpg.add_slider_float(label="位置 (%)", default_value=50.0, min_value=0.0, max_value=100.0,
                               callback=lambda s, a: setattr(state, 'cross_section_pos', a))
            
            dpg.add_spacer(height=5)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Screenshot", callback=lambda: call_cb('take_screenshot'), width=100)
                dpg.add_button(label="Start Record", tag="record_button", callback=lambda: call_cb('toggle_recording'), width=100)
            
            dpg.add_text("Save to: ./screenshots", tag="save_path_text")
            dpg.add_text(f"Frames: {state.frame_count}", tag="frame_count_text")
    
    # メインウィンドウ構築
    with dpg.window(label="Control Panel", tag="main_window", width=500, height=800, no_close=True):
        _create_simulation_control_section()
        _create_tank_section()
        _create_inlet_section()
        _create_outlet_section()
        _create_particle_section()
        _create_visualization_section()
        _create_additional_ports_section()
        _create_obstacles_section()
        _create_analysis_section()
    
    dpg.create_viewport(title='Fluid Simulation', width=520, height=1010, x_pos=50, y_pos=10)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
